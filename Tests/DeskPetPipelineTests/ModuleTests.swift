import Foundation
import Testing
@testable import DeskPetPipeline

private func eventually(_ predicate: @escaping @Sendable () async -> Bool) async throws {
    let end = ContinuousClock.now.advanced(by: .seconds(2))
    while !(await predicate()) {
        guard ContinuousClock.now < end else { throw PipelineError("TEST_TIMEOUT", stage: "module") }
        try await Task.sleep(for: .milliseconds(1))
    }
}
private actor ModuleProbe {
    var submitted: [Int] = []
    var processed: [Int] = []
    var release = false
    func execute(_ input: Int) -> Int { submitted.append(input); return input * 10 }
    func process(_ output: Int) async {
        if output == 10 { while !release { try? await Task.sleep(for: .milliseconds(1)) } }
        processed.append(output)
    }
    func unblock() { release = true }
}
private final class GPUThreadProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var identities: Set<ObjectIdentifier> = []
    private var usedMain = false
    func observe() { lock.withLock { identities.insert(ObjectIdentifier(Thread.current)); usedMain = usedMain || Thread.isMainThread } }
    var count: Int { lock.withLock { identities.count } }
    var isMain: Bool { lock.withLock { usedMain } }
}
private final class GPUBarrier: @unchecked Sendable {
    private let lock = NSLock()
    private var released = false
    func release() { lock.withLock { released = true } }
    var ready: Bool { lock.withLock { released } }
}
private actor BlockingStreamingSTT: StreamingSpeechRecognizer {
    private var released = false
    private let model = MockStreamingSTT()
    func unblock() { released = true }
    func transcribe(_ segment: AudioSegment) async throws -> Transcript { try await model.transcribe(segment) }
    func accept(_ input: STTFrame) async throws -> Transcript? {
        while !released { try await Task.sleep(for: .milliseconds(1)) }
        return try await model.accept(input)
    }
    func cancel(sessionId: UUID) async { await model.cancel(sessionId: sessionId) }
}
private actor StreamingProbe: StreamingSpeechRecognizer {
    var frames: [VoiceActivity] = []
    private let model = MockStreamingSTT()
    func transcribe(_ segment: AudioSegment) async throws -> Transcript { try await model.transcribe(segment) }
    func accept(_ input: STTFrame) async throws -> Transcript? {
        frames.append(input.activity); return try await model.accept(input)
    }
    func cancel(sessionId: UUID) async { await model.cancel(sessionId: sessionId) }
}
private actor MicrophoneProbe: MicrophoneSource {
    private var frames: [AudioFrame]
    init(_ frames: [AudioFrame]) { self.frames = frames }
    func nextFrame() -> AudioFrame? { frames.isEmpty ? nil : frames.removeFirst() }
}
private struct SpeechVAD: VoiceActivityDetector {
    func classify(_ frame: AudioFrame) -> VoiceActivity { frame.chunk.data.isEmpty ? .end : .speech }
}
private actor PlaybackBarrier: SpeakerAdapter {
    var playing = false
    var release = false
    func begin(jobId: UUID, event: OutputEvent) { playing = true }
    func write(_ chunk: PCMChunk, jobId: UUID) {}
    func finish(jobId: UUID) async throws { while !release { try await Task.sleep(for: .milliseconds(1)) } }
    func stop(jobId: UUID) { release = true }
    func unblock() { release = true }
}

struct ModuleTests {
    @Test func boundedSenderWaitsAndCancellationDoesNotEnqueuePayload() async throws {
        let queue = Mailbox<Int>(capacity: 1)
        try await queue.offer(1)
        let second = Task { try await queue.send(2) }
        #expect(await queue.receive() == 1)
        try await second.value; #expect(await queue.receive() == 2)
        try await queue.offer(3)
        let cancelled = Task { try await queue.send(4) }; cancelled.cancel()
        await #expect(throws: CancellationError.self) { try await cancelled.value }
        #expect(await queue.receive() == 3); #expect(await queue.count == 0)
        await queue.close()
    }
    @Test func submissionAndCompletionAreIndependentSubmodules() async throws {
        let input = Mailbox<Int>(capacity: 2); let probe = ModuleProbe()
        let module = PipelineModule(resource: probe, input: input, limit: 2,
            execute: { value, resource in await resource.execute(value) },
            process: { _, result in if let value = try? result.get() { await probe.process(value) } })
        await module.start(); try await input.offer(1)
        try await eventually { await module.snapshot().processingCompletions == 1 }
        try await input.offer(2)
        try await eventually { await probe.submitted == [1, 2] }
        #expect(await probe.processed.isEmpty)
        #expect(await module.snapshot().completedQueue == 1)
        await probe.unblock()
        try await eventually { await probe.processed == [10, 20] }
        await module.stop(); #expect(await module.snapshot().inFlight == 0)
    }

    @Test func gpuSubmitsAndReturnsDifferentOperationsOnOneThread() async throws {
        let gpu = try GPUExecutor(); let probe = GPUThreadProbe(); let barrier = GPUBarrier()
        let slow = Task {
            try await gpu.run(GPUOperation<Int> {
                probe.observe()
                return GPUCompletion { probe.observe(); return barrier.ready ? .success(11) : nil }
            })
        }
        try await eventually { gpu.snapshot().submitted == 1 }
        let fast = try await gpu.run(GPUOperation<String> {
            probe.observe(); return GPUCompletion { probe.observe(); return .success("second") }
        })
        #expect(fast == "second")
        try await eventually { gpu.snapshot().completed == 1 }
        #expect(gpu.snapshot().outstanding == 1)
        barrier.release()
        #expect(try await slow.value == 11)
        await gpu.shutdown()
        #expect(probe.count == 1); #expect(!probe.isMain)
        #expect(gpu.snapshot().submitted == 2); #expect(gpu.snapshot().completed == 2)
    }

    @Test func gpuCancellationRetainsResourcesUntilRuntimeCompletion() async throws {
        let gpu = try GPUExecutor(capacity: 1); let barrier = GPUBarrier()
        let slow = Task { try await gpu.run(GPUOperation<Int> {
            GPUCompletion { barrier.ready ? .success(9) : nil }
        }) }
        try await eventually { gpu.snapshot().submitted == 1 }
        slow.cancel(); await #expect(throws: CancellationError.self) { _ = try await slow.value }
        #expect(gpu.snapshot().outstanding == 1)
        do {
            _ = try await gpu.run(MockGPUCall(value: 8).operation)
            Issue.record("Expected GPU_BUSY")
        } catch let error as PipelineError { #expect(error.code == "GPU_BUSY") }
        barrier.release(); await gpu.shutdown(); #expect(gpu.snapshot().outstanding == 0)
        do {
            _ = try await gpu.run(MockGPUCall(value: 7).operation)
            Issue.record("Expected GPU_STOPPED")
        } catch let error as PipelineError { #expect(error.code == "GPU_STOPPED") }
    }

    @Test func gpuErrorDoesNotStopOtherRequests() async throws {
        let gpu = try GPUExecutor()
        do {
            _ = try await gpu.run(MockGPUCall<Int>(error: PipelineError("FIXTURE", stage: "gpu")).operation)
            Issue.record("Expected fixture error")
        } catch let error as PipelineError { #expect(error.code == "FIXTURE") }
        #expect(try await gpu.run(MockGPUCall(value: "ok").operation) == "ok")
        await gpu.shutdown()
    }

    @Test func vadFramesReachSTTBeforeEndAndDuringPlayback() async throws {
        let stt = StreamingProbe(); let speaker = PlaybackBarrier()
        var adapters = Adapters.mocks(); adapters.stt = stt; adapters.speaker = speaker
        let pipeline = try DeskPetPipeline(adapters: adapters)
        await pipeline.start(automaticReminders: false)
        _ = try await pipeline.submit(Envelope(conversationId: "old", kind: "stt.final", payload: Transcript(text: "hello")))
        try await eventually { await speaker.playing }
        #expect(await pipeline.isAudioInputAllowed())
        let session = try await pipeline.beginSpeech(conversationId: "stream")
        try await pipeline.submitSpeechFrame(sessionId: session,
            frame: AudioFrame(chunk: PCMChunk(data: Data("hel".utf8))), activity: .speech)
        try await eventually { await stt.frames == [.speech] }
        #expect(await pipeline.contextTurns(conversationId: "stream").isEmpty)
        try await pipeline.submitSpeechFrame(sessionId: session,
            frame: AudioFrame(chunk: PCMChunk(data: Data("lo".utf8))), activity: .speech)
        try await pipeline.submitSpeechFrame(sessionId: session,
            frame: AudioFrame(chunk: PCMChunk(data: Data())), activity: .end)
        try await eventually { await pipeline.contextTurns(conversationId: "stream").first?.content == "hello" }
        #expect(await stt.frames == [.speech, .speech, .end])
        await speaker.unblock(); await pipeline.shutdown()
    }

    @Test func ingressDoesNotWaitForSTTInference() async throws {
        let stt = BlockingStreamingSTT()
        var adapters = Adapters.mocks(); adapters.stt = stt
        let pipeline = try DeskPetPipeline(adapters: adapters)
        await pipeline.start(automaticReminders: false)
        let capture = AudioIngress(microphone: MicrophoneProbe([
            AudioFrame(chunk: PCMChunk(data: Data("hello".utf8))), AudioFrame(chunk: PCMChunk(data: Data()))
        ]), vad: SpeechVAD())
        try await capture.run(pipeline: pipeline, conversationId: "stream")
        // Capture has reached EOF while STT is still processing its first frame.
        #expect(await pipeline.contextTurns(conversationId: "stream").isEmpty)
        #expect(await pipeline.inFlightCount() == 1)
        await stt.unblock()
        try await eventually { await pipeline.inFlightCount() == 0 }
        #expect(await pipeline.contextTurns(conversationId: "stream").first?.content == "hello")
        await pipeline.shutdown()
    }
}
