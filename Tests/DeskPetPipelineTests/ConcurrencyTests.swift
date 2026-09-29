import Foundation
import Testing
@testable import DeskPetPipeline

private func finishRequests(_ pipeline: DeskPetPipeline) async throws {
    let deadline = ContinuousClock.now.advanced(by: .seconds(3))
    while await pipeline.inFlightCount() > 0 {
        guard ContinuousClock.now < deadline else { throw PipelineError("TEST_TIMEOUT", stage: "concurrency") }
        try await Task.sleep(for: .milliseconds(2))
    }
}
private actor PendingOutput: OutputModel {
    var started = 0
    func generate(_ request: OutputRequest) async throws -> String {
        started += 1
        try await Task.sleep(for: .seconds(5)); return request.requiredText
    }
}
private actor PendingRouter: RouterModel {
    var started = 0
    func resolveTarget(in transcript: Transcript) async throws -> TargetResolution { .omitted }
    func heads(for input: RouterInput) async throws -> HeadOutputs {
        started += 1; try await Task.sleep(for: .seconds(5)); return HeadOutputs(intent: .chat)
    }
    func confidence(for maskedHeads: HeadOutputs, missingSlots: [String]) async throws -> Double { 1 }
}
private actor PendingSTT: SpeechRecognizer {
    var started = 0
    func transcribe(_ segment: AudioSegment) async throws -> Transcript {
        started += 1; try await Task.sleep(for: .seconds(5)); return Transcript(text: "hello")
    }
}
private actor OutputLatch {
    var arrived = false
    var released = false
    func arrive() { arrived = true }
    func release() { released = true }
    func wait() async throws {
        while !released { try await Task.sleep(for: .milliseconds(2)) }
    }
}
private struct LatchedDisplay: DisplayAdapter {
    let latch: OutputLatch
    func show(_ event: OutputEvent) async throws { await latch.arrive(); try await latch.wait() }
    func idle() async throws {}
}
private struct LatchedWeb: WebConsumer {
    let latch: OutputLatch
    func publish(_ event: OutputEvent) async throws { await latch.arrive(); try await latch.wait() }
}
private struct LatchedSpeaker: SpeakerAdapter {
    let latch: OutputLatch
    func begin(jobId: UUID, event: OutputEvent) async throws {}
    func write(_ chunk: PCMChunk, jobId: UUID) async throws {}
    func finish(jobId: UUID) async throws { await latch.arrive(); try await latch.wait() }
    func stop(jobId: UUID) async {}
}
private func waitUntil(_ predicate: @escaping @Sendable () async -> Bool) async throws {
    let end = ContinuousClock.now.advanced(by: .seconds(1))
    while !(await predicate()) {
        guard ContinuousClock.now < end else { throw PipelineError("TEST_TIMEOUT", stage: "module overlap") }
        try await Task.sleep(for: .milliseconds(2))
    }
}
private func chat(_ pipeline: DeskPetPipeline, conversation: String) async throws -> UUID {
    try await pipeline.submit(Envelope(conversationId: conversation, kind: "stt.final", payload: Transcript(text: "hello")))
}
struct ConcurrencyTests {
    @Test func playbackQueueBackpressurePreservesBurstSpeechAndDisplayProgress() async throws {
        let latch = OutputLatch(); let display = MockDisplay()
        var adapters = Adapters.mocks(display: display); adapters.speaker = LatchedSpeaker(latch: latch)
        var configuration = PipelineConfiguration(); configuration.audioQueueCapacity = 1
        let pipeline = try DeskPetPipeline(adapters: adapters, configuration: configuration)
        await pipeline.start(automaticReminders: false)
        var ids: [UUID] = []
        for index in 0..<6 { ids.append(try await chat(pipeline, conversation: "burst-\(index)")) }
        try await waitUntil { await display.shown().count == 6 }
        #expect(await latch.arrived)
        #expect(await pipeline.inFlightCount() == 6)
        await latch.release(); try await finishRequests(pipeline)
        for id in ids { #expect(await pipeline.record(requestId: id)?.delivery?.audio == .completed) }
        await pipeline.shutdown()
    }
    @Test func routerSubmissionContinuesWhileResultsArePending() async throws {
        let router = PendingRouter(); var adapters = Adapters.mocks(); adapters.router = router
        let pipeline = try DeskPetPipeline(adapters: adapters)
        await pipeline.start(automaticReminders: false)
        _ = try await chat(pipeline, conversation: "a"); _ = try await chat(pipeline, conversation: "b")
        try await waitUntil { await router.started == 2 }
        await pipeline.shutdown()
    }
    @Test func outputSubmissionContinuesWhileResultsArePending() async throws {
        let output = PendingOutput(); var adapters = Adapters.mocks(); adapters.output = output
        let pipeline = try DeskPetPipeline(adapters: adapters)
        await pipeline.start(automaticReminders: false)
        _ = try await chat(pipeline, conversation: "a"); _ = try await chat(pipeline, conversation: "b")
        try await waitUntil { await output.started == 2 }
        await pipeline.shutdown()
    }
    @Test func pendingSTTDoesNotBlockOtherModules() async throws {
        let stt = PendingSTT(); let speaker = MockSpeaker()
        var adapters = Adapters.mocks(speaker: speaker); adapters.stt = stt
        let pipeline = try DeskPetPipeline(adapters: adapters)
        await pipeline.start(automaticReminders: false)
        _ = try await pipeline.submitAudio(AudioSegment(pcm: Data("a".utf8)), conversationId: "a")
        try await waitUntil { await stt.started == 1 }
        let id = try await chat(pipeline, conversation: "b")
        try await waitUntil { await pipeline.record(requestId: id)?.delivery?.delivered == true }
        #expect(await stt.started == 1)
        #expect(await speaker.spoken().count == 1)
        await pipeline.shutdown(); #expect(await pipeline.inFlightCount() == 0)
    }
    @Test func displayDelayDoesNotBlockPhysicalPlaybackOrContext() async throws {
        let latch = OutputLatch(); let speaker = MockSpeaker()
        var adapters = Adapters.mocks(speaker: speaker); adapters.display = LatchedDisplay(latch: latch)
        let pipeline = try DeskPetPipeline(adapters: adapters)
        await pipeline.start(automaticReminders: false)
        let request = try await chat(pipeline, conversation: "a")
        try await waitUntil { await speaker.spoken().count == 1 }
        #expect(await latch.arrived)
        await latch.release(); try await finishRequests(pipeline)
        #expect(await pipeline.record(requestId: request)?.delivery?.audio == .completed)
        await pipeline.shutdown()
    }
    @Test func webDelayDoesNotBlockVoiceOrLCDCompletion() async throws {
        let latch = OutputLatch(); let speaker = MockSpeaker()
        var adapters = Adapters.mocks(speaker: speaker); adapters.web = LatchedWeb(latch: latch)
        let pipeline = try DeskPetPipeline(adapters: adapters)
        await pipeline.start(automaticReminders: false)
        let request = try await chat(pipeline, conversation: "a")
        try await waitUntil { await latch.arrived }
        try await finishRequests(pipeline)
        #expect(await speaker.spoken().count == 1)
        #expect(await pipeline.record(requestId: request)?.delivery?.audio == .completed)
        await latch.release(); await pipeline.shutdown()
    }
    @Test func playbackWaitDoesNotBlockFollowingDisplay() async throws {
        let latch = OutputLatch(); let display = MockDisplay()
        var adapters = Adapters.mocks(display: display); adapters.speaker = LatchedSpeaker(latch: latch)
        let pipeline = try DeskPetPipeline(adapters: adapters)
        await pipeline.start(automaticReminders: false)
        _ = try await chat(pipeline, conversation: "a")
        try await waitUntil { await latch.arrived }
        _ = try await chat(pipeline, conversation: "b")
        try await waitUntil { await display.shown().count == 2 }
        #expect(await pipeline.inFlightCount() == 2)
        await latch.release(); try await finishRequests(pipeline); await pipeline.shutdown()
    }
    @Test func mailboxDeliversResultsExactlyOnce() async throws {
        let queue = Mailbox<Int>(capacity: 32)
        let consumers = (0..<1).map { _ in Task { () -> [Int] in
            var received: [Int] = []
            while let value = await queue.receive() { received.append(value) }
            return received
        } }
        for value in 0..<32 { try await queue.offer(value) }
        // Drain before close, which intentionally discards outstanding payloads on shutdown.
        try await waitUntil { await queue.count == 0 }
        await queue.close()
        var combined: [Int] = []
        for consumer in consumers { combined += await consumer.value }
        #expect(combined.sorted() == Array(0..<32))
    }
    @Test func mailboxReceiverCancellationDoesNotStealNextMessage() async throws {
        let queue = Mailbox<Int>(capacity: 1)
        let consumer = Task { await queue.receive() }; consumer.cancel()
        #expect(await consumer.value == nil)
        try await queue.offer(7); #expect(await queue.receive() == 7)
        await queue.close()
    }
    @Test func pcmQueueBackpressureAndCancellationReleaseProducer() async throws {
        let queue = PCMChannel(capacity: 1)
        try await queue.send(PCMChunk(data: Data([1])))
        let producer = Task { try await queue.send(PCMChunk(data: Data([2]))) }
        try await Task.sleep(for: .milliseconds(5))
        #expect(await queue.bufferedCount == 1)
        producer.cancel()
        await #expect(throws: CancellationError.self) { try await producer.value }
        await queue.finish(error: CancellationError())
    }
    @Test func gpuMocksCompleteWholePipeline() async throws {
        let gpu = try GPUExecutor()
        let pipeline = try DeskPetPipeline(adapters: .gpuMocks(executor: gpu, delay: .milliseconds(2)))
        await pipeline.start(automaticReminders: false)
        _ = try await pipeline.submitAudio(AudioSegment(pcm: Data("hello".utf8)), conversationId: "a")
        _ = try await chat(pipeline, conversation: "b")
        try await finishRequests(pipeline); await pipeline.shutdown()
        #expect(gpu.snapshot().outstanding == 0)
        #expect(gpu.snapshot().threadCount == 1)
        #expect(await pipeline.contextTurns(conversationId: "a").first?.content == "hello")
    }

}

extension ConcurrencyTests {
    @Test func voiceCompletionCommitsContextBeforeSlowFaceAck() async throws {
        let target = Target(entityId: "fixture", name: "업무")
        let latch = OutputLatch(); let speaker = MockSpeaker()
        var adapters = Adapters.mocks(router: MockRouter(scripts: ["완료": MockRoute(heads: HeadOutputs(intent: .complete), resolution: .unique(target))]),
            eureka: MockEureka(items: [ReminderItem(target: target, description: "업무 설명", dueAt: Date().addingTimeInterval(100))]), speaker: speaker)
        adapters.display = LatchedDisplay(latch: latch)
        let pipeline = try DeskPetPipeline(adapters: adapters)
        await pipeline.start(automaticReminders: false)
        _ = try await pipeline.submit(Envelope(conversationId: "a", kind: "stt.final", payload: Transcript(text: "완료")))
        try await waitUntil { await pipeline.contextTurns(conversationId: "a").count == 2 }
        #expect(await latch.arrived)
        #expect(await speaker.spoken().count == 1)
        await latch.release(); try await finishRequests(pipeline); await pipeline.shutdown()
    }
    @Test func shutdownJoinsGPUAndAllModules() async throws {
        let gpu = try GPUExecutor()
        let pipeline = try DeskPetPipeline(adapters: .gpuMocks(executor: gpu, delay: .milliseconds(50)))
        await pipeline.start(automaticReminders: false)
        _ = try await chat(pipeline, conversation: "a"); _ = try await chat(pipeline, conversation: "b")
        try await waitUntil { gpu.snapshot().outstanding > 0 }
        await pipeline.shutdown()
        let snapshot = await pipeline.snapshot()
        #expect(snapshot.activeRequests == 0); #expect(snapshot.apiRequests == 0)
        #expect(snapshot.queued.values.allSatisfy { $0 == 0 })
        #expect(snapshot.modules.values.allSatisfy { $0.inFlight == 0 && $0.processingCompletions == 0 })
        #expect(gpu.snapshot().outstanding == 0)
    }
}
