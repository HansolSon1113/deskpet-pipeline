import Foundation
import Testing
@testable import DeskPetPipeline

private actor LateRecognizer: StreamingSpeechRecognizer {
    var calls = 0
    func transcribe(_ segment: AudioSegment) async throws -> Transcript { Transcript(text: "late") }
    func accept(_ frame: STTFrame) async throws -> Transcript? {
        calls += 1
        // Intentionally noncooperative completion: emulate a native STT callback.
        await withCheckedContinuation { continuation in
            DispatchQueue.global().asyncAfter(deadline: .now() + 0.15) { continuation.resume() }
        }
        return Transcript(text: "late")
    }
    func cancel(sessionId: UUID) async {
        await withCheckedContinuation { continuation in
            DispatchQueue.global().asyncAfter(deadline: .now() + 0.25) { continuation.resume() }
        }
    }
}
struct InputEpochTests {
    @Test func invalidationRejectsLateFinalBeforeCancellationCompletes() async throws {
        let stt = LateRecognizer()
        let eureka = MockEureka()
        var adapters = Adapters.mocks(eureka: eureka); adapters.stt = stt
        let pipeline = try DeskPetPipeline(adapters: adapters)
        await pipeline.start(automaticReminders: false)
        await pipeline.invalidateInput(epoch: 1); await pipeline.resumeInput(epoch: 1)
        let session = try await pipeline.beginSpeech(conversationId: "epoch", inputEpoch: 1)
        try await pipeline.submitSpeechFrame(sessionId: session, frame: .init(chunk: .init(data: Data(repeating: 0, count: 640), sampleRate: 16000), inputEpoch: 1), activity: .speech)
        let deadline = ContinuousClock.now.advanced(by: .seconds(2))
        while await stt.calls == 0 {
            guard ContinuousClock.now < deadline else { throw PipelineError("TEST_TIMEOUT", stage: "test") }
            try await Task.sleep(for: .milliseconds(2))
        }
        await pipeline.invalidateInput(epoch: 2)
        #expect(await pipeline.contextTurns(conversationId: "epoch").isEmpty)
        #expect(await eureka.count() == 0)
        await #expect(throws: PipelineError.self) { _ = try await pipeline.beginSpeech(conversationId: "epoch", inputEpoch: 1) }
        await pipeline.resumeInput(epoch: 1)
        #expect(await !pipeline.acceptsInput(epoch: 2))
        await pipeline.resumeInput(epoch: 2)
        #expect(await pipeline.acceptsInput(epoch: 2))
        #expect(await !pipeline.acceptsInput(epoch: 1))
        await pipeline.shutdown()
    }
}
