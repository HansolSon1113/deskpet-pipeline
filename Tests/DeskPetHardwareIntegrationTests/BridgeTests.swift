import Foundation
import Testing
import DeskPetPipeline
import DeskPetHardware
import struct DeskPetPipeline.OutputEvent
import struct DeskPetPipeline.AudioFrame
import struct DeskPetPipeline.PCMChunk
@testable import DeskPetHardwareIntegration

@Suite(.serialized) struct BridgeTests {
    @Test func mapsTeamOutputWithoutLosingIdentity() {
        let event = OutputEvent(requestId: UUID(), outputId: UUID(), text: "완료", emotion: "cheerful", activity: .speaking,
            displayState: "succeeded", displayMode: .face, displayText: "완료", speech: .init(), actionStatus: .succeeded, reminderKey: "key")
        let mapped = HardwareBridge.map(event)
        #expect(mapped.emotion == "happy")
        #expect(mapped.displayState == "default")
        #expect(mapped.actionStatus == "succeeded")
        #expect(mapped.outputId == event.outputId.uuidString && mapped.reminderKey == "key")
        var conflicting = event
        conflicting.emotion = "sad"
        #expect(HardwareBridge.map(conflicting).emotion == "happy")
        conflicting.actionStatus = .failed
        conflicting.emotion = "cheerful"
        #expect(HardwareBridge.map(conflicting).emotion == "sad")
        conflicting.actionStatus = .unknown
        #expect(HardwareBridge.map(conflicting).emotion == "neutral")
    }
    @Test(.enabled(if: ProcessInfo.processInfo.environment["DESKPET_TEST_SOCKET"] != nil))
    func realSocketPlaybackGateTouchAndFormat() async throws {
        let client = try await HardwareClient.connect(path: ProcessInfo.processInfo.environment["DESKPET_TEST_SOCKET"]!)
        let bridge = HardwareBridge(client: client)
        var adapters = Adapters.mocks(); adapters.speaker = bridge; adapters.display = bridge
        let pipeline = try DeskPetPipeline(adapters: adapters)
        await pipeline.start(automaticReminders: false)
        try await bridge.start(pipeline: pipeline, capture: false)
        let event = OutputEvent(requestId: UUID(), outputId: UUID(), text: "시험", emotion: "cheerful", activity: .speaking,
            displayState: "normal", displayMode: .face, displayText: "시험", speech: .init())
        let id = UUID()
        try await bridge.begin(jobId: id, event: event)
        #expect(await !pipeline.acceptsInput(epoch: client.inputEpoch))
        let start = ContinuousClock.now
        try await bridge.write(.init(data: Data(repeating: 0, count: 9600), sampleRate: 24000), jobId: id)
        await #expect(throws: HardwareError.self) { try await bridge.write(.init(data: Data(repeating: 0, count: 640), sampleRate: 16000), jobId: id) }
        try await bridge.finish(jobId: id)
        #expect(ContinuousClock.now - start >= .milliseconds(180))
        #expect(await !pipeline.acceptsInput(epoch: client.inputEpoch))
        try await Task.sleep(for: .milliseconds(200))
        #expect(await !pipeline.acceptsInput(epoch: client.inputEpoch))
        try await Task.sleep(for: .milliseconds(200))
        #expect(await pipeline.acceptsInput(epoch: client.inputEpoch))
        for phase in ["down", "move", "up"] {
            _ = try await client.command("mock.touch", fields: ["phase": .string(phase), "contactId": .string("test"), "x": .double(0.2), "y": .double(0.7)])
        }
        try await Task.sleep(for: .milliseconds(40))
        let diagnostics = await bridge.diagnostics()
        #expect(diagnostics.contains("touch:down:test"))
        #expect(diagnostics.contains("touch:up:test"))
        await bridge.shutdown(); await pipeline.shutdown()
        try await Task.sleep(for: .milliseconds(100))
    }
    @Test(.enabled(if: ProcessInfo.processInfo.environment["DESKPET_TEST_SOCKET"] != nil))
    func uncertainStopNeverReopensInput() async throws {
        let client = try await HardwareClient.connect(path: ProcessInfo.processInfo.environment["DESKPET_TEST_SOCKET"]!)
        let bridge = HardwareBridge(client: client)
        let pipeline = try DeskPetPipeline(adapters: .mocks())
        await pipeline.start(automaticReminders: false)
        try await bridge.start(pipeline: pipeline, capture: false)
        let event = OutputEvent(requestId: UUID(), outputId: UUID(), text: "시험", emotion: "neutral", activity: .speaking,
            displayState: "normal", displayMode: .face, displayText: "", speech: .init())
        let id = UUID()
        try await bridge.begin(jobId: id, event: event)
        try await bridge.write(.init(data: Data(repeating: 0, count: 32000), sampleRate: 16000), jobId: id)
        _ = try await client.command("mock.fail", fields: ["device": .string("speaker-stop")])
        await bridge.stop(jobId: id)
        try await Task.sleep(for: .milliseconds(400))
        #expect(!client.isCurrentInputEpoch(client.inputEpoch))
        #expect(await !pipeline.acceptsInput(epoch: client.inputEpoch))
        await bridge.shutdown(); await pipeline.shutdown()
        try await Task.sleep(for: .milliseconds(100))
    }
    @Test(.enabled(if: ProcessInfo.processInfo.environment["DESKPET_TEST_SOCKET"] != nil))
    func managedCaptureSurvivesLongPlaybackAndFreshFramesResume() async throws {
        let client = try await HardwareClient.connect(path: ProcessInfo.processInfo.environment["DESKPET_TEST_SOCKET"]!)
        let bridge = HardwareBridge(client: client)
        let pipeline = try DeskPetPipeline(adapters: .mocks())
        await pipeline.start(automaticReminders: false)
        try await bridge.start(pipeline: pipeline)
        let first = try await bridge.nextFrame()
        #expect(first?.chunk.data.count == 640)
        #expect(first?.chunk.sampleRate == 16000)
        let event = OutputEvent(requestId: UUID(), outputId: UUID(), text: "시험", emotion: "neutral", activity: .speaking,
            displayState: "normal", displayMode: .face, displayText: "", speech: .init())
        let id = UUID()
        try await bridge.begin(jobId: id, event: event)
        let next = Task { try await bridge.nextFrame() }
        try await bridge.write(.init(data: Data(repeating: 0, count: 48000), sampleRate: 16000), jobId: id)
        try await bridge.finish(jobId: id)
        let drained = Date()
        let frame = try await next.value
        #expect(frame?.inputEpoch != first?.inputEpoch)
        #expect(frame!.timestamp.timeIntervalSince(drained) >= 0.3)
        #expect(await pipeline.acceptsInput(epoch: frame?.inputEpoch))
        await bridge.shutdown(); await pipeline.shutdown()
        try await Task.sleep(for: .milliseconds(100))
    }
}
