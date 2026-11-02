import Foundation
import Testing
import class DeskPetHardware.HardwareClient
import struct DeskPetHardware.HardwareEvent
import struct DeskPetHardware.HardwareMotionAdapter
@testable import DeskPetPipeline

private actor CaptureVAD: VoiceActivityDetector {
    var count = 0
    func classify(_ frame: AudioFrame) -> VoiceActivity { count += 1; return .silence }
}
private actor HardwareEvidence {
    var events: [HardwareEvent] = []
    func add(_ event: HardwareEvent) { events.append(event) }
    func contains(_ name: String, outputId: UUID? = nil) -> Bool {
        events.contains { $0.name == name && $0.fields["simulated"] == .bool(true) &&
            (outputId == nil || $0.fields["outputId"]?.string == outputId?.uuidString) }
    }
}

@Test(.enabled(if: ProcessInfo.processInfo.environment["DESKPET_HARDWARE_TESTS"] == "1"))
func simulatedHardwareConnectsToPipeline() async throws {
    let client = try await HardwareClient.connect(path: ProcessInfo.processInfo.environment["DESKPET_HARDWARE_SOCKET"] ?? HardwareClient.defaultSocket)
    defer { client.close() }
    let microphone = HardwareMicrophone(client: client), display = HardwareDisplay(client: client)
    let evidence = HardwareEvidence()
    let consumer = Task {
        for try await event in client.events {
            await evidence.add(event)
            if event.name == "audio.started", let id = event.fields["outputId"]?.string { try await display.audioStarted(outputId: id) }
        }
    }
    defer { consumer.cancel() }
    let target = Target(entityId: "hardware-test", name: "보고서 제출")
    let eureka = MockEureka(items: [ReminderItem(target: target, description: "분석 보고서 제출", dueAt: Date().addingTimeInterval(3600))])
    var adapters = Adapters.mocks(router: MockRouter(scripts: ["조회": MockRoute(heads: HeadOutputs(intent: .query, name: target.name), resolution: .unique(target))]), eureka: eureka)
    adapters.speaker = HardwareSpeaker(client: client); adapters.display = display
    let pipeline = try DeskPetPipeline(adapters: adapters)
    await pipeline.start(automaticReminders: false)
    let vad = CaptureVAD()
    var config = AudioIngressConfiguration(); config.sampleRate = 16000
    let ingress = AudioIngress(microphone: microphone, vad: vad, configuration: config)
    try await microphone.start()
    let capture = Task { try await ingress.run(pipeline: pipeline, conversationId: "microphone") }
    defer { capture.cancel() }

    func submit() async throws -> RequestRecord? {
        let id = try await pipeline.submit(Envelope(conversationId: "hardware-test", kind: "stt.final", payload: Transcript(text: "조회")))
        let deadline = ContinuousClock.now.advanced(by: .seconds(3))
        while await pipeline.inFlightCount() > 0 && ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(5)) }
        return await pipeline.record(requestId: id)
    }
    let success = try await submit()
    #expect(success?.action?.status == .succeeded && success?.delivery?.audio == .completed)
    #expect(await vad.count > 0)
    if let outputId = success?.delivery?.outputId { #expect(await evidence.contains("audio.completed", outputId: outputId)) }

    _ = try await client.command("mock.fail", fields: ["device": .string("speaker")])
    let fallback = try await submit()
    #expect(fallback?.action?.status == .succeeded)
    #expect(fallback?.delivery?.audio == .failed && fallback?.delivery?.display == .completed)
    #expect(await eureka.count() == 2)
    if let outputId = fallback?.delivery?.outputId { #expect(await evidence.contains("display.displayed", outputId: outputId)) }
    let motion = HardwareMotionAdapter(client: client)
    try await motion.enable()
    try await motion.apply(preset: "wiggle", requestId: UUID().uuidString, outputId: UUID().uuidString)
    try await motion.disable()
    #expect(await evidence.contains("motion.commanded"))
    try await microphone.stop(); try await capture.value
    await pipeline.shutdown()
}
