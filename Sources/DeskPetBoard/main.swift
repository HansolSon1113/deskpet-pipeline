import Foundation
import DeskPetPipeline
import DeskPetHardware
import DeskPetHardwareIntegration

private actor Results {
    var deliveries = 0
    var failed = false
    func observe(_ event: PipelineEvent) {
        switch event {
        case .delivery(let result):
            deliveries += 1
            if result.audio != .completed || !result.delivered { failed = true }
            print("delivery: \(result)")
        case .error(_, let error): failed = true; print("FAIL pipeline: \(error)")
        case .touch(let touch): print("touch: \(touch.phase) \(touch.x),\(touch.y) simulated=\(touch.simulated)")
        default: break
        }
    }
}
@main struct Board {
    static func main() async throws {
        let args = CommandLine.arguments
        func value(_ key: String, _ fallback: String) -> String {
            guard let i = args.firstIndex(of: key), i + 1 < args.count else { return fallback }; return args[i + 1]
        }
        if args.contains("--help") {
            print("deskpet-board [--socket PATH] [--seconds 0] [--real] [--motion]\nModels/APIs are synthetic fixtures. --real uses configured physical peripherals.")
            return
        }
        guard let seconds = Double(value("--seconds", "0")), seconds.isFinite, seconds >= 0 else { throw PipelineError("INVALID_SECONDS", stage: "cli") }
        let client = try await HardwareClient.connect(path: value("--socket", HardwareClient.defaultSocket))
        let health = try await client.command("health")
        let real = args.contains("--real")
        guard health["mode"]?.string == (real ? "real" : "mock") else { client.close(); throw PipelineError("MODE_MISMATCH", stage: "cli") }
        let vad = BoardVAD(), results = Results()
        let bridge = HardwareBridge(client: client, vad: vad, motionEnabled: args.contains("--motion"))
        var adapters = Adapters.mocks(); adapters.stt = BoardSTT(); adapters.speaker = bridge; adapters.display = bridge
        let pipeline = try DeskPetPipeline(adapters: adapters)
        await pipeline.start(automaticReminders: false)
        let logger = Task { for await event in pipeline.events { await results.observe(event) } }
        var ingressTask: Task<Void, Error>?
        do {
            try await bridge.start(pipeline: pipeline)
            var config = AudioIngressConfiguration(); config.sampleRate = 16000; config.maxFrameBytes = 640
            let ingress = AudioIngress(microphone: bridge, vad: vad, configuration: config)
            ingressTask = Task { try await ingress.run(pipeline: pipeline, conversationId: "board-fixture") }
            let deadline = ContinuousClock.now.advanced(by: .seconds(seconds))
            var iterations = 0
            repeat {
                let before = await results.deliveries
                await vad.arm()
                let timeout = ContinuousClock.now.advanced(by: .seconds(15))
                while await results.deliveries == before {
                    guard ContinuousClock.now < timeout else { throw PipelineError("SMOKE_TIMEOUT", stage: "cli") }
                    if await results.failed { throw PipelineError("SMOKE_ERROR", stage: "cli") }
                    try await Task.sleep(for: .milliseconds(20))
                }
                try await Task.sleep(for: .milliseconds(400))
                iterations += 1
            } while ContinuousClock.now < deadline
            if await results.failed { throw PipelineError("SMOKE_ERROR", stage: "cli") }
            print("PASS iterations=\(iterations) peripherals=\(real ? "real" : "simulated") models=fixture")
            for note in await bridge.diagnostics() { print("diagnostic: \(note)") }
        } catch {
            await bridge.shutdown(); ingressTask?.cancel(); _ = try? await ingressTask?.value
            await pipeline.shutdown(); logger.cancel(); throw error
        }
        await bridge.shutdown(); ingressTask?.cancel(); _ = try? await ingressTask?.value
        await pipeline.shutdown(); logger.cancel()
    }
}
