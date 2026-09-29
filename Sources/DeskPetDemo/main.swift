import DeskPetPipeline
import Foundation

@main struct Demo {
    static func main() async throws {
        let moduleDemo = CommandLine.arguments.contains("--module-demo")
        let benchmark = CommandLine.arguments.contains("--benchmark") || moduleDemo
        let target = Target(entityId: "fixture-report", name: "보고서 제출")
        let due = ISO8601DateFormatter().date(from: "2026-10-02T14:59:00Z")!
        let router = MockRouter(scripts: [
            "보고서 제출 기한을 금요일로 미뤄줘": MockRoute(heads: HeadOutputs(intent: .postpone, name: target.name, dueAt: due), resolution: .unique(target)),
            "그거 끝났어": MockRoute(heads: HeadOutputs(intent: .complete)),
            "커밋 요약해줘": MockRoute(heads: HeadOutputs(intent: .mcp, task: "커밋 요약")),
            "기한을 바꿔줘": MockRoute(heads: HeadOutputs(intent: .postpone))
        ])
        let eureka = MockEureka(items: [ReminderItem(target: target, description: "분석 보고서를 제출해야 해요.", dueAt: due)])
        let gpu = try GPUExecutor()
        var adapters = Adapters.gpuMocks(executor: gpu, scripts: router.scripts, delay: .milliseconds(moduleDemo ? 2 : 0))
        adapters.eureka = eureka
        let pipeline = try DeskPetPipeline(adapters: adapters)
        await pipeline.start(automaticReminders: false)
        let logger = Task {
            for await event in pipeline.events {
                if benchmark { continue }
                switch event {
                case .routed(_, let decision): print("route: \(decision.route.rawValue)")
                case .action(_, let result): print("action: \(result.status.rawValue)")
                case .output(let output): print("DeskPet: \(output.text)")
                case .error(_, let error): print("error: \(error.code)")
                default: break
                }
            }
        }
        let started = ContinuousClock.now
        if moduleDemo {
            var delivered = 0; var audioCompleted = 0
            for _ in 0..<32 {
                var ids: [UUID] = []
                for lane in 0..<8 {
                    ids.append(try await pipeline.submit(Envelope(conversationId: "lane-\(lane)", kind: "stt.final", payload: Transcript(text: "안녕"))))
                }
                while await pipeline.inFlightCount() > 0 { try await Task.sleep(for: .milliseconds(1)) }
                for id in ids {
                    if await pipeline.record(requestId: id)?.delivery?.delivered == true { delivered += 1 }
                    if await pipeline.record(requestId: id)?.delivery?.audio == .completed { audioCompleted += 1 }
                }
            }
            let elapsed = started.duration(to: .now).components
            let seconds = Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18
            let stats = gpu.snapshot()
            print("Module mock requests: 256; delivered: \(delivered); audio completed: \(audioCompleted); elapsed: \(String(format: "%.3f", seconds))s")
            print("GPU dispatcher threads: \(stats.threadCount); mock operations completed: \(stats.completed)")
        } else {
            let inputs = benchmark ? Array(repeating: "안녕", count: 1_000) : ["보고서 제출 기한을 금요일로 미뤄줘", "그거 끝났어", "커밋 요약해줘", "안녕", "기한을 바꿔줘"]
            for text in inputs {
                if !benchmark { print("User: \(text)") }
                _ = try await pipeline.submit(Envelope(conversationId: "demo", kind: "stt.final", payload: Transcript(text: text)))
                while await pipeline.inFlightCount() > 0 { try await Task.sleep(for: .milliseconds(5)) }
            }
        }
        if benchmark { print("Remaining in flight: \(await pipeline.inFlightCount()); dropped diagnostic events: \(await pipeline.droppedEventCount())") }
        let measurements = await pipeline.measurements()
        for (stage, value) in measurements.sorted(by: { $0.key < $1.key }) {
            print("\(stage) p50=\(String(format: "%.2f", value.p50Milliseconds))ms p95=\(String(format: "%.2f", value.p95Milliseconds))ms")
        }
        await pipeline.shutdown(); await gpu.shutdown(); await logger.value
    }
}
