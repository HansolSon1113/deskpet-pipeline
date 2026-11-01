import DeskPetPipeline
import DeskPetHardware
import Foundation

@main struct Demo {
    static func main() async {
        var status: Int32 = 0
        do { try await run() }
        catch is CancellationError { }
        catch {
            let text: String
            if let fault = error as? PipelineError { text = "\(fault.stage): \(fault.code) \(fault.message)\n" }
            else { text = "\(error)\n" }
            FileHandle.standardError.write(Data(text.utf8)); status = 1
        }
        await NativeConsole.flush()
        if status != 0 { exit(status) }
    }
    static func run() async throws {
        if CommandLine.arguments.contains("--jetson") { try await JetsonLauncher.run(CommandLine.arguments); return }
        if CommandLine.arguments.contains("--jetson-native") { try await JetsonLauncher.runNative(CommandLine.arguments); return }
        if CommandLine.arguments.contains("--project") { try await ProjectLauncher.run(CommandLine.arguments); return }
        if CommandLine.arguments.contains("--native") { try await NativeDemo.run(CommandLine.arguments); return }
        let connectedDemo = CommandLine.arguments.contains("--connected-demo")
        let moduleDemo = CommandLine.arguments.contains("--module-demo")
        let benchmark = CommandLine.arguments.contains("--benchmark") || moduleDemo
        let target = Target(entityId: connectedDemo ? "2001" : "fixture-report", name: "보고서 제출")
        let pr = Target(kind: "github_pr", entityId: "NewLine/DeskPet#42", name: "DeskPet PR 42",
                        repository: ["owner": "NewLine", "name": "DeskPet"], prNumber: 42)
        let due = ISO8601DateFormatter().date(from: "2026-10-02T14:59:00Z")!
        let router = MockRouter(scripts: [
            "보고서 제출 상태 알려줘": MockRoute(heads: HeadOutputs(intent: .query, name: target.name), resolution: .unique(target)),
            "DeskPet PR 42 리뷰해줘": MockRoute(heads: HeadOutputs(intent: .reviewPR, name: pr.name), resolution: .unique(pr)),
            "보고서 제출 기한을 금요일로 미뤄줘": MockRoute(heads: HeadOutputs(intent: .postpone, name: target.name, dueAt: due), resolution: .unique(target)),
            "그거 끝났어": MockRoute(heads: HeadOutputs(intent: .complete)),
            "커밋 요약해줘": MockRoute(heads: HeadOutputs(intent: .mcp, task: "커밋 요약")),
            "기한을 바꿔줘": MockRoute(heads: HeadOutputs(intent: .postpone))
        ])
        let eureka = MockEureka(items: [ReminderItem(target: target, description: "분석 보고서를 제출해야 해요.", dueAt: due)])
        let gpu = try GPUExecutor()
        var adapters = Adapters.gpuMocks(executor: gpu, scripts: router.scripts, delay: .milliseconds(moduleDemo ? 2 : 0))
        adapters.eureka = eureka
        var hardware: HardwareClient?
        var hardwareEvents: Task<Void, Never>?
        if CommandLine.arguments.contains("--hardware-simulation") {
            let client = try await HardwareClient.connect(path: ProcessInfo.processInfo.environment["DESKPET_HARDWARE_SOCKET"] ?? HardwareClient.defaultSocket)
            hardware = client
            let display = HardwareDisplay(client: client)
            adapters.speaker = HardwareSpeaker(client: client); adapters.display = display
            hardwareEvents = Task {
                do {
                    for try await event in client.events {
                        if event.name == "audio.started", let id = event.fields["outputId"]?.string { try await display.audioStarted(outputId: id) }
                        if event.name == "device.error" || event.name == "audio.failed" { print("hardware: \(event.name)") }
                    }
                } catch { print("hardware connection: \(error)") }
            }
        }
        defer { hardware?.close(); hardwareEvents?.cancel() }
        if connectedDemo || CommandLine.arguments.contains("--harness-server") || CommandLine.arguments.contains("--eureka-server") {
            let env = ProcessInfo.processInfo.environment
            guard let token = env["HARNESS_WS_TOKEN"] ?? (connectedDemo ? "fixture-pipeline-token-123" : nil),
                  let url = URL(string: env["HARNESS_WS_URL"] ?? (connectedDemo ? "ws://127.0.0.1:8788" : "ws://127.0.0.1:8787")) else {
                throw PipelineError("HARNESS_CONFIGURATION", stage: "harness")
            }
            if connectedDemo || CommandLine.arguments.contains("--harness-server") { adapters.harness = try ServerHarness(url: url, token: token) }
            if connectedDemo || CommandLine.arguments.contains("--eureka-server") { adapters.eureka = try ServerEureka(url: url, token: token) }
        }
        if connectedDemo { print("연결 시연: mock 발화 → Swift pipeline → Eureka/Harness 서버 → pipeline 출력 (업무 데이터·음성·장치는 fixture/mock)") }
        let pipeline = try DeskPetPipeline(adapters: adapters)
        await pipeline.start(automaticReminders: false)
        let logger = Task {
            for await event in pipeline.events {
                if benchmark { continue }
                switch event {
                case .routed(_, let decision): print("route: \(decision.route.rawValue)")
                case .action(_, let result):
                    print("action: \(result.status.rawValue)")
                    if connectedDemo {
                        for key in ["itemId", "itemStatus", "action", "repository", "prNumber", "headSha"] {
                            if let value = result.facts[key] { print("server \(key): \(value)") }
                        }
                    }
                case .output(let output): print("DeskPet: \(output.text)")
                case .error(_, let error): print("error: \(error.code)")
                default: break
                }
            }
        }
        let started = ContinuousClock.now
        var connectedSuccesses = 0
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
            let inputs = connectedDemo ? ["보고서 제출 상태 알려줘", "DeskPet PR 42 리뷰해줘", "안녕"] :
                benchmark ? Array(repeating: "안녕", count: 1_000) : ["보고서 제출 기한을 금요일로 미뤄줘", "그거 끝났어", "커밋 요약해줘", "안녕", "기한을 바꿔줘"]
            for text in inputs {
                if !benchmark { print("User: \(text)") }
                let id = try await pipeline.submit(Envelope(conversationId: "demo", kind: "stt.final", payload: Transcript(text: text)))
                while await pipeline.inFlightCount() > 0 { try await Task.sleep(for: .milliseconds(5)) }
                if connectedDemo, let record = await pipeline.record(requestId: id),
                   record.action?.status == .succeeded, record.delivery?.delivered == true { connectedSuccesses += 1 }
            }
        }
        if benchmark { print("Remaining in flight: \(await pipeline.inFlightCount()); dropped diagnostic events: \(await pipeline.droppedEventCount())") }
        let measurements = await pipeline.measurements()
        for (stage, value) in measurements.sorted(by: { $0.key < $1.key }) {
            print("\(stage) p50=\(String(format: "%.2f", value.p50Milliseconds))ms p95=\(String(format: "%.2f", value.p95Milliseconds))ms")
        }
        await pipeline.shutdown(); await gpu.shutdown(); await logger.value
        if connectedDemo {
            guard connectedSuccesses == 2 else { throw PipelineError("CONNECTED_DEMO_FAILED", stage: "demo", message: "Eureka/Harness 서버 결과와 출력 완료를 확인하지 못했습니다.") }
            print("연결 시연 완료: Eureka 조회·Harness PR 리뷰 모두 성공, pipeline 출력 완료 2/2")
        }
    }
}
