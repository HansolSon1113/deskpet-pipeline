import Foundation
import Dispatch
import DeskPetPipeline
import DeskPetHardware
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
#if os(macOS)
import Darwin
#else
import Glibc
#endif

/// Dedicated Linux / Jetson Orin Nano launcher and native session runner.
/// Leaves existing macOS ProjectLauncher and NativeDemo untouched.
enum JetsonLauncher {
    private final class Children: @unchecked Sendable {
        private let lock = NSLock()
        private var children: [(String, Process)] = []
        private var stopping = false
        var cancelled: Bool { lock.withLock { stopping } }
        func cancel() { lock.withLock { stopping = true } }
        func add(_ name: String, _ process: Process) { lock.withLock { children.append((name, process)) } }
        func remove(_ process: Process) { lock.withLock { children.removeAll { $0.1 === process } } }
        func failed(except pipeline: Process) -> String? {
            lock.withLock { children.first { $0.1 !== pipeline && !$0.1.isRunning }?.0 }
        }
        func stop() async {
            let owned = lock.withLock { children.reversed().map { $0 } }
            for (_, process) in owned where process.isRunning {
                process.interrupt()
                var until = ContinuousClock.now.advanced(by: .seconds(5))
                while process.isRunning, ContinuousClock.now < until { try? await Task.sleep(for: .milliseconds(50)) }
                if process.isRunning { process.terminate() }
                until = .now.advanced(by: .seconds(2))
                while process.isRunning, ContinuousClock.now < until { try? await Task.sleep(for: .milliseconds(50)) }
                if process.isRunning { kill(process.processIdentifier, SIGKILL) }
            }
        }
    }

    private static func fault(_ code: String, _ message: String) -> PipelineError {
        PipelineError(code, stage: "jetson", message: message)
    }

    static func run(_ arguments: [String]) async throws {
        let options = Array(arguments.drop(while: { $0 != "--jetson" }).dropFirst())
        if options.contains("--help") {
            print("./run-linux.sh [--keyboard] [--direct-llm] [--no-router-context] [--short-output] [--metrics-dir DIR] [--real-hardware] [--config FILE] [--no-build] [--check] [--headless] [--read-only]")
            print("Linux / Jetson Orin Nano용 진입점입니다. --real-hardware 지정 시 ALSA/I2C 드라이버 및 config/jetson.json을 사용합니다.")
            return
        }
        var customConfig: String?
        var metricsDirectory: String?
        var i = 0
        while i < options.count {
            if options[i] == "--config" || options[i] == "--metrics-dir" {
                guard i + 1 < options.count, !options[i + 1].hasPrefix("--") else {
                    throw fault("CONFIG_REQUIRED", "\(options[i]) 뒤에 경로를 지정하세요.")
                }
                if options[i] == "--config" { customConfig = options[i + 1] }
                else { metricsDirectory = options[i + 1] }
                i += 2
            } else if ["--keyboard", "--direct-llm", "--no-router-context", "--short-output", "--no-build", "--check", "--headless", "--read-only", "--real-hardware"].contains(options[i]) {
                i += 1
            } else {
                throw fault("UNKNOWN_OPTION", options[i])
            }
        }

        let files = FileManager.default
        let root = URL(fileURLWithPath: files.currentDirectoryPath).standardizedFileURL
        let defaultLocal = root.appendingPathComponent("Models/native.jetson.local.json")
        let configURL = URL(fileURLWithPath: customConfig ?? defaultLocal.path).standardizedFileURL
        if !files.fileExists(atPath: configURL.path), customConfig == nil {
            let example = root.appendingPathComponent("Models/native.jetson.example.json")
            try files.copyItem(at: example, to: configURL)
        }
        let config = try JSONDecoder().decode(NativeDemo.Configuration.self, from: Data(contentsOf: configURL))
        let modelRoot = configURL.deletingLastPathComponent()
        func resolve(_ path: String) -> String {
            path.hasPrefix("/") ? path : modelRoot.appendingPathComponent(path).standardizedFileURL.path
        }

        var environment = try NativeDemo.environment(config.harnessEnvFile.map(resolve))
        let harness = URL(fileURLWithPath: environment["DESKPET_HARNESS_DIR"] ?? root.deletingLastPathComponent().appendingPathComponent("LLM_test").path)
        let hardware = URL(fileURLWithPath: environment["DESKPET_HARDWARE_DIR"] ?? root.deletingLastPathComponent().appendingPathComponent("deskpet-hardware").path)
        for (name, path) in [("Harness", harness.appendingPathComponent("scripts/server.ts").path), ("하드웨어", hardware.appendingPathComponent("CMakeLists.txt").path)] {
            guard files.fileExists(atPath: path) else { throw fault("REPOSITORY_MISSING", "\(name) 저장소를 찾지 못했습니다: \(path)") }
        }
        for key in ["HARNESS_WS_TOKEN", "GITHUB_TEST_REPO", "GITHUB_TEST_PR"] {
            guard environment[key]?.isEmpty == false else { throw fault("CONFIGURATION_MISSING", "기존 Harness .env에 \(key)을 설정하세요.") }
        }
        if !options.contains("--read-only"), environment["DESKPET_STORE_PATH"]?.isEmpty != false {
            throw fault("STORE_REQUIRED", "쓰기 실행에는 기존 .env의 DESKPET_STORE_PATH가 필요합니다.")
        }

        let keyboard = options.contains("--keyboard")
        let directLLM = (config.directLLM ?? false) || options.contains("--direct-llm")
        let roles = (keyboard ? ["router", "output", "emotion", "tts"] : ["router", "output", "stt", "vad", "emotion", "tts"])
            .filter { !directLLM || $0 != "router" }
        for role in roles {
            guard let model = config.models[role], model.role == role else { throw fault("MODEL_CONFIGURATION_MISSING", role) }
            for path in model.paths.values where !files.fileExists(atPath: resolve(path)) {
                throw fault("MODEL_FILE_MISSING", "\(resolve(path))\n모델 준비 절차: docs/native-models.md")
            }
        }

        let port = environment["HARNESS_WS_PORT"] ?? "8787"
        let scheme = environment["HARNESS_WS_TLS_CERT"]?.isEmpty == false ? "wss" : "ws"
        let serverURL = URL(string: environment["HARNESS_WS_URL"] ?? "\(scheme)://127.0.0.1:\(port)")
        guard let serverURL, ["ws", "wss"].contains(serverURL.scheme), ["127.0.0.1", "localhost", "::1"].contains(serverURL.host),
              (serverURL.port ?? (serverURL.scheme == "wss" ? 443 : 80)) == Int(port) else {
            throw fault("LOCAL_SERVER_CONFIGURATION", "HARNESS_WS_URL과 HARNESS_WS_PORT를 같은 로컬 서버로 설정하세요.")
        }
        environment["HARNESS_WS_URL"] = serverURL.absoluteString

        let socket = config.hardwareSocket.map { value in
            value == "auto" ? environment["XDG_RUNTIME_DIR"].map { $0 + "/deskpet-hardware.sock" } ?? "/tmp/deskpet-\(getuid())/hardware.sock" : resolve(value)
        }
        let nativeLibrarySo = root.appendingPathComponent(".build/native/libdeskpet_inference.so").path
        let nativeLibraryDylib = root.appendingPathComponent(".build/native/libdeskpet_inference.dylib").path
        let libraries = Set([resolve(config.library)] + roles.compactMap { config.models[$0] }.map { resolve($0.library ?? config.library) })
        for path in libraries where path != nativeLibrarySo && path != nativeLibraryDylib && !files.fileExists(atPath: path) {
            throw fault("BACKEND_LIBRARY_MISSING", path)
        }

        if options.contains("--check") {
            print("Jetson 실행 구성 준비 완료: 모델 \(roles.count)개 / 기존 Harness 설정 / \(socket == nil ? "진단 화면" : (options.contains("--real-hardware") ? "Jetson 실물 하드웨어" : "하드웨어 모의 화면"))")
            print("외부 API·GPU·마이크는 실행하지 않았습니다.")
            return
        }

        try files.createDirectory(at: root.appendingPathComponent(".build"), withIntermediateDirectories: true)
        let descriptor = open(root.appendingPathComponent(".build/project-run.lock").path, O_CREAT | O_RDWR, 0o600)
        guard descriptor >= 0 else { throw fault("LAUNCH_LOCK_FAILED", "실행 잠금을 열지 못했습니다.") }
        defer { close(descriptor) }
        _ = fcntl(descriptor, F_SETFD, FD_CLOEXEC)
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else { throw fault("ALREADY_RUNNING", "이미 이 진입점으로 실행 중입니다.") }

        let children = Children()
        signal(SIGINT, SIG_IGN); signal(SIGTERM, SIG_IGN); signal(SIGPIPE, SIG_IGN)
        let signals = [SIGINT, SIGTERM].map { number -> DispatchSourceSignal in
            let source = DispatchSource.makeSignalSource(signal: Int32(number), queue: .global())
            source.setEventHandler { children.cancel() }; source.resume(); return source
        }
        defer { for source in signals { source.cancel() }; signal(SIGINT, SIG_DFL); signal(SIGTERM, SIG_DFL); signal(SIGPIPE, SIG_DFL) }

        func command(_ name: String, _ args: [String], directory: URL) async throws {
            if children.cancelled { throw CancellationError() }
            print(name)
            let process = Process(); process.executableURL = URL(fileURLWithPath: "/usr/bin/env"); process.arguments = args
            process.currentDirectoryURL = directory; process.environment = environment
            try process.run(); children.add(name, process)
            while process.isRunning {
                if children.cancelled { throw CancellationError() }
                try await Task.sleep(for: .milliseconds(50))
            }
            children.remove(process)
            guard process.terminationStatus == 0 else { throw fault("BUILD_FAILED", "\(name) (종료 코드 \(process.terminationStatus))") }
        }

        let logs = root.appendingPathComponent(".build/run/\(UUID().uuidString)")
        try files.createDirectory(at: logs, withIntermediateDirectories: true)
        func start(_ name: String, _ args: [String], directory: URL, console: Bool = false, input: Any = FileHandle.nullDevice) throws -> Process {
            if children.cancelled { throw CancellationError() }
            let log = logs.appendingPathComponent(name + ".log"); files.createFile(atPath: log.path, contents: nil, attributes: [.posixPermissions: 0o600])
            let output = try FileHandle(forWritingTo: log); defer { try? output.close() }
            let process = Process(); process.executableURL = URL(fileURLWithPath: "/usr/bin/env"); process.arguments = args
            process.currentDirectoryURL = directory; process.environment = environment
            process.standardError = output; process.standardOutput = console ? FileHandle.standardOutput : output
            process.standardInput = input
            try process.run(); children.add(name, process); return process
        }

        let realHardware = options.contains("--real-hardware")
        let headless = options.contains("--headless")
        do {
            if !options.contains("--no-build") {
                let sherpaSo = root.appendingPathComponent(".build/native-deps/sherpa/lib/libsherpa-onnx-c-api.so").path
                if !files.fileExists(atPath: sherpaSo), environment["SHERPA_ROOT"] == nil {
                    try await command("Linux 음성 추론 SDK 준비", ["sh", "tools/setup-native-deps-linux.sh"], directory: root)
                }
                if libraries.contains(nativeLibrarySo) || libraries.contains(nativeLibraryDylib) {
                    try await command("Jetson 네이티브 모델 연결 빌드 (CUDA)", ["sh", "tools/build-native-linux.sh", "-DDESKPET_PREP=OFF"], directory: root)
                }
                if socket != nil {
                    var configure = ["sh", "tools/configure-hardware-linux.sh", hardware.path, hardware.appendingPathComponent("build").path]
                    if realHardware && headless { configure.append("--real-headless") }
                    else if realHardware { configure.append("--real") }
                    else if headless { configure.append("--headless") }
                    try await command("하드웨어 데몬 설정 준비", configure, directory: root)
                    try await command("하드웨어 데몬 빌드", ["cmake", "--build", "build", "--target", "deskpet-hardware", "-j", "4"], directory: hardware)
                }
                if !files.fileExists(atPath: harness.appendingPathComponent("node_modules/tsx").path) {
                    try await command("기존 Harness 의존성 준비", ["pnpm", "install", "--frozen-lockfile"], directory: harness)
                }
            }
            for path in libraries where !files.fileExists(atPath: path) { throw fault("BACKEND_LIBRARY_MISSING", path) }

            if let socket {
                if files.fileExists(atPath: socket) {
                    throw fault("HARDWARE_ALREADY_RUNNING", "기존 하드웨어 프로세스를 종료한 뒤 다시 실행하세요: \(socket)")
                }
                let hwConfig = realHardware ? "config/jetson.json" : "config/mock.json"
                var args = [hardware.appendingPathComponent("build/deskpet-hardware").path, "--config", hwConfig, "--socket", socket]
                if headless { args.append("--headless") }
                let process = try start("hardware", args, directory: hardware)
                let until = ContinuousClock.now.advanced(by: .seconds(15))
                while true {
                    guard process.isRunning, ContinuousClock.now < until, !children.cancelled else {
                        throw fault("HARDWARE_START_FAILED", "하드웨어 데몬을 시작하지 못했습니다. 로그: \(logs.path)")
                    }
                    if files.fileExists(atPath: socket), let client = try? await HardwareClient.connect(path: socket) {
                        client.close(); break
                    }
                    try await Task.sleep(for: .milliseconds(100))
                }
                print(realHardware ? "Jetson 하드웨어 데몬 준비 완료" : "하드웨어 모의 화면 준비 완료")
            }

            var serverArgs = ["node", "--import", "tsx", "scripts/server.ts"]
            if !options.contains("--read-only") { serverArgs += ["--allow-github-writes", "--allow-eureka-writes"] }
            if await welcomed(serverURL, token: environment["HARNESS_WS_TOKEN"]!) {
                throw fault("HARNESS_ALREADY_RUNNING", "이미 Harness가 실행 중입니다. 기존 서버를 종료한 뒤 다시 실행하세요: \(serverURL.absoluteString)")
            }
            let server = try start("harness", serverArgs, directory: harness)
            let until = ContinuousClock.now.advanced(by: .seconds(15))
            while true {
                guard server.isRunning, ContinuousClock.now < until, !children.cancelled else {
                    throw fault("HARNESS_START_FAILED", "Harness를 시작하지 못했습니다. 로그: \(logs.path)")
                }
                if await welcomed(serverURL, token: environment["HARNESS_WS_TOKEN"]!), server.isRunning { break }
                try await Task.sleep(for: .milliseconds(100))
            }
            let instruction = keyboard ? "문장을 입력하고 Enter를 누르세요. 종료: /quit 또는 Ctrl-C" : "마이크에 말씀하세요. 종료: Ctrl-C"
            print("실행 로그: \(logs.path)\n\(instruction)")

            let keyboardPipe = keyboard ? Pipe() : nil
            var pipelineArgs = [URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL.path, "--jetson-native", configURL.path, keyboard ? "--keyboard" : "--live"]
            if options.contains("--no-router-context") { pipelineArgs.append("--no-router-context") }
            if options.contains("--short-output") { pipelineArgs.append("--short-output") }
            if directLLM { pipelineArgs.append("--direct-llm") }
            if let metricsDirectory { pipelineArgs += ["--metrics-dir", URL(fileURLWithPath: metricsDirectory).standardizedFileURL.path] }
            let pipeline = try start("pipeline", pipelineArgs, directory: root, console: true, input: keyboardPipe?.fileHandleForReading ?? FileHandle.nullDevice)
            try? keyboardPipe?.fileHandleForReading.close()
            let keyboardRelay: DispatchSourceRead? = keyboardPipe.map { pipe in
                let source = DispatchSource.makeReadSource(fileDescriptor: STDIN_FILENO, queue: DispatchQueue(label: "deskpet.jetson.stdin"))
                source.setEventHandler {
                    var bytes = [UInt8](repeating: 0, count: 4096)
                    let count = bytes.withUnsafeMutableBytes { read(STDIN_FILENO, $0.baseAddress, $0.count) }
                    if count == 0 || (count < 0 && errno != EINTR && errno != EAGAIN) {
                        try? pipe.fileHandleForWriting.close(); source.cancel(); return
                    }
                    if count > 0 {
                        do { try pipe.fileHandleForWriting.write(contentsOf: bytes.prefix(count)) }
                        catch { try? pipe.fileHandleForWriting.close(); source.cancel() }
                    }
                }
                source.setCancelHandler { try? pipe.fileHandleForWriting.close() }
                source.resume()
                return source
            }
            defer { keyboardRelay?.cancel() }

            while pipeline.isRunning, !children.cancelled {
                if let name = children.failed(except: pipeline) { throw fault("SERVICE_STOPPED", "\(name)이 종료됐습니다. 로그: \(logs.path)") }
                try await Task.sleep(for: .milliseconds(100))
            }
            if !children.cancelled, pipeline.terminationStatus != 0 {
                throw fault("PIPELINE_STOPPED", "파이프라인 종료 코드 \(pipeline.terminationStatus). 로그: \(logs.path)")
            }
            await children.stop()
        } catch {
            await children.stop()
            if children.cancelled { throw CancellationError() }
            throw error
        }
    }

    static func runNative(_ arguments: [String]) async throws {
        guard let flag = arguments.firstIndex(of: "--jetson-native"), arguments.count > flag + 1 else {
            throw PipelineError("NATIVE_CONFIGURATION_REQUIRED", stage: "jetson", message: "--jetson-native Models/native.jetson.local.json [--probe | --say TEXT | --text TEXT | --live | --keyboard]")
        }
        let configURL = URL(fileURLWithPath: arguments[flag + 1]).standardizedFileURL
        let root = configURL.deletingLastPathComponent()
        func resolve(_ path: String) -> String {
            path.hasPrefix("/") ? path : root.appendingPathComponent(path).standardizedFileURL.path
        }
        let config = try JSONDecoder().decode(NativeDemo.Configuration.self, from: Data(contentsOf: configURL))
        let routerContextEnabled = (config.routerContextEnabled ?? true) && !arguments.contains("--no-router-context")
        let directLLM = (config.directLLM ?? false) || arguments.contains("--direct-llm")
        let shortOutput = (config.shortOutput ?? false) || arguments.contains("--short-output")
        let keyboard = arguments.contains("--keyboard")
        let roles = (keyboard ? ["router", "output", "emotion", "tts"] : ["router", "output", "stt", "vad", "emotion", "tts"])
            .filter { !directLLM || $0 != "router" }
        let performance = try NativeDemo.performanceRecorder(config: config, arguments: arguments, configURL: configURL,
                                                            roles: roles, directLLM: directLLM)
        defer { performance.close() }
        let library = try NativeBackendLibrary(path: resolve(config.library))
        var libraries: [String: NativeBackendLibrary] = [resolve(config.library): library]
        let gpu = try GPUExecutor()
        func model(_ role: String) throws -> NativeInferenceModel {
            guard var c = config.models[role], c.role == role else { throw PipelineError("MODEL_CONFIGURATION_MISSING", stage: role) }
            c.paths = c.paths.mapValues(resolve)
            if role == "output", shortOutput { c.maxTokens = max(c.maxTokens, 256) }
            let path = resolve(c.library ?? config.library)
            if libraries[path] == nil { libraries[path] = try NativeBackendLibrary(path: path) }
            c.library = nil
            return try NativeInferenceModel(library: libraries[path]!, configuration: c, performance: performance)
        }
        if arguments.contains("--probe-cpu") {
            for role in ["stt", "vad", "emotion", "tts"] { _ = try model(role); print("CPU 모델 준비: \(role)") }
            await gpu.shutdown(); return
        }
        if let say = arguments.firstIndex(of: "--say"), arguments.count > say + 1 {
            let output = try model("output")
            let result = try await NativeOutputModel(model: output, executor: gpu, shortOutput: shortOutput).generate(OutputRequest(requestId: UUID(), requiredText: "", messages: [ChatMessage(role: "user", content: arguments[say + 1])]))
            print(result); await gpu.shutdown(); return
        }

        var loaded: [String: NativeInferenceModel] = [:]
        for role in roles { loaded[role] = try model(role); print("모델 준비: \(role)") }
        if arguments.contains("--probe") { await gpu.shutdown(); print("모델 \(loaded.count)개 로드 완료"); return }

        let env = try NativeDemo.environment(config.harnessEnvFile.map(resolve))
        guard let token = env["HARNESS_WS_TOKEN"], !token.isEmpty,
              let url = URL(string: env["HARNESS_WS_URL"] ?? "ws://127.0.0.1:8787"),
              let repo = env["GITHUB_TEST_REPO"]?.split(separator: "/"), repo.count == 2,
              let rawPR = env["GITHUB_TEST_PR"], let number = Int(rawPR), number > 0 else {
            throw PipelineError("DEMO_TARGET_CONFIGURATION_REQUIRED", stage: "jetson",
                message: "기존 Harness 설정의 HARNESS_WS_TOKEN, GITHUB_TEST_REPO, GITHUB_TEST_PR이 필요합니다.")
        }
        let repository = ["owner": String(repo[0]), "name": String(repo[1])]
        let target = Target(kind: "github_pr", entityId: "\(repo.joined(separator: "/"))#\(number)", name: "PR \(number)", repository: repository, prNumber: number)
        let harness = try ServerHarness(url: url, token: token, performance: performance)
        let recognizer: any SpeechRecognizer = keyboard ? KeyboardOnlyRecognizer() : NativeStreamingSTT(model: loaded["stt"]!)
        var adapters = Adapters(stt: recognizer,
            router: loaded["router"].map { NativeRouterModel(model: $0, executor: gpu, configuredTarget: target, contextEnabled: routerContextEnabled) },
            eureka: ServerEureka(connection: harness), harness: harness,
            output: NativeOutputModel(model: loaded["output"]!, executor: gpu, shortOutput: shortOutput),
            tts: NativeSpeechSynthesizer(model: loaded["tts"]!), speaker: MockSpeaker(), display: MockDisplay(), gpuExecutor: gpu)
        adapters.emotion = NativeEmotionClassifier(model: loaded["emotion"]!)
        if let item = env["DEMO_EUREKA_ITEM_ID"], let stage = env["DEMO_EUREKA_STAGE_ID"], !item.isEmpty, !stage.isEmpty {
            adapters.taskRefs = [HarnessTaskRef(itemId: item, stageId: stage, repository: repository, prNumber: number)]
        }

        var hardware: HardwareClient?
        var hardwareEvents: Task<Void, Never>?
        var hardwareMic: HardwareMicrophone?
        if let socket = config.hardwareSocket {
            let resolvedSocket = socket == "auto" ? (env["XDG_RUNTIME_DIR"].map { $0 + "/deskpet-hardware.sock" } ?? HardwareClient.defaultSocket) : resolve(socket)
            let client = try await HardwareClient.connect(path: resolvedSocket)
            hardware = client
            let display = HardwareDisplay(client: client)
            let speaker = HardwareSpeaker(client: client)
            let mic = HardwareMicrophone(client: client)
            hardwareMic = mic
            adapters.display = display
            adapters.speaker = speaker
            if !keyboard { adapters.inputEpochValidator = mic }
            hardwareEvents = Task {
                do {
                    for try await event in client.events {
                        if event.name == "audio.started", let id = event.fields["outputId"]?.string {
                            try await display.audioStarted(outputId: id)
                        }
                        if event.name == "device.error" || event.name == "audio.failed" {
                            print("hardware: \(event.name)")
                        }
                    }
                } catch {
                    print("hardware connection: \(error)")
                }
            }
        }
        defer { hardware?.close(); hardwareEvents?.cancel() }

        var timing = PipelineConfiguration()
        timing.routingMode = directLLM ? .directLLM : .model
        timing.confirmedTextInput = keyboard || arguments.contains("--text")
        timing.routerTimeout = config.routerTimeout ?? 15; timing.outputTimeout = config.outputTimeout ?? 15
        timing.sttTimeout = config.sttTimeout ?? 15; timing.ttsFirstChunkTimeout = config.ttsTimeout ?? 10
        timing.routeConfidenceThreshold = config.routeConfidenceThreshold ?? 0.3
        timing.sttConfidenceThreshold = config.sttConfidenceThreshold ?? 0.3
        timing.requestTimeout = 120
        let pipeline = try DeskPetPipeline(adapters: adapters, configuration: timing)
        await pipeline.start(automaticReminders: false)
        let logger = Task {
            for await event in pipeline.events {
                switch event {
                case .partial(let transcript): print("인식 중: \(transcript.payload.text)")
                case .transcriptFinal(let transcript):
                    if keyboard { print("입력: \(transcript.payload.text)") }
                    else { print("최종 인식: \(transcript.payload.text); STT \(transcript.payload.sttConfidence)") }
                case .output(let output): print("DeskPet: \(output.text)")
                case .action(_, let result):
                    print("업무 상태: \(result.status.rawValue)")
                    if let intent = result.facts["interpretedIntent"] { print("하네스 판단: \(intent)") }
                    if let reasons = result.facts["questionReasons"] { print("하네스 재질문 사유: \(reasons)") }
                case .routed(_, let decision):
                    print("라우팅: \(decision.headOutputs.intention.value.rawValue) → \(decision.route.rawValue); 종합 \(decision.overallConfidence), 의도 \(decision.headOutputs.intention.confidence)")
                case .error(_, let error):
                    let line = "오류: \(error.stage)/\(error.code) \(error.message)"
                    print(line)
                    FileHandle.standardError.write(Data((line + "\n").utf8))
                default: break
                }
            }
        }
        do {
            if let i = arguments.firstIndex(of: "--text"), arguments.count > i + 1 {
                _ = try await pipeline.submit(Envelope(conversationId: "native-demo", kind: "stt.final", payload: Transcript(text: arguments[i + 1], sttConfidence: 1)))
                while await pipeline.inFlightCount() > 0 { try await Task.sleep(for: .milliseconds(10)) }
            } else if keyboard {
                try await NativeDemo.keyboardSession(pipeline: pipeline, target: target,
                    reminder: (config.startupReminder ?? true) ? (config.startupReminderText ?? "PR \(number)번 검토를 잊지 마세요. 변경 내용을 확인해 드릴까요?") : nil)
            } else {
                guard arguments.contains("--live") else { throw PipelineError("NATIVE_MODE_REQUIRED", stage: "jetson") }
                guard let mic = hardwareMic else {
                    throw PipelineError("HARDWARE_MIC_REQUIRED", stage: "jetson", message: "Jetson 실시간 음성 입력에는 hardwareSocket 연결이 필요합니다.")
                }
                try await mic.start()
                if config.startupReminder ?? true {
                    try await pipeline.startScenarioReminder(target: target,
                        text: config.startupReminderText ?? "PR \(number)번 검토를 잊지 마세요. 변경 내용을 확인해 드릴까요?",
                        conversationId: "native-demo")
                }
                var ingress = AudioIngressConfiguration(); ingress.sampleRate = 16_000; ingress.maxSegmentBytes = 960_000
                ingress.microphoneTimeout = nil
                ingress.preRollBytes = 6_400
                let session = Task {
                    try await AudioIngress(microphone: mic, vad: NativeVAD(model: loaded["vad"]!), configuration: ingress).run(pipeline: pipeline, conversationId: "native-demo")
                }
                signal(SIGINT, SIG_IGN)
                let interrupt = DispatchSource.makeSignalSource(signal: Int32(SIGINT), queue: .global())
                interrupt.setEventHandler { session.cancel() }; interrupt.resume()
                defer { interrupt.cancel(); signal(SIGINT, SIG_DFL) }
                try await session.value
            }
        } catch {
            if let hardwareMic { try? await hardwareMic.stop() }
            await pipeline.shutdown(); await harness.close(); await logger.value; await NativeDemo.measurements(pipeline); throw error
        }
        if let hardwareMic { try? await hardwareMic.stop() }
        await pipeline.shutdown(); await harness.close(); await logger.value; await NativeDemo.measurements(pipeline)
    }

    private static func welcomed(_ url: URL, token: String) async -> Bool {
        let session = URLSession(configuration: .ephemeral)
        let socket = session.webSocketTask(with: url); socket.maximumMessageSize = 32_000; socket.resume()
        defer { socket.cancel(with: .goingAway, reason: nil); session.invalidateAndCancel() }
        do {
            return try await withThrowingTaskGroup(of: Bool.self) { group in
                group.addTask {
                    let hello = try JSONSerialization.data(withJSONObject: ["type": "hello", "token": token, "role": "pipeline", "client": "deskpet-jetson-launcher"])
                    try await socket.send(.string(String(decoding: hello, as: UTF8.self)))
                    let message = try await socket.receive()
                    let data: Data
                    switch message { case .data(let bytes): data = bytes; case .string(let text): data = Data(text.utf8); @unknown default: return false }
                    let frame = try JSONSerialization.jsonObject(with: data) as? [String: Any]
                    return frame?["type"] as? String == "welcome" && frame?["protocol"] as? String == "deskpet.harness.v1"
                }
                group.addTask { try await Task.sleep(for: .seconds(1)); socket.cancel(with: .goingAway, reason: nil); return false }
                defer { group.cancelAll() }
                return try await group.next() ?? false
            }
        } catch { return false }
    }
}
