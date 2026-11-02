import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import DeskPetPipeline
import DeskPetHardware
#if os(macOS)
import Darwin
#else
import Glibc
#endif

/// Composition and process lifetime only. All inference stays in the existing
/// native pipeline, and the existing Harness owns business decisions.
enum ProjectLauncher {
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
        PipelineError(code, stage: "project", message: message)
    }
    static func run(_ arguments: [String]) async throws {
        let options = Array(arguments.drop(while: { $0 != "--project" }).dropFirst())
        if options.contains("--help") {
            print("./run.command [--keyboard] [--direct-llm] [--metrics-dir DIR] [--no-router-context] [--short-output] [--config FILE] [--no-build] [--check] [--headless] [--read-only]")
            print("기존 모의 화면, Harness 서버, 파이프라인을 함께 실행합니다. --keyboard는 마이크 대신 터미널 입력을 사용합니다. Ctrl-C로 모두 종료합니다.")
            return
        }
        var customConfig: String?
        var metricsDirectory: String?
        var i = 0
        while i < options.count {
            if options[i] == "--config" || options[i] == "--metrics-dir" {
                guard i+1 < options.count, !options[i+1].hasPrefix("--") else { throw fault("CONFIG_REQUIRED", "\(options[i]) 뒤에 경로를 지정하세요.") }
                if options[i] == "--config" { customConfig = options[i+1] }
                else { metricsDirectory = options[i+1] }
                i += 2
            } else if ["--keyboard", "--direct-llm", "--no-router-context", "--short-output", "--no-build", "--check", "--headless", "--read-only"].contains(options[i]) { i += 1 }
            else { throw fault("UNKNOWN_OPTION", options[i]) }
        }
        let files = FileManager.default
        let root = URL(fileURLWithPath: files.currentDirectoryPath).standardizedFileURL
        let configURL = URL(fileURLWithPath: customConfig ?? root.appendingPathComponent("Models/native.local.json").path).standardizedFileURL
        if !files.fileExists(atPath: configURL.path), customConfig == nil {
            try files.copyItem(at: root.appendingPathComponent("Models/native.example.json"), to: configURL)
        }
        let config = try JSONDecoder().decode(NativeDemo.Configuration.self, from: Data(contentsOf: configURL))
        let modelRoot = configURL.deletingLastPathComponent()
        func resolve(_ path: String) -> String { path.hasPrefix("/") ? path : modelRoot.appendingPathComponent(path).standardizedFileURL.path }
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
            value == "auto" ? environment["XDG_RUNTIME_DIR"].map { $0+"/deskpet-hardware.sock" } ?? "/tmp/deskpet-\(getuid())/hardware.sock" : resolve(value)
        }
        let nativeLibrary = root.appendingPathComponent(".build/native/libdeskpet_inference.dylib").path
        let libraries = Set([resolve(config.library)] + roles.compactMap { config.models[$0] }.map { resolve($0.library ?? config.library) })
        for path in libraries where path != nativeLibrary && !files.fileExists(atPath: path) { throw fault("BACKEND_LIBRARY_MISSING", path) }
        if options.contains("--check") {
            print("실행 구성 준비 완료: 모델 \(roles.count)개 / 기존 Harness 설정 / \(socket == nil ? "진단 화면" : "하드웨어 모의 화면")")
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
            let source = DispatchSource.makeSignalSource(signal: number, queue: .global())
            source.setEventHandler { children.cancel() }; source.resume(); return source
        }
        defer { for source in signals { source.cancel() }; signal(SIGINT, SIG_DFL); signal(SIGTERM, SIG_DFL); signal(SIGPIPE, SIG_DFL) }
        func command(_ name: String, _ args: [String], directory: URL) async throws {
            if children.cancelled { throw CancellationError() }
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
            let log = logs.appendingPathComponent(name+".log"); files.createFile(atPath: log.path, contents: nil, attributes: [.posixPermissions: 0o600])
            let output = try FileHandle(forWritingTo: log); defer { try? output.close() }
            let process = Process(); process.executableURL = URL(fileURLWithPath: "/usr/bin/env"); process.arguments = args
            process.currentDirectoryURL = directory; process.environment = environment
            process.standardError = output; process.standardOutput = console ? FileHandle.standardOutput : output
            process.standardInput = input
            try process.run(); children.add(name, process); return process
        }
        do {
            if !options.contains("--no-build") {
                if libraries.contains(nativeLibrary) {
                    try await command("네이티브 모델 연결 빌드", ["sh", "tools/build-native.sh", "-DDESKPET_PREP=OFF"], directory: root)
                }
                if socket != nil {
                    var configure = ["sh", "tools/configure-hardware.sh", hardware.path, hardware.appendingPathComponent("build").path]
                    if options.contains("--headless") { configure.append("--headless") }
                    try await command("하드웨어 모의 화면 준비", configure, directory: root)
                    try await command("하드웨어 모의 화면 빌드", ["cmake", "--build", "build", "--target", "deskpet-hardware", "-j", "4"], directory: hardware)
                }
                if !files.fileExists(atPath: harness.appendingPathComponent("node_modules/tsx").path) {
                    try await command("기존 Harness 의존성 준비", ["pnpm", "install", "--frozen-lockfile"], directory: harness)
                }
            }
            for path in libraries where !files.fileExists(atPath: path) { throw fault("BACKEND_LIBRARY_MISSING", path) }
            if let socket {
                if files.fileExists(atPath: socket) { throw fault("HARDWARE_ALREADY_RUNNING", "기존 모의 화면을 종료한 뒤 다시 실행하세요: \(socket)") }
                var args = [hardware.appendingPathComponent("build/deskpet-hardware").path, "--config", "config/mock.json", "--socket", socket]
                if options.contains("--headless") { args.append("--headless") }
                let process = try start("hardware", args, directory: hardware)
                let until = ContinuousClock.now.advanced(by: .seconds(15))
                while true {
                    guard process.isRunning, ContinuousClock.now < until, !children.cancelled else { throw fault("HARDWARE_START_FAILED", "모의 화면을 시작하지 못했습니다. 로그: \(logs.path)") }
                    if files.fileExists(atPath: socket), let client = try? await HardwareClient.connect(path: socket) {
                        client.close(); break
                    }
                    try await Task.sleep(for: .milliseconds(100))
                }
            }
            var serverArgs = ["node", "--import", "tsx", "scripts/server.ts"]
            if !options.contains("--read-only") { serverArgs += ["--allow-github-writes", "--allow-eureka-writes"] }
            if await welcomed(serverURL, token: environment["HARNESS_WS_TOKEN"]!) {
                throw fault("HARNESS_ALREADY_RUNNING", "이미 Harness가 실행 중입니다. 기존 서버를 종료한 뒤 다시 실행하세요: \(serverURL.absoluteString)")
            }
            let server = try start("harness", serverArgs, directory: harness)
            let until = ContinuousClock.now.advanced(by: .seconds(15))
            while true {
                guard server.isRunning, ContinuousClock.now < until, !children.cancelled else { throw fault("HARNESS_START_FAILED", "Harness를 시작하지 못했습니다. 기존 서버 실행 여부와 로그를 확인하세요: \(logs.path)") }
                if await welcomed(serverURL, token: environment["HARNESS_WS_TOKEN"]!), server.isRunning { break }
                try await Task.sleep(for: .milliseconds(100))
            }
            let keyboardPipe = keyboard ? Pipe() : nil
            var pipelineArgs = [URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL.path, "--native", configURL.path, keyboard ? "--keyboard" : "--live"]
            if options.contains("--no-router-context") { pipelineArgs.append("--no-router-context") }
            if options.contains("--short-output") { pipelineArgs.append("--short-output") }
            if directLLM { pipelineArgs.append("--direct-llm") }
            if let metricsDirectory { pipelineArgs += ["--metrics-dir", URL(fileURLWithPath: metricsDirectory).standardizedFileURL.path] }
            let pipeline = try start("pipeline", pipelineArgs, directory: root, console: true, input: keyboardPipe?.fileHandleForReading ?? FileHandle.nullDevice)
            try? keyboardPipe?.fileHandleForReading.close()
            let keyboardRelay: DispatchSourceRead? = keyboardPipe.map { pipe in
                let source = DispatchSource.makeReadSource(fileDescriptor: STDIN_FILENO, queue: DispatchQueue(label: "deskpet.project.stdin"))
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
            if !children.cancelled, pipeline.terminationStatus != 0 { throw fault("PIPELINE_STOPPED", "파이프라인 종료 코드 \(pipeline.terminationStatus). 로그: \(logs.path)") }
            await children.stop()
        } catch {
            await children.stop()
            if children.cancelled { throw CancellationError() }
            throw error
        }
    }
    private static func welcomed(_ url: URL, token: String) async -> Bool {
        let session = URLSession(configuration: .ephemeral)
        let socket = session.webSocketTask(with: url); socket.maximumMessageSize = 32_000; socket.resume()
        defer { socket.cancel(with: .goingAway, reason: nil); session.invalidateAndCancel() }
        do {
            return try await withThrowingTaskGroup(of: Bool.self) { group in
                group.addTask {
                    let hello = try JSONSerialization.data(withJSONObject: ["type": "hello", "token": token, "role": "pipeline", "client": "deskpet-launcher"])
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
