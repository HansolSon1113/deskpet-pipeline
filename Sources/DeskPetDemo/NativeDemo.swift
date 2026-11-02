import Foundation
import DeskPetPipeline
import DeskPetHardware
#if os(macOS)
import Darwin
#endif

struct NativeDemo {
    struct Configuration: Decodable {
        var library: String
        var models: [String: NativeModelConfiguration]
        var harnessEnvFile: String?
        var hardwareSocket: String?
        var routerTimeout: TimeInterval?
        var outputTimeout: TimeInterval?
        var sttTimeout: TimeInterval?
        var ttsTimeout: TimeInterval?
        var routeConfidenceThreshold: Double?
        var sttConfidenceThreshold: Double?
        var routerContextEnabled: Bool?
        var directLLM: Bool?
        var shortOutput: Bool?
        var performanceDirectory: String?
        var startupReminder: Bool?
        var startupReminderText: String?
    }
    static func environment(_ file: String?) throws -> [String: String] {
        var result: [String: String] = [:]
        if let file {
            let source = try String(contentsOfFile: file, encoding: .utf8)
            for line in source.split(separator: "\n") where !line.hasPrefix("#") {
                let parts = line.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
                if parts.count == 2 { result[String(parts[0]).trimmingCharacters(in: .whitespaces)] = String(parts[1]).trimmingCharacters(in: .whitespaces) }
            }
        }
        return result.merging(ProcessInfo.processInfo.environment) { _, env in env }
    }
    static func run(_ arguments: [String]) async throws {
        guard let flag = arguments.firstIndex(of: "--native"), arguments.count > flag + 1 else {
            throw PipelineError("NATIVE_CONFIGURATION_REQUIRED", stage: "demo", message: "--native Models/native.local.json [--probe | --say TEXT | --text TEXT | --live | --keyboard]")
        }
        let configURL = URL(fileURLWithPath: arguments[flag + 1]).standardizedFileURL
        let root = configURL.deletingLastPathComponent()
        func resolve(_ path: String) -> String {
            if path.hasPrefix("/") { return path }
            return root.appendingPathComponent(path).standardizedFileURL.path
        }
        let config = try JSONDecoder().decode(Configuration.self, from: Data(contentsOf: configURL))
        let routerContextEnabled = (config.routerContextEnabled ?? true) && !arguments.contains("--no-router-context")
        let directLLM = (config.directLLM ?? false) || arguments.contains("--direct-llm")
        let shortOutput = (config.shortOutput ?? false) || arguments.contains("--short-output")
        let keyboard = arguments.contains("--keyboard")
        let roles = (keyboard ? ["router", "output", "emotion", "tts"] : ["router", "output", "stt", "vad", "emotion", "tts"])
            .filter { !directLLM || $0 != "router" }
        let performance = try performanceRecorder(config: config, arguments: arguments, configURL: configURL,
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
        if let flag = arguments.firstIndex(of: "--evaluate-router"), arguments.count > flag+1 {
            func argument(_ name: String) throws -> String {
                guard let i = arguments.firstIndex(of: name), arguments.count > i+1 else { throw PipelineError("EVALUATION_ARGUMENT_MISSING", stage: "evaluation", message: name) }
                return arguments[i+1]
            }
            var c = config.models["router"]!; c.paths = c.paths.mapValues(resolve)
            let path = resolve(c.library ?? config.library)
            let selected = try libraries[path] ?? NativeBackendLibrary(path: path)
            let limit = arguments.firstIndex(of: "--limit").flatMap { arguments.count > $0+1 ? Int(arguments[$0+1]) : nil }
            guard limit.map({ $0 > 0 }) ?? true else { throw PipelineError("EVALUATION_LIMIT_INVALID", stage: "evaluation") }
            try await RouterEvaluation.run(library: selected, configuration: c, executor: gpu, dataset: arguments[flag+1],
                split: try argument("--split"), reference: try argument("--reference"), report: try argument("--report"), limit: limit)
            await gpu.shutdown(); return
        }
        if arguments.contains("--probe-cpu") {
            for role in ["stt", "vad", "emotion", "tts"] { _ = try model(role) }
            await gpu.shutdown(); return
        }
        if let flag = arguments.firstIndex(of: "--recognize-pcm"), arguments.count > flag+1 {
            let pcm = try Data(contentsOf: URL(fileURLWithPath: arguments[flag+1]))
            guard !pcm.isEmpty, pcm.count <= 960_000, pcm.count.isMultiple(of: 2) else { throw PipelineError("STT_PCM_INVALID", stage: "stt") }
            let stt = NativeStreamingSTT(model: try model("stt"))
            _ = try await stt.transcribe(AudioSegment(pcm: pcm))
            await gpu.shutdown(); return
        }
        if let check = arguments.firstIndex(of: "--speech-check"), arguments.count > check + 1 {
            let text = arguments[check+1]
            let emotion = try await NativeEmotionClassifier(model: model("emotion")).emotion(for: text)
            let tts = NativeSpeechSynthesizer(model: try model("tts"))
            let event = OutputEvent(requestId: UUID(), outputId: UUID(), text: text, emotion: emotion,
                activity: .speaking, displayState: "normal", displayMode: .face, displayText: text, speech: SpeechPolicy())
            let source = try await tts.audio(for: event); var pcm = Data(); var rate = 0
            while let chunk = try await source.nextChunk() { pcm.append(chunk.data); rate = chunk.sampleRate }
            guard rate > 0, !pcm.isEmpty else { throw PipelineError("NO_AUDIO", stage: "tts") }
            if arguments.contains("--recognize-generated") {
                let mono16k = resampleForDiagnostic(pcm, rate: rate)
                let vad = NativeVAD(model: try model("vad")); try await vad.reset()
                for start in stride(from: 0, to: mono16k.count, by: 1024) {
                    let data = mono16k.subdata(in: start..<min(start+1024, mono16k.count))
                    _ = try await vad.classify(AudioFrame(chunk: PCMChunk(data: data, sampleRate: 16_000)))
                }
                _ = try await NativeStreamingSTT(model: model("stt")).transcribe(AudioSegment(pcm: mono16k))
            }
            if let flag = arguments.firstIndex(of: "--wav"), arguments.count > flag + 1 {
                var wave = Data("RIFF".utf8)
                func u32(_ value: Int) -> Data { var n = UInt32(value).littleEndian; return withUnsafeBytes(of: &n) { Data($0) } }
                wave.append(u32(36+pcm.count)); wave.append(Data("WAVEfmt ".utf8)); wave.append(u32(16))
                wave.append(contentsOf: [1,0,1,0]); wave.append(u32(rate)); wave.append(u32(rate*2))
                wave.append(contentsOf: [2,0,16,0]); wave.append(Data("data".utf8)); wave.append(u32(pcm.count)); wave.append(pcm)
                try wave.write(to: URL(fileURLWithPath: arguments[flag+1]), options: .atomic)
            }
            await gpu.shutdown(); return
        }
        if let say = arguments.firstIndex(of: "--say"), arguments.count > say + 1 {
            let output = try model("output")
            _ = try await NativeOutputModel(model: output, executor: gpu, shortOutput: shortOutput).generate(OutputRequest(requestId: UUID(), requiredText: "", messages: [ChatMessage(role: "user", content: arguments[say+1])]))
            await gpu.shutdown(); return
        }
        var loaded: [String: NativeInferenceModel] = [:]
        for role in roles { loaded[role] = try model(role) }
        if arguments.contains("--probe") { await gpu.shutdown(); return }
        let env = try environment(config.harnessEnvFile.map(resolve))
        guard let token = env["HARNESS_WS_TOKEN"], !token.isEmpty,
              let url = URL(string: env["HARNESS_WS_URL"] ?? "ws://127.0.0.1:8787"),
              let repo = env["GITHUB_TEST_REPO"]?.split(separator: "/"), repo.count == 2,
              let rawPR = env["GITHUB_TEST_PR"], let number = Int(rawPR), number > 0 else {
            throw PipelineError("DEMO_TARGET_CONFIGURATION_REQUIRED", stage: "demo",
                message: "기존 Harness 설정의 HARNESS_WS_TOKEN, GITHUB_TEST_REPO, GITHUB_TEST_PR이 필요합니다.")
        }
        let repository = ["owner": String(repo[0]), "name": String(repo[1])]
        let target = Target(kind: "github_pr", entityId: "\(repo.joined(separator: "/"))#\(number)", name: "PR \(number)", repository: repository, prNumber: number)
        let harness = try ServerHarness(url: url, token: token, performance: performance)
        let recognizer: any SpeechRecognizer
        if keyboard { recognizer = KeyboardOnlyRecognizer() }
        else { recognizer = NativeStreamingSTT(model: loaded["stt"]!) }
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
        var hardwareDisplay: HardwareDisplay?
        if let socket = config.hardwareSocket {
            let client = try await HardwareClient.connect(path: socket == "auto" ? HardwareClient.defaultSocket : resolve(socket)); hardware = client
            let display = HardwareDisplay(client: client); hardwareDisplay = display; adapters.display = display
        }
        defer { hardware?.close() }
        #if os(macOS)
        let mic = MacMicrophoneSource()
        if arguments.contains("--live") || keyboard {
            let display = hardwareDisplay
            adapters.speaker = MacSpeakerAdapter(microphone: mic, onStarted: { id in
                if let display { try? await display.audioStarted(outputId: id.uuidString) }
            })
            if !keyboard { adapters.inputEpochValidator = mic }
        }
        #endif
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
                case .transcriptFinal(let transcript):
                    if keyboard { await NativeConsole.write("input", payload: Data(transcript.payload.text.utf8)) }
                case .output(let output): await NativeConsole.write("DeskPet", payload: Data(output.text.utf8))
                case .action(_, let result):
                    await NativeConsole.value("action", result)
                case .routed(_, let decision):
                    await NativeConsole.value("route", decision.route)
                case .error(_, let error):
                    await NativeConsole.value("error", error)
                default: break
                }
            }
        }
        do {
            if let i = arguments.firstIndex(of: "--text"), arguments.count > i + 1 {
                _ = try await pipeline.submit(Envelope(conversationId: "native-demo", kind: "stt.final", payload: Transcript(text: arguments[i+1], sttConfidence: 1)))
                while await pipeline.inFlightCount() > 0 { try await Task.sleep(for: .milliseconds(10)) }
            } else if keyboard {
                try await keyboardSession(pipeline: pipeline, target: target,
                    reminder: (config.startupReminder ?? true) ? (config.startupReminderText ?? "PR \(number)번 검토를 잊지 마세요. 변경 내용을 확인해 드릴까요?") : nil)
            } else {
                #if os(macOS)
                guard arguments.contains("--live") else { throw PipelineError("NATIVE_MODE_REQUIRED", stage: "demo") }
                try await mic.start()
                if config.startupReminder ?? true {
                    try await pipeline.startScenarioReminder(target: target,
                        text: config.startupReminderText ?? "PR \(number)번 검토를 잊지 마세요. 변경 내용을 확인해 드릴까요?",
                        conversationId: "native-demo")
                }
                var ingress = AudioIngressConfiguration(); ingress.sampleRate = 16_000; ingress.maxSegmentBytes = 960_000
                ingress.microphoneTimeout = nil // Playback deliberately suppresses microphone delivery.
                ingress.preRollBytes = 6_400 // 200 ms at 16 kHz PCM16, bounded independently of STT.
                let session = Task {
                    try await AudioIngress(microphone: mic, vad: NativeVAD(model: loaded["vad"]!), configuration: ingress).run(pipeline: pipeline, conversationId: "native-demo")
                }
                signal(SIGINT, SIG_IGN)
                let interrupt = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global())
                interrupt.setEventHandler { session.cancel() }; interrupt.resume()
                defer { interrupt.cancel(); signal(SIGINT, SIG_DFL) }
                try await session.value
                #else
                throw PipelineError("USE_JETSON_HARDWARE_INPUT", stage: "demo")
                #endif
            }
        } catch {
            #if os(macOS)
            mic.stop()
            #endif
            await pipeline.shutdown(); await harness.close(); await logger.value; throw error
        }
        #if os(macOS)
        mic.stop()
        #endif
        await pipeline.shutdown(); await harness.close(); await logger.value
    }
    static func performanceRecorder(config: Configuration, arguments: [String], configURL: URL,
                                    roles: [String], directLLM: Bool) throws -> ModelPerformance {
        let parent: URL
        if let flag = arguments.firstIndex(of: "--metrics-dir") {
            guard arguments.count > flag + 1, !arguments[flag+1].hasPrefix("--") else {
                throw PipelineError("METRICS_DIRECTORY_REQUIRED", stage: "performance")
            }
            parent = URL(fileURLWithPath: arguments[flag+1]).standardizedFileURL
        } else if let configured = config.performanceDirectory {
            parent = configured.hasPrefix("/") ? URL(fileURLWithPath: configured)
                : configURL.deletingLastPathComponent().appendingPathComponent(configured).standardizedFileURL
        } else {
            parent = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(".build/performance")
        }
        let models = config.models.filter { roles.contains($0.key) }.mapValues { value in
            var value = value
            value.paths = value.paths.mapValues { path in
                path.hasPrefix("/") ? path : configURL.deletingLastPathComponent().appendingPathComponent(path).standardizedFileURL.path
            }
            return value
        }
        return try ModelPerformance(parent: parent, mode: directLLM ? "direct-llm" : "router",
            models: models, keyboard: arguments.contains("--keyboard"),
            routerContext: !directLLM && (config.routerContextEnabled ?? true) && !arguments.contains("--no-router-context"))
    }
    static func measurements(_ pipeline: DeskPetPipeline) async {
        for (stage, value) in await pipeline.measurements().sorted(by: { $0.key < $1.key }) {
            print("\(stage): n=\(value.count), p50=\(String(format: "%.1f", value.p50Milliseconds))ms, p95=\(String(format: "%.1f", value.p95Milliseconds))ms")
        }
        #if os(macOS)
        var usage = rusage()
        if getrusage(RUSAGE_SELF, &usage) == 0 { print("Process peak memory: \(usage.ru_maxrss) bytes") }
        #endif
    }
    static func resampleForDiagnostic(_ pcm: Data, rate: Int) -> Data {
        // Diagnostic file conversion only. Live capture uses AVAudioConverter.
        let input = pcm.withUnsafeBytes { raw -> [Double] in
            let bytes = raw.bindMemory(to: UInt8.self)
            return stride(from: 0, to: bytes.count, by: 2).map { i in Double(Int16(bitPattern: UInt16(bytes[i]) | UInt16(bytes[i+1]) << 8)) }
        }
        var output = Data(); output.reserveCapacity(input.count*32_000/rate)
        for i in 0..<(input.count*16_000/rate) {
            let x = Double(i)*Double(rate)/16_000; let a = Int(x); let b = min(a+1,input.count-1)
            var value = Int16(input[a]+(input[b]-input[a])*(x-Double(a))).littleEndian
            withUnsafeBytes(of: &value) { output.append(contentsOf: $0) }
        }
        return output
    }
}
