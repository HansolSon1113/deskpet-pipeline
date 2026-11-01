import Foundation
import CNativeInference

/// This manifest describes an implementation, not a public llama/ggml type.
public struct NativeModelConfiguration: Codable, Sendable {
    public var backend: String
    public var role: String
    public var device: String
    public var paths: [String: String]
    public var library: String?
    public var contextTokens: Int = 2048
    public var maxTokens: Int = 128
    public var threads: Int = 2
    public init(backend: String = "llama.cpp", role: String, device: String, paths: [String: String]) {
        self.backend = backend; self.role = role; self.device = device; self.paths = paths
    }
}

public final class NativeBackendLibrary: @unchecked Sendable {
    fileprivate let handle: OpaquePointer
    public init(path: String) throws {
        var error: UnsafeMutablePointer<CChar>?
        guard let handle = dp_library_open(path, &error) else {
            defer { dp_string_free(error) }
            throw PipelineError("NATIVE_LIBRARY_UNAVAILABLE", stage: "native.load", message: error.map { String(cString: $0) } ?? path)
        }
        self.handle = handle
    }
    deinit { dp_library_close(handle) }
}

public final class NativeInferenceModel: @unchecked Sendable {
    private let library: NativeBackendLibrary
    private let handle: UnsafeMutableRawPointer
    public let configuration: NativeModelConfiguration
    public init(library: NativeBackendLibrary, configuration: NativeModelConfiguration) throws {
        if ["router", "output"].contains(configuration.role) {
            guard ["metal", "cuda"].contains(configuration.device) else {
                throw PipelineError("GPU_REQUIRED", stage: "native.load")
            }
        }
        self.library = library; self.configuration = configuration
        let json = String(decoding: try JSONEncoder().encode(configuration), as: UTF8.self)
        var error: UnsafeMutablePointer<CChar>?
        guard let handle = dp_model_create(library.handle, json, &error) else {
            defer { dp_string_free(error) }
            throw PipelineError("MODEL_LOAD_FAILED", stage: configuration.role, message: error.map { String(cString: $0) } ?? "")
        }
        self.handle = handle
    }
    deinit { dp_model_destroy(library.handle, handle) }
    fileprivate func submit<Input: Encodable>(_ input: Input) throws -> NativeInferenceJob {
        try Task.checkCancellation()
        let json = String(decoding: try JSONEncoder().encode(input), as: UTF8.self)
        var error: UnsafeMutablePointer<CChar>?
        guard let job = dp_model_submit(library.handle, handle, json, &error) else {
            defer { dp_string_free(error) }
            throw PipelineError("NATIVE_SUBMIT_FAILED", stage: configuration.role, message: error.map { String(cString: $0) } ?? "")
        }
        return NativeInferenceJob(model: self, library: library, handle: job)
    }
    public func run<Input: Encodable & Sendable, Output: Decodable & Sendable>(_ input: Input, as: Output.Type) async throws -> Output {
        guard !["router", "output"].contains(configuration.role) else { throw PipelineError("GPU_EXECUTOR_REQUIRED", stage: configuration.role) }
        let job = try submit(input)
        return try await withTaskCancellationHandler {
            // Keep the job/model/library alive until physical completion, even on cancellation.
            while true {
                if let result = job.poll() {
                    return try await decode(result, as: Output.self)
                }
                if Task.isCancelled {
                    job.cancel()
                    // A cancelled Task.sleep returns immediately. Keep waiting
                    // for physical completion without spinning a CPU core.
                    await withCheckedContinuation { continuation in
                        DispatchQueue.global().asyncAfter(deadline: .now() + 0.002) { continuation.resume() }
                    }
                } else { try? await Task.sleep(for: .milliseconds(2)) }
            }
        } onCancel: { job.cancel() }
    }
    public func gpu<Input: Encodable & Sendable, Output: Decodable & Sendable>(
        _ input: Input, executor: GPUExecutor, as: Output.Type
    ) async throws -> Output {
        guard ["router", "output"].contains(configuration.role), ["metal", "cuda"].contains(configuration.device) else {
            throw PipelineError("GPU_REQUIRED", stage: configuration.role)
        }
        let operation = GPUOperation<Data> { [self] in
            let job = try submit(input)
            return GPUCompletion(cancel: { job.cancel() }, poll: { job.poll() })
        }
        let result: Result<Data, any Error>
        do { result = .success(try await executor.run(operation)) }
        catch { result = .failure(error) }
        return try await decode(result, as: Output.self)
    }
    private func decode<Output: Decodable>(_ result: Result<Data, any Error>, as: Output.Type) async throws -> Output {
        switch result {
        case .success(let data):
            await NativeConsole.modelResult(configuration.role, payload: data)
        case .failure(let error):
            if let native = error as? PipelineError, native.code == "NATIVE_INFERENCE_FAILED" {
                await NativeConsole.write(configuration.role, payload: Data(native.message.utf8))
            }
        }
        try Task.checkCancellation()
        return try JSONDecoder().decode(Output.self, from: result.get())
    }
}

private final class NativeInferenceJob: @unchecked Sendable {
    let model: NativeInferenceModel
    let library: NativeBackendLibrary
    let handle: UnsafeMutableRawPointer
    init(model: NativeInferenceModel, library: NativeBackendLibrary, handle: UnsafeMutableRawPointer) {
        self.model = model; self.library = library; self.handle = handle
    }
    func cancel() { dp_model_cancel(library.handle, handle) }
    func poll() -> Result<Data, any Error>? {
        var result: UnsafePointer<CChar>?; var error: UnsafePointer<CChar>?
        let status = dp_model_poll(library.handle, handle, &result, &error)
        guard status != 0 else { return nil }
        guard status == 1, let result else {
            return .failure(PipelineError("NATIVE_INFERENCE_FAILED", stage: model.configuration.role,
                message: error.map { String(cString: $0) } ?? ""))
        }
        return .success(Data(String(cString: result).utf8))
    }
    deinit { dp_model_release_job(library.handle, handle) }
}

public struct NativeOutputModel: OutputModel {
    public let model: NativeInferenceModel
    public let executor: GPUExecutor
    public init(model: NativeInferenceModel, executor: GPUExecutor) { self.model = model; self.executor = executor }
    public func generate(_ request: OutputRequest) async throws -> String {
        guard request.action == nil, request.reminder == nil, !request.isQuestion, !request.messages.isEmpty else { return request.requiredText }
        struct Reply: Decodable, Sendable { var text: String }
        struct Input: Encodable, Sendable { var messages: [ChatMessage] }
        let result = try await model.gpu(Input(messages: request.messages), executor: executor, as: Reply.self)
        return result.text
    }
}

public struct NativeRouterModel: RouterModel {
    public let model: NativeInferenceModel
    public let executor: GPUExecutor
    public let configuredTarget: Target?
    public let contextEnabled: Bool
    public init(model: NativeInferenceModel, executor: GPUExecutor, configuredTarget: Target? = nil, contextEnabled: Bool = true) {
        self.model = model; self.executor = executor; self.configuredTarget = configuredTarget
        self.contextEnabled = contextEnabled
    }
    public func resolveTarget(in transcript: Transcript) async throws -> TargetResolution {
        let text = transcript.text.lowercased()
        guard let configuredTarget, text.range(of: "pr|풀[ ]?리퀘스트|깃허브|리뷰", options: .regularExpression) != nil else { return .omitted }
        if let match = text.range(of: "(?:pr[ #]*|풀[ ]?리퀘스트[ #]*)([0-9]+)", options: .regularExpression) {
            let digits = text[match].filter(\.isNumber)
            if let number = Int(digits), number > 0, let repository = configuredTarget.repository {
                return .unique(Target(kind: "github_pr", entityId: "\(repository["owner"] ?? "")/\(repository["name"] ?? "")#\(number)",
                                      name: "PR \(number)", repository: repository, prNumber: number))
            }
        }
        return .unique(configuredTarget)
    }
    public func heads(for input: RouterInput) async throws -> HeadOutputs {
        struct Input: Encodable, Sendable {
            var messages: [ChatMessage]
            var contextMode: String
            var currentMessageIndex: Int
        }
        let history = contextEnabled ? input.turns.map { ChatMessage(role: $0.role, content: $0.content) } : []
        let messages = history + [.init(role: "user", content: input.transcript.text)]
        // Preserve all original role turns. Metadata identifies the final query;
        // no instructions, summaries or marker text enter the trained input.
        // legacy also prevents injection by an older, not-yet-rebuilt plugin.
        var result = try await model.gpu(Input(messages: messages, contextMode: "legacy",
            currentMessageIndex: history.count), executor: executor, as: HeadOutputs.self)
        try NativeRouterContract.validate(result, messages: messages)
        result = try NativeRouterContract.bindExecutionIntent(result)
        if let raw = result.spans?["DueAt"]?.text {
            result.dueAt.value = NativeRouterContract.date(raw, relativeTo: input.transcript.audioEndedAt)
        }
        if input.pendingClarification?.pendingId != nil, let answer = NativeRouterContract.answer(input.transcript.text) {
            result.intention.value = answer ? .confirmApprove : .confirmReject
        }
        return result
    }
    public func confidence(for heads: HeadOutputs, missingSlots: [String]) async throws -> Double {
        guard let score = heads.modelConfidence, score.isFinite, (0...1).contains(score) else {
            throw PipelineError("ROUTER_CONFIDENCE_UNAVAILABLE", stage: "router")
        }
        return score
    }
}

/// Backend-independent semantics; TensorRT returns the same trained head data.
public enum NativeRouterContract {
    /// Bind the original heads, including results from older backend plugins.
    /// Executor availability must never turn an eureka prediction into llm.
    public static func bindExecutionIntent(_ heads: HeadOutputs) throws -> HeadOutputs {
        var result = heads
        guard let prediction = heads.modelIntention?.value else {
            throw PipelineError("ROUTER_CONTRACT_INVALID", stage: "router")
        }
        switch prediction {
        case .eureka:
            guard let operation = heads.modelOperation else {
                throw PipelineError("ROUTER_OPERATION_UNSUPPORTED", stage: "router",
                    message: heads.operation?.value ?? "Operation이 없습니다.")
            }
            result.intention.value = operation.executionIntent
        case .llm: result.intention.value = .harnessTask
        case .talk: result.intention.value = .chat
        case .unclear: result.intention.value = .unclear
        }
        return result
    }
    public static func answer(_ text: String) -> Bool? {
        let value = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if value.range(of: "^(응|네|예)([ ,.!]*승인해(줘|요)?)?[.! ]*$|^(승인|완료|반영|진행)(해(줘|요)?)?[.! ]*$", options: .regularExpression) != nil { return true }
        if value.range(of: "^(아니(요)?|거절(해(줘|요)?)?|취소(해(줘|요)?)?|하지[ ]?마(세요)?)[.! ]*$", options: .regularExpression) != nil { return false }
        return nil
    }
    public static func validate(_ heads: HeadOutputs, messages: [ChatMessage]) throws {
        guard let modelIntent = heads.modelIntention, let confidence = heads.modelConfidence,
              confidence.isFinite, (0...1).contains(confidence),
              modelIntent.confidence.isFinite, (0...1).contains(modelIntent.confidence) else {
            throw PipelineError("ROUTER_CONTRACT_INVALID", stage: "router")
        }
        if modelIntent.value == .eureka {
            guard heads.modelOperation != nil, let score = heads.operation?.confidence,
                  score.isFinite, (0...1).contains(score) else {
                throw PipelineError("ROUTER_OPERATION_UNSUPPORTED", stage: "router",
                    message: heads.operation?.value ?? "Operation이 없습니다.")
            }
        }
        for span in (heads.spans ?? [:]).values {
            guard messages.indices.contains(span.message), span.start >= 0, span.end > span.start else {
                throw PipelineError("ROUTER_SPAN_INVALID", stage: "router")
            }
            let chars = Array(messages[span.message].content.unicodeScalars)
            guard span.end <= chars.count, String(String.UnicodeScalarView(chars[span.start..<span.end])) == span.text else {
                throw PipelineError("ROUTER_SPAN_INVALID", stage: "router")
            }
        }
        guard (heads.generationFinished ?? [:]).values.allSatisfy({ $0 }) else {
            throw PipelineError("ROUTER_GENERATION_TRUNCATED", stage: "router")
        }
    }
    public static func date(_ text: String, relativeTo reference: Date) -> Date? {
        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let value = iso.date(from: text) ?? ISO8601DateFormatter().date(from: text) { return value }
        let format = DateFormatter(); format.locale = Locale(identifier: "ko_KR"); format.isLenient = false
        format.timeZone = TimeZone(identifier: "Asia/Seoul"); format.dateFormat = "yyyy-MM-dd"
        if text.range(of: "^[0-9]{4}-[0-9]{2}-[0-9]{2}$", options: .regularExpression) != nil, let value = format.date(from: text) { return value }
        // Ambiguous free-form dates remain raw spans and require clarification.
        var calendar = Calendar(identifier: .gregorian); calendar.timeZone = format.timeZone
        let days = ["오늘": 0, "내일": 1, "모레": 2]
        guard let offset = days[text], let date = calendar.date(byAdding: .day, value: offset, to: reference) else { return nil }
        return calendar.date(bySettingHour: 23, minute: 59, second: 0, of: date)
    }
}

public actor NativeStreamingSTT: StreamingSpeechRecognizer {
    private let model: NativeInferenceModel
    private struct Session { var pcm = Data(); var first: Date; var lastPartialBytes = 0 }
    private var sessions: [UUID: Session] = [:]
    public init(model: NativeInferenceModel) { self.model = model }
    private func recognize(_ pcm: Data, start: Date, end: Date, final: Bool) async throws -> Transcript {
        struct Input: Encodable, Sendable { var pcm: String; var language = "ko" }
        struct Reply: Codable, Sendable { var text: String; var confidence: Double }
        let result = try await model.run(Input(pcm: pcm.base64EncodedString()), as: Reply.self)
        if final && !result.text.isEmpty { await NativeConsole.value("stt", result) }
        return Transcript(text: result.text, isFinal: final, sttConfidence: result.confidence,
                          audioStartedAt: start, audioEndedAt: end)
    }
    public func transcribe(_ segment: AudioSegment) async throws -> Transcript {
        try await recognize(segment.pcm, start: segment.startedAt, end: segment.endedAt, final: true)
    }
    public func accept(_ frame: STTFrame) async throws -> Transcript? {
        guard frame.frame.chunk.sampleRate == 16_000, frame.frame.chunk.channels == 1 else {
            throw PipelineError("STT_FORMAT", stage: "stt")
        }
        if frame.activity == .silence { return nil }
        guard sessions[frame.sessionId] != nil || sessions.count < 2 else { throw PipelineError("STT_BUSY", stage: "stt") }
        var s = sessions[frame.sessionId] ?? Session(first: frame.frame.timestamp)
        guard s.pcm.count + frame.frame.chunk.data.count <= 960_000 else {
            sessions.removeValue(forKey: frame.sessionId); throw PipelineError("AUDIO_SEGMENT_OVERFLOW", stage: "stt")
        }
        s.pcm.append(frame.frame.chunk.data)
        let final = frame.activity == .end
        let recognizeNow = final || s.pcm.count - s.lastPartialBytes >= 32_000
        s.lastPartialBytes = recognizeNow ? s.pcm.count : s.lastPartialBytes
        sessions[frame.sessionId] = s
        guard recognizeNow else { return nil }
        let result = try await recognize(s.pcm, start: s.first, end: frame.frame.timestamp, final: final)
        guard sessions[frame.sessionId] != nil else { throw CancellationError() }
        if final { sessions.removeValue(forKey: frame.sessionId) }
        return result.text.isEmpty ? nil : result
    }
    public func cancel(sessionId: UUID) { sessions.removeValue(forKey: sessionId) }
}

public struct NativeVAD: ResettableVoiceActivityDetector {
    public let model: NativeInferenceModel
    public init(model: NativeInferenceModel) { self.model = model }
    public func reset() async throws {
        struct Input: Encodable, Sendable { var reset = true }
        struct Reply: Decodable, Sendable { var activity: String }
        _ = try await model.run(Input(), as: Reply.self)
    }
    public func classify(_ frame: AudioFrame) async throws -> VoiceActivity {
        struct Input: Encodable, Sendable { var pcm: String }
        struct Reply: Decodable, Sendable { var activity: String }
        let r = try await model.run(Input(pcm: frame.chunk.data.base64EncodedString()), as: Reply.self)
        switch r.activity { case "speech": return .speech; case "end": return .end; default: return .silence }
    }
}

public struct NativeEmotionClassifier: EmotionClassifier {
    public let model: NativeInferenceModel
    public init(model: NativeInferenceModel) { self.model = model }
    public func emotion(for text: String) async throws -> String {
        struct Input: Encodable, Sendable { var text: String; var words: [String] }
        struct Reply: Decodable, Sendable { var positive: Double }
        let normalized = text.precomposedStringWithCanonicalMapping
        let r = try await model.run(Input(text: normalized, words: Self.words(normalized)), as: Reply.self)
        return r.positive >= 0.8 ? "cheerful" : r.positive <= 0.2 ? "sad" : "neutral"
    }
    /// BertNormalizer(clean_text=true, lowercase=false, strip_accents=false)
    /// and BertPreTokenizer for the pinned NSMC tokenizer.
    static func words(_ text: String) -> [String] {
        var words: [String] = []; var current = ""
        func flush() { if !current.isEmpty { words.append(current); current = "" } }
        for scalar in text.unicodeScalars {
            let n = scalar.value; let category = scalar.properties.generalCategory
            let space = n == 9 || n == 10 || n == 13 || n == 32 || category == .spaceSeparator
            if space { flush(); continue }
            if n == 0 || n == 0xfffd || [.control, .format, .privateUse, .surrogate, .unassigned].contains(category) { continue }
            let chinese = (0x4e00...0x9fff).contains(n) || (0x3400...0x4dbf).contains(n) || (0x20000...0x2a6df).contains(n)
                || (0x2a700...0x2b73f).contains(n) || (0x2b740...0x2b81f).contains(n) || (0x2b820...0x2ceaf).contains(n)
                || (0xf900...0xfaff).contains(n) || (0x2f800...0x2fa1f).contains(n)
            let asciiPunctuation = (33...47).contains(n) || (58...64).contains(n) || (91...96).contains(n) || (123...126).contains(n)
            let punctuation = [.connectorPunctuation, .dashPunctuation, .openPunctuation, .closePunctuation, .initialPunctuation, .finalPunctuation, .otherPunctuation].contains(category)
            if chinese || asciiPunctuation || punctuation { flush(); words.append(String(scalar)) }
            else { current.unicodeScalars.append(scalar) }
        }
        flush(); return words
    }
}

public struct NativeSpeechSynthesizer: SpeechSynthesizer {
    public let model: NativeInferenceModel
    public init(model: NativeInferenceModel) { self.model = model }
    public func audio(for event: OutputEvent) async throws -> any AudioSource {
        struct Input: Encodable, Sendable { var text: String }
        struct Reply: Decodable, Sendable { var pcm: String; var sampleRate: Int }
        let r = try await model.run(Input(text: event.text), as: Reply.self)
        guard let data = Data(base64Encoded: r.pcm), !data.isEmpty, data.count <= 32_000_000 else {
            throw PipelineError("TTS_ERROR", stage: "tts")
        }
        return NativePCMSource(data: data, rate: r.sampleRate)
    }
}
private actor NativePCMSource: AudioSource {
    let data: Data; let rate: Int; var offset = 0
    init(data: Data, rate: Int) { self.data = data; self.rate = rate }
    func nextChunk() -> PCMChunk? {
        guard offset < data.count else { return nil }
        let end = min(data.count, offset + 16_384); defer { offset = end }
        return PCMChunk(data: data.subdata(in: offset..<end), sampleRate: rate, channels: 1)
    }
}
