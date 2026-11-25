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
    public var streaming: Bool?
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
    private let performance: ModelPerformance?
    public let configuration: NativeModelConfiguration
    public init(library: NativeBackendLibrary, configuration: NativeModelConfiguration, performance: ModelPerformance? = nil) throws {
        if ["router", "output"].contains(configuration.role) {
            guard ["metal", "cuda"].contains(configuration.device) else {
                throw PipelineError("GPU_REQUIRED", stage: "native.load")
            }
        }
        self.library = library; self.configuration = configuration; self.performance = performance
        let span = performance?.begin(role: configuration.role, backend: configuration.backend, device: configuration.device, phase: "load")
        var loaded = false
        defer { performance?.end(span, status: loaded ? "ok" : "failed") }
        let json = String(decoding: try JSONEncoder().encode(configuration), as: UTF8.self)
        var error: UnsafeMutablePointer<CChar>?
        guard let handle = dp_model_create(library.handle, json, &error) else {
            defer { dp_string_free(error) }
            throw PipelineError("MODEL_LOAD_FAILED", stage: configuration.role, message: error.map { String(cString: $0) } ?? "")
        }
        self.handle = handle
        loaded = true
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
    public func run<Input: Encodable & Sendable, Output: Decodable & Sendable>(_ input: Input, as: Output.Type, measurement: String = "inference") async throws -> Output {
        guard !["router", "output"].contains(configuration.role) else { throw PipelineError("GPU_EXECUTOR_REQUIRED", stage: configuration.role) }
        let span = performance?.begin(role: configuration.role, backend: configuration.backend, device: configuration.device, phase: measurement)
        var completed = false
        defer { if !completed { performance?.end(span, status: Task.isCancelled ? "cancelled" : "failed") } }
        let job = try submit(input)
        return try await withTaskCancellationHandler {
            // Keep the job/model/library alive until physical completion, even on cancellation.
            while true {
                if let result = job.poll() {
                    performance?.end(span, status: Self.status(result)); completed = true
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
        return try await enqueue(input, executor: executor, as: Output.self, measurement: "inference")
    }
    fileprivate func enqueue<Input: Encodable & Sendable, Output: Decodable & Sendable>(
        _ input: Input, executor: GPUExecutor, as: Output.Type, measurement: String = "inference"
    ) async throws -> Output {
        let span = performance?.begin(role: configuration.role, backend: configuration.backend, device: configuration.device, phase: measurement)
        let operation = GPUOperation<Data> { [self] in
            let job = try submit(input)
            return GPUCompletion(cancel: { job.cancel() }, poll: { job.poll() })
        }
        let result: Result<Data, any Error>
        do { result = .success(try await executor.run(operation)) }
        catch { result = .failure(error) }
        performance?.end(span, status: Self.status(result))
        return try await decode(result, as: Output.self)
    }
    private static func status(_ result: Result<Data, any Error>) -> String {
        if Task.isCancelled { return "cancelled" }
        switch result { case .success: return "ok"; case .failure: return "failed" }
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
    public static let shortOutputPrompt = """
    당신은 데스크펫 음성 비서입니다. 응답은 곧바로 TTS 음성으로 읽히므로 다음 규칙을 반드시 지키세요.
    1. 이모지, 이모티콘, 별표(*, **), 밑줄(_), 샵(#), 하이픈(-), 백틱(`) 등 기호나 마크다운 서식을 절대 사용하지 마세요.
    2. 소괄호(())나 대괄호([])를 사용한 부가 서술, 부연 설명, 병기를 절대 쓰지 마세요. 꼭 필요한 내용은 괄호 없이 문장으로 자연스럽게 풀어 쓰거나 생략하세요.
    3. 번호 매기기나 글머리 기호 목록 대신 소리 내어 읽기 자연스러운 한국어 평문 문장으로 이어서 답하세요.
    4. 추천은 기호를 묻거나 여러 개를 나열하지 말고 임의로 하나만 골라 답하고, 설명·방법·레시피는 핵심 항목(예: 필요한 재료나 준비물)과 요약만 간단히 답하세요.
    """
    public let model: NativeInferenceModel
    public let executor: GPUExecutor
    public let shortOutput: Bool
    public let contextProvider: OutputContextProvider?
    public init(model: NativeInferenceModel, executor: GPUExecutor, shortOutput: Bool = false, contextProvider: OutputContextProvider? = nil) {
        self.model = model
        self.executor = executor
        self.shortOutput = shortOutput
        self.contextProvider = contextProvider
    }
    public static func preparedMessages(for messages: [ChatMessage], shortOutput: Bool) -> [ChatMessage] {
        guard shortOutput, !messages.isEmpty else { return messages }
        if messages.first?.role == "system" { return messages }
        return [ChatMessage(role: "system", content: shortOutputPrompt)] + messages
    }
    public static func sanitizeForSpeech(_ text: String) -> String {
        var output = text
        for pattern in ["[ \\t]*\\([^()]*\\)", "[ \\t]*（[^（）]*）", "[ \\t]*\\[[^\\[\\]]*\\]"] {
            while let range = output.range(of: pattern, options: .regularExpression) {
                output.removeSubrange(range)
            }
        }
        output = output.replacingOccurrences(of: "(?m)^\\s*[-•+]\\s+", with: "", options: .regularExpression)
        let forbidden = CharacterSet(charactersIn: "*_`~#()（）[]")
        output = String(output.unicodeScalars.filter { !forbidden.contains($0) })
        output = output.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
        output = output.replacingOccurrences(of: "\\s+([,.!?:;])", with: "$1", options: .regularExpression)
        let trimmed = output.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? text : trimmed
    }
    public func generate(_ request: OutputRequest) async throws -> String {
        guard request.action == nil, request.reminder == nil, !request.isQuestion, !request.messages.isEmpty else { return request.requiredText }
        struct Reply: Decodable, Sendable { var text: String }
        struct Input: Encodable, Sendable { var messages: [ChatMessage] }
        var messages = Self.preparedMessages(for: request.messages, shortOutput: shortOutput)
        if let contextProvider {
            let ctx = await contextProvider.currentContextLine()
            if messages.first?.role == "system" {
                messages[0].content += "\n\(ctx)"
            } else {
                messages = [ChatMessage(role: "system", content: ctx)] + messages
            }
        }
        let result = try await model.gpu(Input(messages: messages), executor: executor, as: Reply.self)
        return shortOutput ? Self.sanitizeForSpeech(result.text) : result.text
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
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let value = iso.date(from: trimmed) ?? ISO8601DateFormatter().date(from: trimmed) { return value }

        let seoul = TimeZone(identifier: "Asia/Seoul") ?? .current
        let format = DateFormatter(); format.locale = Locale(identifier: "ko_KR"); format.isLenient = false
        format.timeZone = seoul; format.dateFormat = "yyyy-MM-dd"
        if trimmed.range(of: "^[0-9]{4}-[0-9]{2}-[0-9]{2}$", options: .regularExpression) != nil,
           let value = format.date(from: trimmed) { return value }

        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = seoul

        func firstGroups(_ pattern: String) -> [String]? {
            guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
            let ns = trimmed as NSString
            guard let match = regex.firstMatch(in: trimmed, range: NSRange(location: 0, length: ns.length)) else {
                return nil
            }
            return (0..<match.numberOfRanges).map { idx in
                let range = match.range(at: idx)
                return range.location == NSNotFound ? "" : ns.substring(with: range)
            }
        }

        func parseClockTime() -> (hour: Int, minute: Int)? {
            if let g = firstGroups("(오전|오후|저녁|밤|새벽|낮)?\\s*([0-9]{1,2})\\s*시(?!간)(?:\\s*([0-9]{1,2})\\s*분|\\s*(반))?"),
               var hour = Int(g[2]) {
                let meridiem = g[1]
                let minute = !g[3].isEmpty ? (Int(g[3]) ?? -1) : (!g[4].isEmpty ? 30 : 0)
                if (meridiem == "오후" || meridiem == "저녁" || meridiem == "밤"), (1...11).contains(hour) {
                    hour += 12
                } else if (meridiem == "오전" || meridiem == "새벽"), hour == 12 {
                    hour = 0
                }
                guard (0...23).contains(hour), (0...59).contains(minute) else { return nil }
                return (hour, minute)
            }
            if trimmed.contains("정오") { return (12, 0) }
            if trimmed.contains("자정") { return (23, 59) }
            return nil
        }

        func applyTime(_ date: Date, _ time: (hour: Int, minute: Int)?) -> Date? {
            let h = time?.hour ?? 23
            let m = time?.minute ?? 59
            return calendar.date(bySettingHour: h, minute: m, second: 0, of: date)
        }

        if let g = firstGroups("([0-9]+)\\s*시간\\s*(뒤|후)"), let n = Int(g[1]), n > 0 {
            return reference.addingTimeInterval(Double(n) * 3600)
        }
        if let g = firstGroups("([0-9]+)\\s*분\\s*(뒤|후)"), let n = Int(g[1]), n > 0 {
            return reference.addingTimeInterval(Double(n) * 60)
        }

        let time = parseClockTime()

        if let g = firstGroups("([0-9]{4})-([0-9]{2})-([0-9]{2})"),
           let y = Int(g[1]), let m = Int(g[2]), let d = Int(g[3]) {
            var comps = calendar.dateComponents([.year, .month, .day], from: reference)
            comps.year = y; comps.month = m; comps.day = d
            comps.hour = time?.hour ?? 23; comps.minute = time?.minute ?? 59; comps.second = 0
            return calendar.date(from: comps)
        }

        if let g = firstGroups("(?:([0-9]{4})\\s*년\\s*)?([0-9]{1,2})\\s*월\\s*([0-9]{1,2})\\s*일"),
           let m = Int(g[2]), let d = Int(g[3]), (1...12).contains(m), (1...31).contains(d) {
            let refComps = calendar.dateComponents([.year, .month, .day], from: reference)
            var year = !g[1].isEmpty ? (Int(g[1]) ?? refComps.year ?? 2026) : (refComps.year ?? 2026)
            if g[1].isEmpty, let rm = refComps.month, let rd = refComps.day, (m < rm || (m == rm && d < rd)) {
                year += 1
            }
            var comps = DateComponents(year: year, month: m, day: d, hour: time?.hour ?? 23, minute: time?.minute ?? 59, second: 0)
            comps.timeZone = seoul
            return calendar.date(from: comps)
        }

        if let g = firstGroups("([0-9]+)\\s*일\\s*(뒤|후)"), let n = Int(g[1]), n >= 0,
           let base = calendar.date(byAdding: .day, value: n, to: reference) {
            return applyTime(base, time)
        }
        if let g = firstGroups("([0-9]+)\\s*주\\s*(뒤|후)"), let n = Int(g[1]), n >= 0,
           let base = calendar.date(byAdding: .day, value: n * 7, to: reference) {
            return applyTime(base, time)
        }
        if let g = firstGroups("([0-9]+)\\s*(개월|달)\\s*(뒤|후)"), let n = Int(g[1]), n >= 0,
           let base = calendar.date(byAdding: .month, value: n, to: reference) {
            return applyTime(base, time)
        }

        let relativeDays: [(String, Int)] = [
            ("글피", 3), ("모레", 2),
            ("내일", 1), ("명일", 1),
            ("오늘", 0), ("금일", 0), ("당일", 0),
            ("어제", -1)
        ]
        for (keyword, offset) in relativeDays where trimmed.contains(keyword) {
            if let base = calendar.date(byAdding: .day, value: offset, to: reference) {
                return applyTime(base, time)
            }
        }

        let weekdays: [String: Int] = ["월": 0, "화": 1, "수": 2, "목": 3, "금": 4, "토": 5, "일": 6]
        let refMon = (calendar.component(.weekday, from: reference) + 5) % 7
        if let g = firstGroups("(?:(다다음[ ]?주|다음[ ]?주|차주|이번[ ]?주|금주)\\s*)?(월|화|수|목|금|토|일)요일"),
           let tgt = weekdays[g[2]] {
            let prefix = g[1].replacingOccurrences(of: " ", with: "")
            let delta: Int
            switch prefix {
            case "다다음주": delta = (tgt - refMon) + 14
            case "다음주", "차주": delta = (tgt - refMon) + 7
            case "이번주", "금주": delta = tgt - refMon
            default: delta = (tgt - refMon + 7) % 7
            }
            if let base = calendar.date(byAdding: .day, value: delta, to: reference) {
                return applyTime(base, time)
            }
        }

        if firstGroups("다음[ ]?주말") != nil,
           let base = calendar.date(byAdding: .day, value: (5 - refMon) + 7, to: reference) {
            return applyTime(base, time)
        }
        if firstGroups("(?:이번[ ]?)?주말") != nil,
           let base = calendar.date(byAdding: .day, value: (5 - refMon + 7) % 7, to: reference) {
            return applyTime(base, time)
        }
        if firstGroups("(?:이번[ ]?달[ ]?말|월말)") != nil,
           let range = calendar.range(of: .day, in: .month, for: reference) {
            var comps = calendar.dateComponents([.year, .month], from: reference)
            comps.day = range.upperBound - 1
            comps.hour = time?.hour ?? 23; comps.minute = time?.minute ?? 59; comps.second = 0
            return calendar.date(from: comps)
        }
        if firstGroups("(다음[ ]?주|차주)") != nil,
           let base = calendar.date(byAdding: .day, value: 7, to: reference) {
            return applyTime(base, time)
        }

        if let time {
            return applyTime(reference, time)
        }

        // Ambiguous free-form dates remain raw spans and require clarification.
        return nil
    }
}

public actor NativeStreamingSTT: StreamingSpeechRecognizer {
    private let model: NativeInferenceModel
    private let executor: GPUExecutor?
    private struct Session { var pendingPcm = Data(); var totalBytes = 0; var lastPartialBytes = 0; var first: Date }
    private var sessions: [UUID: Session] = [:]
    public init(model: NativeInferenceModel, executor: GPUExecutor? = nil) {
        self.model = model
        self.executor = executor
    }
    private func recognize(_ pcm: Data, sessionId: UUID?, action: String?, start: Date, end: Date, final: Bool) async throws -> Transcript {
        struct Input: Encodable, Sendable { var pcm: String; var language = "ko"; var sessionId: String?; var action: String? }
        struct Reply: Codable, Sendable { var text: String; var confidence: Double }
        let input = Input(pcm: pcm.base64EncodedString(), sessionId: sessionId?.uuidString, action: action)
        let measurement = final ? "final" : "partial"
        let result: Reply
        if let executor {
            result = try await model.enqueue(input, executor: executor, as: Reply.self, measurement: measurement)
        } else {
            result = try await model.run(input, as: Reply.self, measurement: measurement)
        }
        if final && !result.text.isEmpty { await NativeConsole.value("stt", result) }
        return Transcript(text: result.text, isFinal: final, sttConfidence: result.confidence,
                          audioStartedAt: start, audioEndedAt: end)
    }
    public func transcribe(_ segment: AudioSegment) async throws -> Transcript {
        try await recognize(segment.pcm, sessionId: nil, action: nil, start: segment.startedAt, end: segment.endedAt, final: true)
    }
    public func accept(_ frame: STTFrame) async throws -> Transcript? {
        guard frame.frame.chunk.sampleRate == 16_000, frame.frame.chunk.channels == 1 else {
            throw PipelineError("STT_FORMAT", stage: "stt")
        }
        if frame.activity == .silence { return nil }
        guard sessions[frame.sessionId] != nil || sessions.count < 2 else { throw PipelineError("STT_BUSY", stage: "stt") }
        var s = sessions[frame.sessionId] ?? Session(first: frame.frame.timestamp)
        guard s.totalBytes + frame.frame.chunk.data.count <= 960_000 else {
            await cancel(sessionId: frame.sessionId); throw PipelineError("AUDIO_SEGMENT_OVERFLOW", stage: "stt")
        }
        s.pendingPcm.append(frame.frame.chunk.data)
        s.totalBytes += frame.frame.chunk.data.count
        let final = frame.activity == .end
        let streaming = model.configuration.streaming ?? false
        let recognizeNow: Bool
        if streaming {
            recognizeNow = final || s.pendingPcm.count >= 6_400
        } else {
            recognizeNow = final || (s.pendingPcm.count - s.lastPartialBytes >= 32_000)
            if recognizeNow { s.lastPartialBytes = s.pendingPcm.count }
        }
        guard recognizeNow else {
            sessions[frame.sessionId] = s
            return nil
        }
        let chunk = s.pendingPcm
        if streaming { s.pendingPcm.removeAll(keepingCapacity: !final) }
        sessions[frame.sessionId] = s
        let result = try await recognize(
            chunk,
            sessionId: streaming ? frame.sessionId : nil,
            action: streaming ? (final ? "end" : "chunk") : nil,
            start: s.first,
            end: frame.frame.timestamp,
            final: final
        )
        guard sessions[frame.sessionId] != nil else { throw CancellationError() }
        if final { sessions.removeValue(forKey: frame.sessionId) }
        return result.text.isEmpty ? nil : result
    }
    public func cancel(sessionId: UUID) async {
        guard sessions.removeValue(forKey: sessionId) != nil else { return }
        if model.configuration.streaming ?? false {
            struct CancelInput: Encodable, Sendable { var sessionId: String; var action = "cancel" }
            struct Reply: Decodable, Sendable { var text: String }
            _ = try? await model.run(CancelInput(sessionId: sessionId.uuidString), as: Reply.self, measurement: "cancel")
        }
    }
}

public struct NativeVAD: ResettableVoiceActivityDetector {
    public let model: NativeInferenceModel
    public init(model: NativeInferenceModel) { self.model = model }
    public func reset() async throws {
        struct Input: Encodable, Sendable { var reset = true }
        struct Reply: Decodable, Sendable { var activity: String }
        _ = try await model.run(Input(), as: Reply.self, measurement: "reset")
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
    private let executor: GPUExecutor?
    public init(model: NativeInferenceModel, executor: GPUExecutor? = nil) {
        self.model = model
        self.executor = executor
    }
    public func emotion(for text: String) async throws -> String {
        struct Input: Encodable, Sendable { var text: String; var words: [String] }
        struct Reply: Decodable, Sendable { var positive: Double }
        let normalized = text.precomposedStringWithCanonicalMapping
        let r: Reply
        if let executor {
            r = try await model.enqueue(Input(text: normalized, words: Self.words(normalized)), executor: executor, as: Reply.self)
        } else {
            r = try await model.run(Input(text: normalized, words: Self.words(normalized)), as: Reply.self)
        }
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
    private let executor: GPUExecutor?
    public init(model: NativeInferenceModel, executor: GPUExecutor? = nil) {
        self.model = model
        self.executor = executor
    }
    public func audio(for event: OutputEvent) async throws -> any AudioSource {
        struct Input: Encodable, Sendable { var text: String }
        struct Reply: Decodable, Sendable { var pcm: String; var sampleRate: Int }
        let input = Input(text: event.text)
        let r: Reply
        if let executor {
            r = try await model.enqueue(input, executor: executor, as: Reply.self)
        } else {
            r = try await model.run(input, as: Reply.self)
        }
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
