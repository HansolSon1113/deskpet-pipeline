import Foundation

public struct MockBehavior: Sendable {
    public var delay: TimeInterval
    public var error: PipelineError?
    public init(delay: TimeInterval = 0, error: PipelineError? = nil) { self.delay = delay; self.error = error }
    func run() async throws {
        try Task.checkCancellation()
        if delay > 0 { try await Task.sleep(for: .seconds(delay)) }
        if let error { throw error }
    }
}
public struct MockSTT: SpeechRecognizer {
    public var behavior: MockBehavior
    public init(behavior: MockBehavior = .init()) { self.behavior = behavior }
    public func transcribe(_ segment: AudioSegment) async throws -> Transcript {
        try await behavior.run()
        // Test-only: synthetic UTF-8 bytes stand in for PCM recognition.
        guard let text = String(data: segment.pcm, encoding: .utf8) else { throw PipelineError("STT_UNAVAILABLE", stage: "stt") }
        return Transcript(text: text, audioStartedAt: segment.startedAt, audioEndedAt: segment.endedAt)
    }
}
public actor MockStreamingSTT: StreamingSpeechRecognizer {
    private var sessions: [UUID: (data: Data, start: Date)] = [:]
    public let behavior: MockBehavior
    public init(behavior: MockBehavior = .init()) { self.behavior = behavior }
    public func transcribe(_ segment: AudioSegment) async throws -> Transcript {
        try await MockSTT(behavior: behavior).transcribe(segment)
    }
    public func accept(_ input: STTFrame) async throws -> Transcript? {
        try await behavior.run(); try Task.checkCancellation()
        switch input.activity {
        case .silence: return nil
        case .speech:
            guard sessions[input.sessionId] != nil || sessions.count < 64 else { throw PipelineError("MOCK_STT_BUSY", stage: "stt") }
            var value = sessions[input.sessionId] ?? (Data(), input.frame.timestamp)
            guard value.data.count + input.frame.chunk.data.count <= 2_000_000 else { throw PipelineError("INPUT_TOO_LARGE", stage: "stt") }
            value.data.append(input.frame.chunk.data); sessions[input.sessionId] = value
            return Transcript(text: String(decoding: value.data, as: UTF8.self), isFinal: false,
                              audioStartedAt: value.start, audioEndedAt: input.frame.timestamp)
        case .end:
            guard let value = sessions.removeValue(forKey: input.sessionId), !value.data.isEmpty else { return nil }
            return Transcript(text: String(decoding: value.data, as: UTF8.self),
                              audioStartedAt: value.start, audioEndedAt: input.frame.timestamp)
        }
    }
    public func cancel(sessionId: UUID) { sessions.removeValue(forKey: sessionId) }
}
public struct MockRoute: Sendable {
    public var heads: HeadOutputs
    public var resolution: TargetResolution
    public init(heads: HeadOutputs, resolution: TargetResolution = .omitted) {
        self.heads = heads; self.resolution = resolution
    }
}
public struct MockRouter: RouterModel {
    public var scripts: [String: MockRoute]
    public var behavior: MockBehavior
    public init(scripts: [String: MockRoute] = [:], behavior: MockBehavior = .init()) { self.scripts = scripts; self.behavior = behavior }
    public func resolveTarget(in transcript: Transcript) async throws -> TargetResolution {
        try await behavior.run(); return scripts[transcript.text]?.resolution ?? .omitted
    }
    public func heads(for input: RouterInput) async throws -> HeadOutputs {
        try await behavior.run(); return scripts[input.transcript.text]?.heads ?? HeadOutputs(intent: .chat)
    }
    public func confidence(for maskedHeads: HeadOutputs, missingSlots: [String]) async throws -> Double {
        try await behavior.run()
        return missingSlots.isEmpty ? maskedHeads.intention.confidence : 0.5
    }
}
public actor MockEureka: EurekaAdapter {
    private var items: [String: ReminderItem]
    private var receipts: [UUID: ActionResult] = [:]
    private var executionCount = 0
    private let behavior: MockBehavior
    private let loseMutationReply: Bool
    public init(items: [ReminderItem] = [], behavior: MockBehavior = .init(), loseMutationReply: Bool = false) {
        self.items = Dictionary(items.map { ($0.target.entityId, $0) }, uniquingKeysWith: { _, latest in latest })
        self.behavior = behavior; self.loseMutationReply = loseMutationReply
    }
    public func count() -> Int { executionCount }
    public func replace(_ item: ReminderItem) { items[item.target.entityId] = item }
    public func execute(_ request: ActionRequest) async throws -> ActionResult {
        try await behavior.run()
        if let receipt = receipts[request.operationId] { return receipt }
        if receipts.count >= 256 { throw PipelineError("MOCK_RECEIPT_FULL", stage: "eureka") }
        executionCount += 1
        let intent = request.decision.headOutputs.intention.value
        let target = request.decision.resolvedTarget
        let result: ActionResult
        let supported: Set<Intent> = [.register, .create, .postpone, .complete, .updatePriority, .setPriority, .query]
        guard supported.contains(intent) else {
            return ActionResult(operationId: request.operationId, status: .failed, action: intent, target: target,
                error: PipelineError("EUREKA_OPERATION_NOT_IMPLEMENTED", stage: "eureka", message: intent.rawValue))
        }
        let newName = intent == .create ? request.decision.headOutputs.task.value : request.decision.headOutputs.name.value
        if [.register, .create].contains(intent), let name = newName {
            let created = Target(entityId: UUID().uuidString, name: name)
            items[created.entityId] = ReminderItem(target: created, description: request.decision.headOutputs.task.value ?? name,
                                                  dueAt: request.decision.headOutputs.dueAt.value ?? Date().addingTimeInterval(86_400))
            result = ActionResult(operationId: request.operationId, status: .succeeded, action: intent, target: created)
        } else if let target, var item = items[target.entityId] {
            var facts: [String: String] = [:]
            switch intent {
            case .postpone:
                guard let due = request.decision.headOutputs.dueAt.value else { throw PipelineError("MISSING_SLOT", stage: "eureka") }
                item.dueAt = due; facts["dueAt"] = DateCodec.rfc3339(due)
            case .complete: item.eligible = false; facts["completed"] = "true"
            case .updatePriority, .setPriority: facts["priority"] = request.decision.headOutputs.priority.value
            case .query: facts = ["description": item.description, "dueAt": DateCodec.rfc3339(item.dueAt)]
            default: break
            }
            items[target.entityId] = item
            result = ActionResult(operationId: request.operationId, status: .succeeded, action: intent, target: target, facts: facts)
        } else {
            result = ActionResult(operationId: request.operationId, status: .failed, action: intent, target: target,
                                  error: PipelineError("EUREKA_NOT_FOUND", stage: "eureka"))
        }
        receipts[request.operationId] = result
        if loseMutationReply && intent.isMutation { throw PipelineError("REPLY_LOST", stage: "eureka") }
        return result
    }
    public func reconcile(_ request: ActionRequest) async throws -> ActionResult {
        try await behavior.run()
        return receipts[request.operationId] ?? ActionResult(operationId: request.operationId, status: .unknown,
            action: request.decision.headOutputs.intention.value, target: request.decision.resolvedTarget)
    }
    public func reminders() async throws -> [ReminderItem] { try await behavior.run(); return Array(items.values) }
    public func reminder(itemId: String) async throws -> ReminderItem? { try await behavior.run(); return items[itemId] }
}
public actor MockHarness: HarnessAdapter {
    private let behavior: MockBehavior
    private var calls = 0
    public init(behavior: MockBehavior = .init()) { self.behavior = behavior }
    public func count() -> Int { calls }
    public func run(_ request: HarnessRequest) async throws -> HarnessResult {
        calls += 1; try await behavior.run()
        guard request.action.decision.headOutputs.intention.value == .mcp else {
            return HarnessResult(action: ActionResult(operationId: request.action.operationId, status: .needsClarification,
                action: .unclear, target: request.action.decision.resolvedTarget), question: "어떤 작업을 원하시나요?")
        }
        return HarnessResult(action: ActionResult(operationId: request.action.operationId, status: .succeeded,
            action: .mcp, facts: ["mock": "true"]), resultText: "mock 도구에서 조회한 결과입니다.")
    }
}
public struct MockOutput: OutputModel {
    public var behavior: MockBehavior
    public var replacement: String?
    public init(behavior: MockBehavior = .init(), replacement: String? = nil) { self.behavior = behavior; self.replacement = replacement }
    public func generate(_ request: OutputRequest) async throws -> String {
        try await behavior.run(); return replacement ?? request.requiredText
    }
}
public actor MockAudioSource: AudioSource {
    private var remaining: Int
    private let behavior: MockBehavior
    public init(chunks: Int = 3, behavior: MockBehavior = .init()) { remaining = chunks; self.behavior = behavior }
    public func nextChunk() async throws -> PCMChunk? {
        try await behavior.run(); guard remaining > 0 else { return nil }
        remaining -= 1; return PCMChunk(data: Data(repeating: 0, count: 960))
    }
}
public struct MockTTS: SpeechSynthesizer {
    public var behavior: MockBehavior
    public init(behavior: MockBehavior = .init()) { self.behavior = behavior }
    public func audio(for event: OutputEvent) async throws -> any AudioSource { try await behavior.run(); return MockAudioSource() }
}
public actor MockSpeaker: SpeakerAdapter {
    private let behavior: MockBehavior
    private var texts: [String] = []
    private var chunks = 0
    private var currentText: [UUID: String] = [:]
    public init(behavior: MockBehavior = .init()) { self.behavior = behavior }
    public func spoken() -> [String] { texts }
    public func chunkCount() -> Int { chunks }
    public func begin(jobId: UUID, event: OutputEvent) async throws { try await behavior.run(); currentText[jobId] = event.text }
    public func write(_ chunk: PCMChunk, jobId: UUID) async throws { try await behavior.run(); chunks += 1 }
    public func finish(jobId: UUID) async throws {
        try await behavior.run()
        if let text = currentText.removeValue(forKey: jobId) {
            if texts.count == 256 { texts.removeFirst() }; texts.append(text)
        }
    }
    public func stop(jobId: UUID) async { currentText.removeValue(forKey: jobId) }
}
public actor MockDisplay: DisplayAdapter {
    private let behavior: MockBehavior
    private var outputs: [OutputEvent] = []
    public init(behavior: MockBehavior = .init()) { self.behavior = behavior }
    public func shown() -> [OutputEvent] { outputs }
    public func show(_ event: OutputEvent) async throws {
        try await behavior.run(); if outputs.count == 256 { outputs.removeFirst() }; outputs.append(event)
    }
    public func idle() async throws { try await behavior.run() }
}
public struct MockWeb: WebConsumer {
    public var behavior: MockBehavior
    public init(behavior: MockBehavior = .init()) { self.behavior = behavior }
    public func publish(_ event: OutputEvent) async throws { try await behavior.run() }
}
public extension Adapters {
    static func mocks(router: MockRouter = .init(), eureka: MockEureka = .init(), harness: MockHarness = .init(),
                      output: MockOutput = .init(), tts: MockTTS = .init(), speaker: MockSpeaker = .init(),
                      display: MockDisplay = .init(), web: MockWeb? = nil) -> Adapters {
        Adapters(stt: MockStreamingSTT(), router: router, eureka: eureka, harness: harness, output: output,
                 tts: tts, speaker: speaker, display: display, web: web)
    }
}

// These model fixtures are owned by their respective modules. The executor only
// receives opaque operations and never contains model routing or model state.
public struct MockGPURouter: RouterModel {
    public let executor: GPUExecutor
    public var scripts: [String: MockRoute]
    public var delay: Duration
    public init(executor: GPUExecutor, scripts: [String: MockRoute] = [:], delay: Duration = .zero) {
        self.executor = executor; self.scripts = scripts; self.delay = delay
    }
    public func resolveTarget(in transcript: Transcript) async throws -> TargetResolution {
        try await executor.run(MockGPUCall(value: scripts[transcript.text]?.resolution ?? .omitted, delay: delay).operation)
    }
    public func heads(for input: RouterInput) async throws -> HeadOutputs {
        try await executor.run(MockGPUCall(value: scripts[input.transcript.text]?.heads ?? HeadOutputs(intent: .chat), delay: delay).operation)
    }
    public func confidence(for heads: HeadOutputs, missingSlots: [String]) async throws -> Double {
        try await executor.run(MockGPUCall(value: missingSlots.isEmpty ? heads.intention.confidence : 0.5, delay: delay).operation)
    }
}
public struct MockGPUOutput: OutputModel {
    public let executor: GPUExecutor
    public var delay: Duration
    public init(executor: GPUExecutor, delay: Duration = .zero) { self.executor = executor; self.delay = delay }
    public func generate(_ request: OutputRequest) async throws -> String {
        try await executor.run(MockGPUCall(value: request.requiredText, delay: delay).operation)
    }
}
public extension Adapters {
    static func gpuMocks(executor: GPUExecutor, scripts: [String: MockRoute] = [:], delay: Duration = .zero) -> Adapters {
        Adapters(stt: MockStreamingSTT(), router: MockGPURouter(executor: executor, scripts: scripts, delay: delay),
                 eureka: MockEureka(), harness: MockHarness(), output: MockGPUOutput(executor: executor, delay: delay),
                 tts: MockTTS(), speaker: MockSpeaker(), display: MockDisplay(), gpuExecutor: executor)
    }
}
