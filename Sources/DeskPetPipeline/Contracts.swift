import Foundation

public struct Envelope<Payload: Sendable & Codable>: Sendable, Codable {
    public var schemaVersion = "1.0"
    public var messageId = UUID()
    public var requestId: UUID
    public var conversationId: String
    public var turnId: UUID
    public var parentId: UUID?
    public var createdAt = Date()
    public var deadlineAt: Date?
    public var kind: String
    public var payload: Payload
    public init(requestId: UUID = UUID(), conversationId: String, turnId: UUID = UUID(),
                parentId: UUID? = nil, deadlineAt: Date? = nil, kind: String, payload: Payload) {
        self.requestId = requestId; self.conversationId = conversationId; self.turnId = turnId
        self.parentId = parentId; self.deadlineAt = deadlineAt; self.kind = kind; self.payload = payload
    }
}
public struct Transcript: Sendable, Codable {
    public var text: String
    public var isFinal: Bool
    public var sttConfidence: Double
    public var language: String
    public var audioStartedAt: Date
    public var audioEndedAt: Date
    public init(text: String, isFinal: Bool = true, sttConfidence: Double = 1,
                language: String = "ko-KR", audioStartedAt: Date = Date(), audioEndedAt: Date = Date()) {
        self.text = text; self.isFinal = isFinal; self.sttConfidence = sttConfidence
        self.language = language; self.audioStartedAt = audioStartedAt; self.audioEndedAt = audioEndedAt
    }
}
public enum Intent: String, Sendable, Codable {
    case register = "eureka.register", query = "eureka.query", complete = "eureka.complete"
    case postpone = "eureka.postpone", updatePriority = "eureka.priority"
    case mcp = "mcp.task", chat = "chat", unclear = "unclear"
    case reviewPR = "pr.review"
    case acknowledgeReminder = "reminder.acknowledge", snoozeReminder = "reminder.snooze"
    public var isMutation: Bool { [.register, .complete, .postpone, .updatePriority, .mcp].contains(self) }
}
public struct Head<Value: Sendable & Codable>: Sendable, Codable {
    public var value: Value
    public var confidence: Double
    public init(_ value: Value, confidence: Double = 0.99) { self.value = value; self.confidence = confidence }
}
public struct HeadOutputs: Sendable, Codable {
    public var intention: Head<Intent>
    public var target: Head<String?>
    public var name: Head<String?>
    public var dueAt: Head<Date?>
    public var priority: Head<String?>
    public var task: Head<String?>
    public var note: Head<String?>
    public init(intent: Intent, name: String? = nil, dueAt: Date? = nil, priority: String? = nil,
                task: String? = nil, note: String? = nil, confidence: Double = 0.99) {
        intention = Head(intent, confidence: confidence); target = Head(name == nil ? nil : "item")
        self.name = Head(name); self.dueAt = Head(dueAt); self.priority = Head(priority)
        self.task = Head(task); self.note = Head(note)
    }
}
public struct Target: Sendable, Codable, Equatable {
    public var kind: String
    public var entityId: String
    public var name: String
    public var repository: [String: String]?
    public var prNumber: Int?
    public init(kind: String = "item", entityId: String, name: String,
                repository: [String: String]? = nil, prNumber: Int? = nil) {
        self.kind = kind; self.entityId = entityId; self.name = name
        self.repository = repository; self.prNumber = prNumber
    }
    public func isSameEntity(as other: Target) -> Bool { kind == other.kind && entityId == other.entityId }
}
public enum TargetResolution: Sendable { case omitted, unique(Target), ambiguous }
public enum Route: String, Sendable, Codable { case eureka, harness, output, reminder }
public struct RouteDecision: Sendable, Codable {
    public var headOutputs: HeadOutputs
    public var overallConfidence: Double
    public var missingRequiredSlots: [String]
    public var route: Route
    public var sourceTurnIds: [UUID]
    public var resolvedTarget: Target?
    public init(headOutputs: HeadOutputs, overallConfidence: Double, missingRequiredSlots: [String],
                route: Route, sourceTurnIds: [UUID], resolvedTarget: Target? = nil) {
        self.headOutputs = headOutputs; self.overallConfidence = overallConfidence
        self.missingRequiredSlots = missingRequiredSlots; self.route = route
        self.sourceTurnIds = sourceTurnIds; self.resolvedTarget = resolvedTarget
    }
}
public enum ActionStatus: String, Sendable, Codable {
    case succeeded, failed, unknown, cancelled
    case needsClarification = "needs_clarification"
}
public struct PipelineError: Error, Sendable, Codable, Equatable {
    public var code: String
    public var stage: String
    public var message: String
    public var retryable: Bool
    public init(_ code: String, stage: String, message: String = "", retryable: Bool = false) {
        self.code = code; self.stage = stage; self.message = message; self.retryable = retryable
    }
}
public struct ActionRequest: Sendable, Codable {
    public var operationId: UUID
    public var requestId: UUID
    public var decision: RouteDecision
    public var deadlineAt: Date
    public init(operationId: UUID = UUID(), requestId: UUID, decision: RouteDecision, deadlineAt: Date) {
        self.operationId = operationId; self.requestId = requestId; self.decision = decision; self.deadlineAt = deadlineAt
    }
}
public struct ActionResult: Sendable, Codable {
    public var operationId: UUID
    public var status: ActionStatus
    public var action: Intent
    public var target: Target?
    public var facts: [String: String]
    public var error: PipelineError?
    public init(operationId: UUID = UUID(), status: ActionStatus, action: Intent = .chat,
                target: Target? = nil, facts: [String: String] = [:], error: PipelineError? = nil) {
        self.operationId = operationId; self.status = status; self.action = action
        self.target = target; self.facts = facts; self.error = error
    }
}
public struct HarnessRequest: Sendable {
    public var action: ActionRequest
    public var transcript: Transcript
    public var context: RouterInput
    public var wikiSearchAllowed = true
    // A real harness must enforce approval before mutation tools.
    public var mutationApproved = false
}
public struct HarnessResult: Sendable {
    public var action: ActionResult
    public var question: String?
    public var resultText: String?
    public init(action: ActionResult, question: String? = nil, resultText: String? = nil) {
        self.action = action; self.question = question; self.resultText = resultText
    }
}
public struct ReminderItem: Sendable, Codable {
    public var target: Target
    public var description: String
    public var dueAt: Date
    public var timeZone: String
    public var eligible: Bool
    public init(target: Target, description: String, dueAt: Date, timeZone: String = "Asia/Seoul", eligible: Bool = true) {
        self.target = target; self.description = description; self.dueAt = dueAt
        self.timeZone = timeZone; self.eligible = eligible
    }
    public var key: String {
        let epoch = (try? DateCodec.epochMilliseconds(dueAt)).map(String.init) ?? "invalid"
        return "\(target.entityId):\(epoch):due"
    }
}
public struct OutputRequest: Sendable {
    public var outputId = UUID()
    public var requestId: UUID
    public var action: ActionResult?
    public var reminder: ReminderItem?
    public var requiredText: String
    public var isQuestion = false
    public init(outputId: UUID = UUID(), requestId: UUID, action: ActionResult? = nil,
                reminder: ReminderItem? = nil, requiredText: String, isQuestion: Bool = false) {
        self.outputId = outputId; self.requestId = requestId; self.action = action
        self.reminder = reminder; self.requiredText = requiredText; self.isQuestion = isQuestion
    }
}
public enum Activity: String, Sendable, Codable { case idle, listening, processing, speaking; case awaitingAnswer = "awaiting_answer" }
public enum DisplayMode: String, Sendable, Codable { case face, text }
public struct SpeechPolicy: Sendable, Codable {
    public var enabled = true
    public var priority = "normal"
    public var voice = "default"
    public init(enabled: Bool = true, priority: String = "normal", voice: String = "default") {
        self.enabled = enabled; self.priority = priority; self.voice = voice
    }
}
public struct Motion: Sendable, Codable {
    public var name: String; public var intensity: Double
    public init(name: String, intensity: Double) { self.name = name; self.intensity = intensity }
}
public struct OutputEvent: Sendable, Codable {
    public var requestId: UUID
    public var outputId: UUID
    public var text: String
    public var emotion: String
    public var activity: Activity
    public var displayState: String
    public var displayMode: DisplayMode
    public var displayText: String
    public var speech: SpeechPolicy
    public var motion: Motion?
    public var actionStatus: ActionStatus?
    public var reminderKey: String?
    public init(requestId: UUID, outputId: UUID, text: String, emotion: String,
                activity: Activity, displayState: String, displayMode: DisplayMode,
                displayText: String, speech: SpeechPolicy, motion: Motion? = nil,
                actionStatus: ActionStatus? = nil, reminderKey: String? = nil) {
        self.requestId = requestId; self.outputId = outputId; self.text = text; self.emotion = emotion
        self.activity = activity; self.displayState = displayState; self.displayMode = displayMode
        self.displayText = displayText; self.speech = speech; self.motion = motion
        self.actionStatus = actionStatus; self.reminderKey = reminderKey
    }
}
public enum DeliveryStatus: String, Sendable, Codable { case completed, failed, suppressed }
public struct DeliveryResult: Sendable, Codable {
    public var outputId: UUID
    public var audio: DeliveryStatus
    public var display: DeliveryStatus
    public var errors: [PipelineError]
    public var delivered: Bool { audio == .completed || display == .completed }
}
public struct PCMChunk: Sendable { public var data: Data; public var sampleRate: Int; public var channels: Int
    public init(data: Data, sampleRate: Int = 24_000, channels: Int = 1) {
        self.data = data; self.sampleRate = sampleRate; self.channels = channels
    }
}
public struct AudioSegment: Sendable {
    public var pcm: Data
    public var startedAt: Date
    public var endedAt: Date
    public init(pcm: Data, startedAt: Date = Date(), endedAt: Date = Date()) {
        self.pcm = pcm; self.startedAt = startedAt; self.endedAt = endedAt
    }
}
public enum InteractionEvent: Sendable { case speechStarted(String), cancel(String), acknowledgeReminder, snoozeReminder(TimeInterval) }
public enum PipelineEvent: Sendable {
    case partial(Envelope<Transcript>), accepted(UUID), routed(UUID, RouteDecision)
    case action(UUID, ActionResult), output(OutputEvent), delivery(DeliveryResult)
    case error(UUID?, PipelineError), activity(String, Activity), questionWaitExpired(UUID, UUID)
}
public enum DateCodec {
    public static func rfc3339(_ date: Date) -> String { ISO8601DateFormatter().string(from: date) }
    public static func epochMilliseconds(_ date: Date) throws -> Int64 {
        let value = (date.timeIntervalSince1970 * 1000).rounded()
        guard value.isFinite, value >= Double(Int64.min), value < Double(Int64.max) else {
            throw PipelineError("INVALID_DATE", stage: "date")
        }
        return Int64(value)
    }
    public static func reminderDeadline(_ item: ReminderItem) -> String {
        humanDateTime(item.dueAt, timeZone: item.timeZone)
    }
    public static func humanDateTime(_ date: Date, timeZone: String = "Asia/Seoul") -> String {
        let formatter = DateFormatter(); formatter.locale = Locale(identifier: "ko_KR")
        formatter.timeZone = TimeZone(identifier: timeZone) ?? TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyy년 M월 d일 HH시 mm분"
        return "\(formatter.string(from: date)) (\(timeZone))"
    }
}

public enum ContractCodec {
    public static func encoder() -> JSONEncoder {
        let encoder = JSONEncoder(); encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys]; return encoder
    }
    public static func decoder() -> JSONDecoder {
        let decoder = JSONDecoder(); decoder.dateDecodingStrategy = .iso8601; return decoder
    }
}
