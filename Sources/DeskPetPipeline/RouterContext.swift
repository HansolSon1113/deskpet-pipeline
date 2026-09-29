import Foundation

public struct ContextStamp: Sendable, Equatable {
    public var contextId: UUID
    public var version: Int
}
public struct ContextTurn: Sendable {
    public var role: String
    public var content: String
    public var turnId: UUID
    public var requestId: UUID
    public var outputId: UUID?
}
public struct PendingClarification: Sendable {
    public var requestId: UUID
    public var slots: [String]
    public var expiresAt: Date
}
public struct RouterInput: Sendable {
    public var transcript: Transcript
    public var stamp: ContextStamp
    public var turns: [ContextTurn]
    public var activeTarget: Target?
    public var pendingClarification: PendingClarification?
    public var contextDescription: String
    public var turnId: UUID
    public var requestId: UUID
    // Real adapters apply their tokenizer's chat template to roles and contextDescription.
}
struct RouteRecord: Sendable { var turnId: UUID; var decision: RouteDecision }
actor RouterContextStore {
    struct State {
        var id = UUID(); var version = 0
        var target: Target?; var turns: [ContextTurn] = []; var history: [RouteRecord] = []
        var pending: PendingClarification?; var updated = Date()
    }
    private var states: [String: State] = [:]
    private let config: PipelineConfiguration
    init(_ config: PipelineConfiguration) { self.config = config }
    private func state(_ conversation: String) -> State {
        if let value = states[conversation], Date().timeIntervalSince(value.updated) < config.contextExpiry { return value }
        if states.count >= config.maxConversations, states[conversation] == nil,
           let oldest = states.min(by: { $0.value.updated < $1.value.updated })?.key { states.removeValue(forKey: oldest) }
        let fresh = State(); states[conversation] = fresh; return fresh
    }
    func activeTarget(_ conversation: String) -> Target? { state(conversation).target }
    func pendingRequest(_ conversation: String) -> UUID? {
        let value = state(conversation)
        guard let pending = value.pending, pending.expiresAt > Date() else { return nil }
        return pending.requestId
    }
    func prepare(_ envelope: Envelope<Transcript>, resolution: TargetResolution) -> RouterInput {
        var value = state(envelope.conversationId)
        if case .unique(let target) = resolution,
           value.target.map({ !$0.isSameEntity(as: target) }) ?? true {
            if value.target == nil, value.pending != nil { value.target = target }
            else { value = State(); value.target = target }
        }
        value.updated = Date(); states[envelope.conversationId] = value
        var turns = value.turns
        if case .ambiguous = resolution { turns = [] }
        let active: Target? = { if case .ambiguous = resolution { return nil }; return value.target }()
        let description = active.map { "현재 대상: \($0.name). 이전 해석은 실행 성공을 뜻하지 않음." } ?? ""
        var budget = config.contextBytes - envelope.payload.text.utf8.count - description.utf8.count
        var selected: [ContextTurn] = []
        for turn in turns.reversed() {
            guard turn.content.utf8.count <= budget else { break }
            budget -= turn.content.utf8.count; selected.insert(turn, at: 0)
        }
        return RouterInput(transcript: envelope.payload, stamp: ContextStamp(contextId: value.id, version: value.version),
                           turns: selected, activeTarget: active, pendingClarification: value.pending,
                           contextDescription: description, turnId: envelope.turnId, requestId: envelope.requestId)
    }
    func commit(_ envelope: Envelope<Transcript>, input: RouterInput, decision: RouteDecision?) -> ContextStamp? {
        guard var value = states[envelope.conversationId], value.id == input.stamp.contextId,
              value.version == input.stamp.version else { return nil }
        guard !value.turns.contains(where: { $0.turnId == envelope.turnId }) else { return nil }
        value.turns.append(ContextTurn(role: "user", content: envelope.payload.text, turnId: envelope.turnId,
                                       requestId: envelope.requestId, outputId: nil))
        if let decision { value.history.append(RouteRecord(turnId: envelope.turnId, decision: decision)) }
        value.version += 1; value.updated = Date(); trim(&value)
        states[envelope.conversationId] = value
        return ContextStamp(contextId: value.id, version: value.version)
    }
    func clarification(_ conversation: String, requestId: UUID, slots: [String], stamp: ContextStamp) {
        guard var value = states[conversation], value.id == stamp.contextId else { return }
        value.pending = PendingClarification(requestId: requestId, slots: slots, expiresAt: Date().addingTimeInterval(config.contextExpiry))
        states[conversation] = value
    }
    func clearPending(_ conversation: String) {
        guard var value = states[conversation] else { return }; value.pending = nil; states[conversation] = value
    }
    func delivered(_ event: OutputEvent, conversation: String, target: Target?, stamp: ContextStamp?, reminder: Bool) {
        var value = state(conversation)
        if reminder, let target {
            if value.target.map({ !$0.isSameEntity(as: target) }) ?? true { value = State(); value.target = target }
        } else {
            guard let stamp, stamp.contextId == value.id else { return }
            if let target { guard value.target?.isSameEntity(as: target) == true else { return } }
            else { guard value.target == nil else { return } }
        }
        guard !value.turns.contains(where: { $0.outputId == event.outputId }) else { return }
        value.turns.append(ContextTurn(role: "assistant", content: event.text, turnId: UUID(),
                                       requestId: event.requestId, outputId: event.outputId))
        value.version += 1; value.updated = Date(); trim(&value); states[conversation] = value
    }
    private func trim(_ value: inout State) {
        while value.turns.count > config.maxContextTurns || value.turns.reduce(0, { $0 + $1.content.utf8.count }) > config.contextBytes {
            value.turns.removeFirst()
        }
        if value.history.count > config.maxContextTurns { value.history.removeFirst(value.history.count - config.maxContextTurns) }
    }
    func turns(_ conversation: String) -> [ContextTurn] { state(conversation).turns }
    func reset(_ conversation: String) { states.removeValue(forKey: conversation) }
}

public enum DecisionComposer {
    public static func compose(_ heads: HeadOutputs, confidence: Double, input: RouterInput,
                               resolution: TargetResolution, threshold: Double, sttThreshold: Double) -> RouteDecision {
        let intent = heads.intention.value
        let target: Target? = { switch resolution { case .unique(let target): return target
            case .omitted: return input.activeTarget; case .ambiguous: return nil } }()
        var missing: [String] = []
        if [.query, .complete, .postpone, .updatePriority].contains(intent), target == nil { missing.append("target") }
        if intent == .register, heads.name.value?.isEmpty != false { missing.append("name") }
        if intent == .postpone, heads.dueAt.value == nil { missing.append("dueAt") }
        if intent == .updatePriority, heads.priority.value == nil { missing.append("priority") }
        if intent == .mcp, heads.task.value?.isEmpty != false { missing.append("task") }
        if case .ambiguous = resolution { missing.append("uniqueTarget") }
        let requiredScores = [heads.intention.confidence] + (target == nil ? [] : [heads.target.confidence]) +
            (intent == .postpone ? [heads.dueAt.confidence] : []) +
            (intent == .register ? [heads.name.confidence] : []) +
            (intent == .updatePriority ? [heads.priority.confidence] : []) +
            (intent == .mcp ? [heads.task.confidence] : [])
        let reliable = ([confidence] + requiredScores).allSatisfy { $0.isFinite && $0 >= threshold && $0 <= 1 }
        let sttReliable = input.transcript.sttConfidence.isFinite && input.transcript.sttConfidence >= sttThreshold && input.transcript.sttConfidence <= 1
        let route: Route
        if !reliable || !sttReliable || !missing.isEmpty || intent == .unclear { route = .harness }
        else if intent == .chat { route = .output }
        else if [.acknowledgeReminder, .snoozeReminder].contains(intent) { route = .reminder }
        else if intent == .mcp { route = .harness }
        else { route = .eureka }
        return RouteDecision(headOutputs: heads, overallConfidence: confidence, missingRequiredSlots: missing,
                             route: route, sourceTurnIds: input.turns.map(\.turnId) + [input.turnId], resolvedTarget: target)
    }
    public static func mask(_ heads: HeadOutputs) -> HeadOutputs {
        var result = heads
        let intent = heads.intention.value
        if ![.postpone, .register].contains(intent) { result.dueAt = Head(nil, confidence: 1) }
        if ![.updatePriority, .register].contains(intent) { result.priority = Head(nil, confidence: 1) }
        if ![.mcp, .register].contains(intent) { result.task = Head(nil, confidence: 1) }
        if intent == .chat { result.target = Head(nil); result.name = Head(nil); result.note = Head(nil) }
        return result
    }
}
