import Foundation

enum RoutingHintSanitizer {
    private static let forbiddenPhrases = [
        "일상 대화로 이어갈게요",
        "일상 대화로 전환할게요",
        "일상 대화로 넘길게요",
        "일상 대화로 보낼게요",
        "일상 대화 모드로",
        "reroute_talk",
        "delegate_swarm",
        "direct_eureka"
    ]

    static func containsForbiddenRoutingHint(_ text: String) -> Bool {
        let normalized = text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalized.isEmpty else { return false }
        return forbiddenPhrases.contains { normalized.contains($0.lowercased()) }
    }

    static func sanitizeIncoming(_ text: String?) -> String? {
        guard let raw = text?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else {
            return nil
        }
        return containsForbiddenRoutingHint(raw) ? nil : raw
    }

    static func safeFallback(_ text: String, defaultText: String = "안녕하세요. 무엇을 도와드릴까요?") -> String {
        sanitizeIncoming(text) ?? defaultText
    }
}

enum ExternalInjectionOrigin: String, Sendable {
    case harnessCompletion = "harness.completion"
    case harnessNotice = "harness.notice"
}

enum ExternalInjectionRejectionReason: String, Sendable {
    case notRunning = "NOT_RUNNING"
    case malformedPayload = "MALFORMED_PAYLOAD"
    case unregisteredNotice = "UNREGISTERED_NOTICE"
    case staleTurn = "STALE_TURN"
    case duplicateInjection = "DUPLICATE_INJECTION"
    case payloadTooLarge = "RESULT_TOO_LARGE"
    case internalRoutingLeak = "INTERNAL_ROUTING_LEAK"
}

enum ExternalInjectionDirective: Sendable {
    case enqueueTalk(messages: [ChatMessage], fallbackText: String)
    case enqueueAction(requiredText: String, isQuestion: Bool)
}

struct ValidatedHarnessNotice: Sendable {
    let requestId: UUID
    let status: ActionStatus
    let facts: [String: String]
    let directive: ExternalInjectionDirective
}

enum ExternalNoticePreparation: Sendable {
    case admit(ValidatedHarnessNotice)
    case deferEarly(requestId: UUID, data: Data)
    case reject(ExternalInjectionRejectionReason, requestId: UUID?)
}

enum ExternalCompletionPreparation: Sendable {
    case admit(ExternalInjectionDirective)
    case reject(ExternalInjectionRejectionReason)
}

struct ExternalInjectionContext: Sendable {
    let isRunning: Bool
    let isCurrentTurn: Bool
    let hasNoticeJob: Bool
    let hasActiveRequest: Bool
    let turns: [ContextTurn]
    let userText: String
}

/// Gatekeeper + Adapter module for validating and normalizing external/Harness
/// injections before any item is admitted into `outputQueue`.
actor ExternalInjectionModule {
    private let config: PipelineConfiguration
    private var seenKeys: Set<String> = []
    private var seenOrder: [String] = []

    init(configuration: PipelineConfiguration) {
        self.config = configuration
    }

    func reset() {
        seenKeys.removeAll()
        seenOrder.removeAll()
    }

    func prepareNotice(
        _ data: Data,
        lookup: @Sendable (UUID) async -> ExternalInjectionContext?
    ) async -> ExternalNoticePreparation {
        guard let frame = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let notice = frame["notice"] as? [String: Any],
              let rawId = notice["requestId"] as? String,
              let requestId = UUID(uuidString: rawId),
              let action = notice["pipelineAction"] as? [String: Any],
              let rawStatus = action["status"] as? String,
              let status = ActionStatus(rawValue: rawStatus) else {
            return .reject(.malformedPayload, requestId: nil)
        }

        let facts = action["facts"] as? [String: String] ?? [:]
        let rawQuestion = notice["pipelineQuestion"] as? String
        let rawResultText = frame["resultText"] as? String
        guard validateBounds(facts: facts, question: rawQuestion, resultText: rawResultText) else {
            return .reject(.payloadTooLarge, requestId: requestId)
        }

        guard let context = await lookup(requestId) else {
            return .reject(.unregisteredNotice, requestId: requestId)
        }
        guard context.isRunning else {
            return .reject(.notRunning, requestId: requestId)
        }
        guard context.hasNoticeJob else {
            if context.hasActiveRequest {
                return .deferEarly(requestId: requestId, data: data)
            }
            return .reject(.unregisteredNotice, requestId: requestId)
        }
        guard context.isCurrentTurn else {
            return .reject(.staleTurn, requestId: requestId)
        }

        let noticeId = (notice["noticeId"] as? String).flatMap { $0.isEmpty ? nil : $0 }
        let dedupeKey = noticeId.map { "notice:\($0)" }
            ?? "notice:\(requestId.uuidString):\(status.rawValue):\(rawQuestion ?? ""):\(rawResultText ?? "")"
        guard recordIfUnique(dedupeKey) else {
            return .reject(.duplicateInjection, requestId: requestId)
        }

        let directive = buildDirective(
            status: status,
            facts: facts,
            question: rawQuestion,
            resultText: rawResultText,
            turns: context.turns,
            userText: context.userText,
            factualFallback: Self.defaultActionText(status: status, facts: facts, resultText: rawResultText)
        )
        return .admit(ValidatedHarnessNotice(
            requestId: requestId,
            status: status,
            facts: facts,
            directive: directive
        ))
    }

    func prepareHarnessCompletion(
        requestId: UUID,
        operationId: UUID,
        status: ActionStatus,
        facts: [String: String],
        question: String?,
        resultText: String?,
        context: ExternalInjectionContext,
        factualFallback: String
    ) -> ExternalCompletionPreparation {
        guard context.isRunning else { return .reject(.notRunning) }
        guard context.isCurrentTurn else { return .reject(.staleTurn) }
        guard validateBounds(facts: facts, question: question, resultText: resultText) else {
            return .reject(.payloadTooLarge)
        }

        let dedupeKey = "completion:\(requestId.uuidString):\(operationId.uuidString):\(status.rawValue):\(question ?? ""):\(resultText ?? "")"
        guard recordIfUnique(dedupeKey) else {
            return .reject(.duplicateInjection)
        }

        let directive = buildDirective(
            status: status,
            facts: facts,
            question: question,
            resultText: resultText,
            turns: context.turns,
            userText: context.userText,
            factualFallback: factualFallback
        )
        return .admit(directive)
    }

    private func buildDirective(
        status: ActionStatus,
        facts: [String: String],
        question: String?,
        resultText: String?,
        turns: [ContextTurn],
        userText: String,
        factualFallback: String
    ) -> ExternalInjectionDirective {
        let interruptedText = Self.interruptedSummary(from: facts)
        if status == .needsClarification {
            let safeQuestion = RoutingHintSanitizer.sanitizeIncoming(question)
                ?? RoutingHintSanitizer.sanitizeIncoming(resultText)
                ?? "필요한 내용을 다시 말씀해 주세요."
            return .enqueueAction(requiredText: safeQuestion, isQuestion: true)
        }

        if let interruptedText, status != .succeeded || facts["interrupted"] != nil {
            let safeSummary = RoutingHintSanitizer.sanitizeIncoming(resultText) ?? interruptedText
            return .enqueueAction(requiredText: safeSummary, isQuestion: false)
        }

        if facts["reroute"] == "talk", status == .succeeded {
            let history = turns.map { ChatMessage(role: $0.role, content: $0.content) }
            let trimmedUser = userText.trimmingCharacters(in: .whitespacesAndNewlines)
            let messages = trimmedUser.isEmpty ? history : history + [ChatMessage(role: "user", content: userText)]
            return .enqueueTalk(messages: messages, fallbackText: "안녕하세요. 무엇을 도와드릴까요?")
        }

        let sanitized = RoutingHintSanitizer.sanitizeIncoming(resultText)
            ?? RoutingHintSanitizer.safeFallback(factualFallback, defaultText: "처리 결과를 확인했어요.")
        return .enqueueAction(requiredText: sanitized, isQuestion: false)
    }

    private func validateBounds(facts: [String: String], question: String?, resultText: String?) -> Bool {
        let half = config.maxOutputBytes / 2
        let factsBytes = facts.reduce(0) { $0 + $1.key.utf8.count + $1.value.utf8.count }
        return facts.count <= 32
            && factsBytes <= half
            && (question?.utf8.count ?? 0) <= half
            && (resultText?.utf8.count ?? 0) <= half
    }

    private func recordIfUnique(_ key: String) -> Bool {
        guard !seenKeys.contains(key) else { return false }
        let limit = max(16, config.journalCapacity)
        if seenOrder.count >= limit {
            let evicted = seenOrder.removeFirst()
            seenKeys.remove(evicted)
        }
        seenKeys.insert(key)
        seenOrder.append(key)
        return true
    }

    private static func interruptedSummary(from facts: [String: String]) -> String? {
        guard let raw = facts["interrupted"], let data = raw.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        if let summary = json["summaryText"] as? String,
           let clean = RoutingHintSanitizer.sanitizeIncoming(summary) {
            return clean
        }
        guard let tasks = json["tasks"] as? [[String: Any]], !tasks.isEmpty else {
            return nil
        }
        let goals = tasks.compactMap { task -> String? in
            guard let goal = task["goal"] as? String else { return nil }
            let trimmed = goal.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        }
        guard !goals.isEmpty else { return nil }
        return "응답이 없어 작업을 중단했습니다. 중단된 작업: \(goals.joined(separator: ", "))"
    }

    private static func defaultActionText(status: ActionStatus, facts: [String: String], resultText: String?) -> String {
        if let clean = RoutingHintSanitizer.sanitizeIncoming(resultText) { return clean }
        if let interrupted = interruptedSummary(from: facts) { return interrupted }
        switch status {
        case .succeeded: return "요청 처리 결과를 확인했어요."
        case .failed: return "요청을 처리하지 못했어요."
        case .unknown: return "요청이 반영됐는지 아직 확인하지 못했어요. 다시 실행하지 않고 상태 확인이 필요해요."
        case .cancelled: return "요청 취소가 확인됐어요."
        case .needsClarification: return "필요한 내용을 다시 말씀해 주세요."
        }
    }
}
