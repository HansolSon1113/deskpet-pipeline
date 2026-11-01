import Foundation

/// Connects the existing Harness port to LLM_test's deskpet.harness.v1 server.
/// Wire mapping follows docs/pipeline-bridge.md; no retry or approval inference.
public struct ServerHarness: HarnessAdapter, HarnessEventConsumer, HarnessNoticeSource {
    private let transport: ServerConnection
    private let url: URL
    private let token: String
    public init(url: URL, token: String) throws {
        guard ["ws", "wss"].contains(url.scheme), !token.isEmpty else {
            throw PipelineError("HARNESS_CONFIGURATION", stage: "harness")
        }
        self.url = url; self.token = token; self.transport = ServerConnection(url: url, token: token)
    }

    public func run(_ request: HarnessRequest) async throws -> HarnessResult {
        let input = request.context
        var context: [String: Any] = ["requestId": input.requestId.uuidString, "turnId": input.turnId.uuidString,
            "transcript": try object(input.transcript),
            "stamp": ["contextId": input.stamp.contextId.uuidString, "version": input.stamp.version],
            "turns": input.turns.map { ["role": $0.role, "content": $0.content, "turnId": $0.turnId.uuidString] }]
        if let target = input.activeTarget { context["activeTarget"] = try object(target) }
        var payload: [String: Any] = ["action": try object(request.action), "transcript": try object(request.transcript),
            "context": context, "conversationId": request.conversationId ?? input.stamp.contextId.uuidString,
            "wikiSearchAllowed": request.wikiSearchAllowed, "mutationApproved": request.mutationApproved]
        if let pending = request.pendingId { context["pendingClarification"] = ["pendingId": pending]; payload["context"] = context }
        payload["taskRefs"] = try object(request.taskRefs)
        var action = try wireAction(request.action)
        var decision = action["decision"] as! [String: Any]
        if let target = request.action.decision.resolvedTarget, target.kind == "github_pr",
           let repository = target.repository, let number = target.prNumber {
            decision["resolvedTarget"] = ["kind": "github_pr", "repository": repository, "prNumber": number]
        }
        action["decision"] = decision; payload["action"] = action
        let data = try await call("harness.run", action: request.action, payload: payload, replyKind: "harness.run.result")
        let result = try ContractCodec.decoder().decode(RunReply.self, from: data)
        if [.succeeded, .failed, .cancelled].contains(result.action.status) { await transport.retire(request.action.requestId) }
        return HarnessResult(action: try result.action.local(request.action), question: result.question, resultText: result.resultText)
    }

    public func reconcile(_ request: ActionRequest) async throws -> ActionResult {
        let data = try await call("harness.reconcile", action: request, payload: wireAction(request), replyKind: "action.result")
        return try ContractCodec.decoder().decode(ActionReply.self, from: data).local(request)
    }

    func object<T: Encodable>(_ value: T) throws -> Any {
        try JSONSerialization.jsonObject(with: ContractCodec.encoder().encode(value))
    }
    func wireAction(_ request: ActionRequest) throws -> [String: Any] {
        var action = try object(request) as! [String: Any]
        var decision = action["decision"] as! [String: Any]
        var heads = decision["headOutputs"] as! [String: Any]
        var metadata: [String: Any] = [:]
        // The existing Harness validates headOutputs as {value, confidence}.
        // Rich native metadata travels in its allowed decision extension.
        for key in ["spans", "modelConfidence", "generationFinished"] {
            if let value = heads.removeValue(forKey: key) { metadata[key] = value }
        }
        decision["headOutputs"] = heads
        if !metadata.isEmpty { decision["modelMetadata"] = metadata }
        action["decision"] = decision
        return action
    }
    private func call(_ kind: String, action: ActionRequest, payload: Any, replyKind: String) async throws -> Data {
        try await call(kind, requestId: action.requestId, deadline: action.deadlineAt, payload: payload, replyKind: replyKind)
    }
    func call(_ kind: String, requestId: UUID, deadline: Date, payload: Any, replyKind: String) async throws -> Data {
        let messageId = UUID().uuidString
        let envelope: [String: Any] = ["schemaVersion": "1.0", "messageId": messageId, "requestId": requestId.uuidString,
            "createdAt": DateCodec.rfc3339(Date()), "deadlineAt": DateCodec.rfc3339(deadline), "kind": kind, "payload": payload]
        let frame = try JSONSerialization.data(withJSONObject: ["type": "request", "envelope": envelope])
        return try await transport.request(frame: frame, messageId: messageId, requestId: requestId.uuidString,
                                            replyKind: replyKind, deadline: deadline)
    }
    public func report(_ event: HarnessPipelineEvent) async throws {
        var payload: [String: Any] = ["messageId": UUID().uuidString, "requestId": event.requestId.uuidString,
            "sourceRevision": try await transport.revision(event.requestId), "occurredAt": DateCodec.rfc3339(Date()), "kind": event.kind]
        if let value = event.pendingId { payload["pendingId"] = value }
        if let value = event.outputId { payload["outputId"] = value }
        if let value = event.deliveredText { payload["deliveredText"] = value }
        if let value = event.channel { payload["channel"] = value }
        let data = try await call("pipeline.event", requestId: event.requestId, deadline: Date().addingTimeInterval(5), payload: payload, replyKind: "pipeline.event.result")
        let result = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        guard result?["applied"] as? Bool == true else { throw PipelineError("HARNESS_EVENT_REJECTED", stage: "harness") }
        if ["question_wait_expired", "request_cancelled"].contains(event.kind) { await transport.retire(event.requestId) }
    }
    public var notices: AsyncStream<Data> { get async { transport.notices } }
    public func close() async { await transport.close() }

}

private struct RunReply: Decodable { var action: ActionReply; var question: String?; var resultText: String? }
struct ActionReply: Decodable {
    var operationId: UUID; var status: ActionStatus; var facts: [String: String]; var error: WireError?
    func local(_ request: ActionRequest) throws -> ActionResult {
        guard operationId == request.operationId else { throw PipelineError("CORRELATION_MISMATCH", stage: "harness") }
        return ActionResult(operationId: operationId, status: status, action: request.decision.headOutputs.intention.value,
            target: request.decision.resolvedTarget, facts: facts,
            error: error.map { PipelineError($0.code, stage: $0.stage ?? "harness", message: $0.message, retryable: $0.retryable) })
    }
}
struct WireError: Decodable { var code: String; var stage: String?; var message: String; var retryable: Bool }
