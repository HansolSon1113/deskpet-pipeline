import Foundation

/// Connects the existing Harness port to LLM_test's deskpet.harness.v1 server.
/// Wire mapping follows docs/pipeline-bridge.md; no retry or approval inference.
public struct ServerHarness: HarnessAdapter {
    private let url: URL
    private let token: String
    public init(url: URL, token: String) throws {
        guard ["ws", "wss"].contains(url.scheme), !token.isEmpty else {
            throw PipelineError("HARNESS_CONFIGURATION", stage: "harness")
        }
        self.url = url; self.token = token
    }

    public func run(_ request: HarnessRequest) async throws -> HarnessResult {
        let input = request.context
        var context: [String: Any] = ["requestId": input.requestId.uuidString, "turnId": input.turnId.uuidString,
            "transcript": try object(input.transcript),
            "stamp": ["contextId": input.stamp.contextId.uuidString, "version": input.stamp.version],
            "turns": input.turns.map { ["role": $0.role, "content": $0.content, "turnId": $0.turnId.uuidString] }]
        if let target = input.activeTarget { context["activeTarget"] = try object(target) }
        let payload: [String: Any] = ["action": try object(request.action), "transcript": try object(request.transcript),
            "context": context, "conversationId": input.stamp.contextId.uuidString,
            "wikiSearchAllowed": request.wikiSearchAllowed, "mutationApproved": request.mutationApproved]
        let data = try await call("harness.run", action: request.action, payload: payload, replyKind: "harness.run.result")
        let result = try ContractCodec.decoder().decode(RunReply.self, from: data)
        return HarnessResult(action: try result.action.local(request.action), question: result.question, resultText: result.resultText)
    }

    public func reconcile(_ request: ActionRequest) async throws -> ActionResult {
        let data = try await call("harness.reconcile", action: request, payload: object(request), replyKind: "action.result")
        return try ContractCodec.decoder().decode(ActionReply.self, from: data).local(request)
    }

    func object<T: Encodable>(_ value: T) throws -> Any {
        try JSONSerialization.jsonObject(with: ContractCodec.encoder().encode(value))
    }
    private func call(_ kind: String, action: ActionRequest, payload: Any, replyKind: String) async throws -> Data {
        try await call(kind, requestId: action.requestId, deadline: action.deadlineAt, payload: payload, replyKind: replyKind)
    }
    func call(_ kind: String, requestId: UUID, deadline: Date, payload: Any, replyKind: String) async throws -> Data {
        let messageId = UUID().uuidString
        let stage = kind.hasPrefix("eureka.") ? "eureka" : "harness"
        let envelope: [String: Any] = ["schemaVersion": "1.0", "messageId": messageId, "requestId": requestId.uuidString,
            "createdAt": DateCodec.rfc3339(Date()), "deadlineAt": DateCodec.rfc3339(deadline), "kind": kind, "payload": payload]
        let frame = try JSONSerialization.data(withJSONObject: ["type": "request", "envelope": envelope])
        let hello = try JSONSerialization.data(withJSONObject: ["type": "hello", "token": token, "role": "pipeline", "client": "deskpet-pipeline"])
        let session = URLSession(configuration: .ephemeral)
        let socket = session.webSocketTask(with: url); socket.maximumMessageSize = 256_000
        defer { socket.cancel(with: .normalClosure, reason: nil); session.invalidateAndCancel() }
        socket.resume()
        return try await withTimeout(min(30, deadline.timeIntervalSinceNow), stage: stage) {
            try await withTaskCancellationHandler {
                try await socket.send(.string(String(decoding: hello, as: UTF8.self)))
                let welcome = try JSONSerialization.jsonObject(with: await Self.receive(socket)) as? [String: Any]
                guard welcome?["type"] as? String == "welcome", welcome?["protocol"] as? String == "deskpet.harness.v1",
                      welcome?["schemaVersion"] as? String == "1.0", welcome?["role"] as? String == "pipeline" else {
                    throw PipelineError("HARNESS_HANDSHAKE", stage: "harness")
                }
                try await socket.send(.string(String(decoding: frame, as: UTF8.self)))
                let reply = try JSONSerialization.jsonObject(with: await Self.receive(socket)) as? [String: Any]
                guard reply?["type"] as? String == "reply", reply?["causationId"] as? String == messageId,
                      reply?["requestId"] as? String == requestId.uuidString else {
                    throw PipelineError("CORRELATION_MISMATCH", stage: "harness")
                }
                guard reply?["ok"] as? Bool == true else {
                    let code = (reply?["error"] as? [String: Any])?["code"] as? String ?? "HARNESS_SERVER_ERROR"
                    throw PipelineError(code, stage: "harness")
                }
                guard reply?["kind"] as? String == replyKind, let payload = reply?["payload"] else {
                    throw PipelineError("HARNESS_REPLY_INVALID", stage: "harness")
                }
                return try JSONSerialization.data(withJSONObject: payload)
            } onCancel: { socket.cancel(with: .goingAway, reason: nil) }
        }
    }
    private static func receive(_ socket: URLSessionWebSocketTask) async throws -> Data {
        switch try await socket.receive() {
        case .data(let data): return data
        case .string(let text): return Data(text.utf8)
        @unknown default: throw PipelineError("HARNESS_REPLY_INVALID", stage: "harness")
        }
    }
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
