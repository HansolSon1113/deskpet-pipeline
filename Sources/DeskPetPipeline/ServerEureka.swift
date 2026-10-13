import Foundation

/// Direct Eureka port, separate from Harness. Uses the server's existing EurekaBridge.
public struct ServerEureka: EurekaAdapter {
    private let connection: ServerHarness
    public init(url: URL, token: String) throws { connection = try ServerHarness(url: url, token: token) }

    public func execute(_ request: ActionRequest) async throws -> ActionResult { try await action("eureka.execute", request) }
    public func reconcile(_ request: ActionRequest) async throws -> ActionResult { try await action("eureka.reconcile", request) }
    private func action(_ kind: String, _ request: ActionRequest) async throws -> ActionResult {
        var payload = try connection.object(request) as! [String: Any]
        var decision = payload["decision"] as! [String: Any]
        if let target = request.decision.resolvedTarget {
            decision["resolvedTarget"] = ["kind": "eureka_item", "itemId": target.entityId]
            // A targeted Swift query corresponds to the server's status read (P-12).
            if request.decision.headOutputs.intention.value == .query {
                var heads = decision["headOutputs"] as! [String: Any]
                heads["intention"] = ["value": "eureka.status", "confidence": request.decision.headOutputs.intention.confidence]
                decision["headOutputs"] = heads
            }
        }
        payload["decision"] = decision
        let data = try await connection.call(kind, requestId: request.requestId, deadline: request.deadlineAt,
                                             payload: payload, replyKind: "action.result")
        return try ContractCodec.decoder().decode(ActionReply.self, from: data).local(request)
    }

    public func reminders() async throws -> [ReminderItem] {
        let data = try await read("eureka.reminders", itemId: nil, replyKind: "eureka.reminders.result")
        let result = try JSONDecoder().decode(RemindersReply.self, from: data)
        return try result.items.map { try $0.local() }
    }
    public func reminder(itemId: String) async throws -> ReminderItem? {
        let data = try await read("eureka.reminder", itemId: itemId, replyKind: "eureka.reminder.result")
        return try JSONDecoder().decode(ReminderReply.self, from: data).item?.local()
    }
    private func read(_ kind: String, itemId: String?, replyKind: String) async throws -> Data {
        let requestId = UUID()
        var payload = ["requestId": requestId.uuidString]
        if let itemId { payload["itemId"] = itemId }
        return try await connection.call(kind, requestId: requestId, deadline: Date().addingTimeInterval(8),
                                         payload: payload, replyKind: replyKind)
    }
}

private struct RemindersReply: Decodable { var items: [WireReminder] }
private struct ReminderReply: Decodable { var item: WireReminder? }
private struct WireReminder: Decodable {
    var itemId: String; var name: String; var description: String; var dueAt: String; var timezone: String; var state: String
    func local() throws -> ReminderItem {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        guard let due = formatter.date(from: dueAt) ?? ISO8601DateFormatter().date(from: dueAt), !description.isEmpty else {
            throw PipelineError("EUREKA_REMINDER_INVALID", stage: "eureka")
        }
        return ReminderItem(target: Target(entityId: itemId, name: name), description: description, dueAt: due,
                            timeZone: timezone, eligible: state != "done" && state != "hold")
    }
}
