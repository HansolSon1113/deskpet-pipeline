import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// One authenticated connection, one receiver. No automatic replay of external work.
actor ServerConnection {
    struct Pending {
        let requestId: String
        let kind: String
        let continuation: CheckedContinuation<Data, any Error>
    }
    let url: URL
    let token: String
    let notices: AsyncStream<Data>
    private let noticeSink: AsyncStream<Data>.Continuation
    private var session: URLSession?
    private var socket: URLSessionWebSocketTask?
    private var connecting: Task<Void, any Error>?
    private var reader: Task<Void, Never>?
    private var pending: [String: Pending] = [:]
    private var revisions: [UUID: Int] = [:]
    init(url: URL, token: String) {
        self.url = url; self.token = token
        let pair = AsyncStream<Data>.makeStream(bufferingPolicy: .bufferingOldest(64))
        notices = pair.stream; noticeSink = pair.continuation
    }
    func revision(_ id: UUID) throws -> Int {
        guard revisions[id] != nil || revisions.count < 256 else { throw PipelineError("HARNESS_EVENT_CAPACITY", stage: "harness") }
        let next = (revisions[id] ?? 0) + 1; revisions[id] = next; return next
    }
    func retire(_ id: UUID) { revisions.removeValue(forKey: id) }
    func connect() async throws {
        if let connecting { return try await connecting.value }
        if socket != nil { return }
        let task = Task { try await self.open() }
        connecting = task
        do { try await task.value; connecting = nil }
        catch { connecting = nil; throw error }
    }
    private func open() async throws {
        let session = URLSession(configuration: .ephemeral)
        let socket = session.webSocketTask(with: url); socket.maximumMessageSize = 256_000; socket.resume()
        do {
            let hello = try JSONSerialization.data(withJSONObject: ["type": "hello", "token": token, "role": "pipeline", "client": "deskpet-pipeline"])
            try await socket.send(.string(String(decoding: hello, as: UTF8.self)))
            let data = try await withTimeout(5, stage: "harness.hello") { try await Self.receive(socket) }
            let welcome = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            guard welcome?["type"] as? String == "welcome", welcome?["protocol"] as? String == "deskpet.harness.v1",
                  welcome?["role"] as? String == "pipeline", welcome?["schemaVersion"] as? String == "1.0" else {
                throw PipelineError("HARNESS_HANDSHAKE", stage: "harness")
            }
            self.session = session; self.socket = socket
            reader = Task { [weak self] in
                do { while !Task.isCancelled { let data = try await Self.receive(socket); await self?.received(data) } }
                catch { await self?.disconnected() }
            }
        } catch { socket.cancel(with: .goingAway, reason: nil); session.invalidateAndCancel(); throw error }
    }
    func request(frame: Data, messageId: String, requestId: String, replyKind: String, deadline: Date) async throws -> Data {
        try await connect()
        guard let socket else { throw PipelineError("HARNESS_DISCONNECTED", stage: "harness") }
        return try await withTimeout(min(30, deadline.timeIntervalSinceNow), stage: "harness") { [self] in
            try await withTaskCancellationHandler {
                try await self.send(frame, socket: socket, messageId: messageId, requestId: requestId, replyKind: replyKind)
            } onCancel: { Task { await self.cancel(messageId) } }
        }
    }
    private func send(_ frame: Data, socket: URLSessionWebSocketTask, messageId: String, requestId: String, replyKind: String) async throws -> Data {
        try Task.checkCancellation()
        guard pending.count < 32 else { throw PipelineError("HARNESS_BUSY", stage: "harness") }
        return try await withCheckedThrowingContinuation { continuation in
            pending[messageId] = Pending(requestId: requestId, kind: replyKind, continuation: continuation)
            Task {
                do { try await socket.send(.string(String(decoding: frame, as: UTF8.self))) }
                catch { self.cancel(messageId) }
            }
        }
    }
    private func cancel(_ id: String) {
        pending.removeValue(forKey: id)?.continuation.resume(throwing: PipelineError("HARNESS_REPLY_UNAVAILABLE", stage: "harness"))
    }
    private func received(_ data: Data) {
        do {
            guard let frame = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
            if frame["type"] as? String == "notice" {
                if let notice = frame["notice"] as? [String: Any], let action = notice["pipelineAction"] as? [String: Any],
                   let status = action["status"] as? String, ["succeeded", "failed", "cancelled"].contains(status),
                   let raw = notice["requestId"] as? String, let id = UUID(uuidString: raw) { retire(id) }
                if case .dropped = noticeSink.yield(data) { disconnected() }
                return
            }
            guard frame["type"] as? String == "reply", let id = frame["causationId"] as? String,
                  let call = pending.removeValue(forKey: id) else { return }
            do {
                guard frame["requestId"] as? String == call.requestId else { throw PipelineError("CORRELATION_MISMATCH", stage: "harness") }
                guard frame["ok"] as? Bool == true else {
                    throw PipelineError((frame["error"] as? [String: Any])?["code"] as? String ?? "HARNESS_SERVER_ERROR", stage: "harness")
                }
                guard frame["kind"] as? String == call.kind, let payload = frame["payload"] else { throw PipelineError("HARNESS_REPLY_INVALID", stage: "harness") }
                call.continuation.resume(returning: try JSONSerialization.data(withJSONObject: payload))
            } catch { call.continuation.resume(throwing: error) }
        } catch { disconnected() }
    }
    func close() { reader?.cancel(); reader = nil; disconnected(); noticeSink.finish() }
    private func disconnected() {
        socket?.cancel(with: .goingAway, reason: nil); socket = nil
        session?.invalidateAndCancel(); session = nil
        let calls = pending; pending.removeAll()
        for call in calls.values { call.continuation.resume(throwing: PipelineError("HARNESS_DISCONNECTED", stage: "harness")) }
    }
    private static func receive(_ socket: URLSessionWebSocketTask) async throws -> Data {
        switch try await socket.receive() {
        case .data(let data): return data
        case .string(let string): return Data(string.utf8)
        @unknown default: throw PipelineError("HARNESS_REPLY_INVALID", stage: "harness")
        }
    }
    deinit { reader?.cancel(); socket?.cancel(with: .goingAway, reason: nil); session?.invalidateAndCancel() }
}
