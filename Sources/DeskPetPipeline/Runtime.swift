import Foundation

public struct PipelineConfiguration: Sendable {
    public var queueCapacity = 16
    public var audioQueueCapacity = 2
    public var streamingFrameQueueCapacity = 64
    public var pcmQueueCapacity = 4
    public var maxInFlight = 64
    public var maxConcurrentEurekaRequests = 8
    public var maxConcurrentHarnessRequests = 8
    public var journalCapacity = 256
    public var eventCapacity = 256
    public var contextBytes = 8_192
    public var maxContextTurns = 16
    public var maxConversations = 8
    public var maxTranscriptBytes = 4_096
    public var maxAudioBytes = 2_000_000
    public var maxChunkBytes = 64_000
    public var maxOutputBytes = 8_192
    public var maxReminderItems = 100
    public var sttConfidenceThreshold = 0.7
    public var routeConfidenceThreshold = 0.8
    public var sttTimeout: TimeInterval = 5
    public var routerTimeout: TimeInterval = 2
    public var eurekaTimeout: TimeInterval = 8
    public var harnessTimeout: TimeInterval = 30
    public var outputTimeout: TimeInterval = 5
    public var ttsFirstChunkTimeout: TimeInterval = 3
    public var playbackTimeout: TimeInterval = 30
    public var requestTimeout: TimeInterval = 60
    public var remoteTimeout: TimeInterval = 30
    public var contextExpiry: TimeInterval = 300
    public var reminderPollInterval: TimeInterval = 300
    public var reminderQuietInterval: TimeInterval = 5
    public var questionWait: TimeInterval = 15
    public var textDisplayDuration: TimeInterval = 15
    public var progressDelay: TimeInterval = 10
    public init() {}
    func validate() throws {
        let counts = [queueCapacity, audioQueueCapacity, streamingFrameQueueCapacity, pcmQueueCapacity, maxInFlight, maxConcurrentEurekaRequests, maxConcurrentHarnessRequests, journalCapacity, eventCapacity, contextBytes,
                      maxContextTurns, maxConversations, maxTranscriptBytes, maxAudioBytes,
                      maxChunkBytes, maxOutputBytes, maxReminderItems]
        let times = [sttTimeout, routerTimeout, eurekaTimeout, harnessTimeout, outputTimeout,
                     ttsFirstChunkTimeout, playbackTimeout, requestTimeout, remoteTimeout,
                     contextExpiry, reminderPollInterval, questionWait, textDisplayDuration, progressDelay]
        guard counts.allSatisfy({ $0 > 0 }), times.allSatisfy({ $0.isFinite && $0 > 0 }),
              reminderQuietInterval.isFinite, reminderQuietInterval >= 0,
              (0...1).contains(sttConfidenceThreshold), (0...1).contains(routeConfidenceThreshold) else {
            throw PipelineError("INVALID_CONFIGURATION", stage: "startup")
        }
    }
}

// Input admission uses nonblocking offer/reserve. A single bounded producer can use
// send for downstream backpressure (TTS -> playback). Cancelled waiters are reclaimed.
actor Mailbox<Element: Sendable> {
    private var storage: [Element] = []
    private let capacity: Int
    private var closed = false
    private var reservations: Set<UUID> = []
    private var receivers: [(UUID, CheckedContinuation<Element?, Never>)] = []
    private var sender: (id: UUID, value: Element, continuation: CheckedContinuation<Void, any Error>)?
    init(capacity: Int) { self.capacity = capacity }
    func offer(_ value: Element) throws {
        guard !closed else { throw PipelineError("CANCELLED", stage: "queue") }
        guard storage.count + reservations.count < capacity else { throw PipelineError("QUEUE_FULL", stage: "queue") }
        deliver(value)
    }
    func send(_ value: Element) async throws {
        try Task.checkCancellation()
        guard !closed else { throw CancellationError() }
        if storage.count + reservations.count < capacity { deliver(value); return }
        guard sender == nil else { throw PipelineError("MULTIPLE_PENDING_PRODUCERS", stage: "queue") }
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
                if Task.isCancelled { continuation.resume(throwing: CancellationError()) }
                else { sender = (id, value, continuation) }
            }
        } onCancel: { Task { await self.cancelSender(id) } }
    }
    private func cancelSender(_ id: UUID) {
        guard let pending = sender, pending.id == id else { return }
        sender = nil; pending.continuation.resume(throwing: CancellationError())
    }
    private func admitSender() {
        guard !closed, let pending = sender, storage.count + reservations.count < capacity else { return }
        sender = nil; deliver(pending.value); pending.continuation.resume()
    }
    private func deliver(_ value: Element) {
        if receivers.isEmpty { storage.append(value) }
        else { receivers.removeFirst().1.resume(returning: value) }
    }
    func reserve() throws -> UUID {
        guard !closed else { throw PipelineError("CANCELLED", stage: "queue") }
        guard storage.count + reservations.count < capacity else { throw PipelineError("QUEUE_FULL", stage: "queue") }
        let id = UUID(); reservations.insert(id); return id
    }
    func commit(_ value: Element, reservation: UUID) throws {
        guard !closed, reservations.remove(reservation) != nil else { throw PipelineError("CANCELLED", stage: "queue") }
        deliver(value)
    }
    func release(_ reservation: UUID) { reservations.remove(reservation); admitSender() }
    func pop() -> Element? {
        guard !storage.isEmpty else { return nil }
        let value = storage.removeFirst(); admitSender(); return value
    }
    func receive() async -> Element? {
        guard !Task.isCancelled else { return nil }
        if let value = pop() { return value }
        guard !closed, receivers.count < 64 else { return nil }
        let id = UUID()
        return await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                if Task.isCancelled { continuation.resume(returning: nil) }
                else { receivers.append((id, continuation)) }
            }
        } onCancel: { Task { await self.cancelReceiver(id) } }
    }
    private func cancelReceiver(_ id: UUID) {
        guard let index = receivers.firstIndex(where: { $0.0 == id }) else { return }
        receivers.remove(at: index).1.resume(returning: nil)
    }
    func remove(where predicate: @Sendable (Element) -> Bool) {
        storage.removeAll(where: predicate)
        if let pending = sender, predicate(pending.value) { cancelSender(pending.id) }
        admitSender()
    }
    func close() {
        closed = true; storage.removeAll(); reservations.removeAll()
        if let pending = sender { cancelSender(pending.id) }
        let pending = receivers; receivers.removeAll()
        for receiver in pending { receiver.1.resume(returning: nil) }
    }
    var count: Int { storage.count }
}

func withTimeout<T: Sendable>(_ seconds: TimeInterval, stage: String,
                              operation: @escaping @Sendable () async throws -> T) async throws -> T {
    try Task.checkCancellation()
    guard seconds > 0 else { throw PipelineError("DEADLINE_EXCEEDED", stage: stage) }
    return try await withThrowingTaskGroup(of: T.self) { group in
        group.addTask { try await operation() }
        group.addTask {
            try await Task.sleep(for: .seconds(seconds))
            throw PipelineError("DEADLINE_EXCEEDED", stage: stage)
        }
        defer { group.cancelAll() }
        let value = try await group.next()!
        try Task.checkCancellation()
        return value
    }
}

public struct StageMeasurement: Sendable {
    public var count: Int
    public var p50Milliseconds: Double
    public var p95Milliseconds: Double
    public var maxMilliseconds: Double
}
actor Metrics {
    struct Samples { var count = 0; var values: [Double] = []; var maximum = 0.0 }
    private var samples: [String: Samples] = [:]
    func record(_ stage: String, since start: ContinuousClock.Instant) {
        let duration = start.duration(to: .now).components
        let ms = Double(duration.seconds) * 1_000 + Double(duration.attoseconds) / 1e15
        var sample = samples[stage, default: Samples()]; sample.count += 1
        sample.maximum = max(sample.maximum, ms)
        if sample.values.count == 256 { sample.values.removeFirst() }
        sample.values.append(ms); samples[stage] = sample
    }
    func snapshot() -> [String: StageMeasurement] {
        samples.mapValues { sample in
            let sorted = sample.values.sorted()
            func percentile(_ q: Double) -> Double { sorted[max(0, Int(ceil(Double(sorted.count) * q)) - 1)] }
            return StageMeasurement(count: sample.count, p50Milliseconds: percentile(0.5),
                                    p95Milliseconds: percentile(0.95), maxMilliseconds: sample.maximum)
        }
    }
}
