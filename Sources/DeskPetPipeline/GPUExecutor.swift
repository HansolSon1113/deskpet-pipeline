import Foundation

// The runtime provides a nonblocking completion probe. It owns stream/context details.
public struct GPUCompletion<Value: Sendable>: Sendable {
    public let poll: @Sendable () -> Result<Value, any Error>?
    public init(poll: @escaping @Sendable () -> Result<Value, any Error>?) { self.poll = poll }
}
public struct GPUOperation<Value: Sendable>: Sendable {
    public let submit: @Sendable () throws -> GPUCompletion<Value>
    public init(submit: @escaping @Sendable () throws -> GPUCompletion<Value>) { self.submit = submit }
}
public struct GPUExecutorSnapshot: Sendable {
    public var outstanding: Int
    public var submitted: Int
    public var completed: Int
    public var threadCount: Int
}
private final class GPUWork<Value: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Value, any Error>?
    private var result: Result<Value, any Error>?
    private var cancelled = false
    private let operation: GPUOperation<Value>
    // Only the dedicated GPU thread touches the completion handle.
    private var completion: GPUCompletion<Value>?
    init(operation: GPUOperation<Value>) { self.operation = operation }
    func install(_ continuation: CheckedContinuation<Value, any Error>) {
        let result = lock.withLock { () -> Result<Value, any Error>? in
            if let result = self.result { return result }
            self.continuation = continuation; return nil
        }
        if let result { continuation.resume(with: result) }
    }
    func cancel() { lock.withLock { cancelled = true }; resolve(.failure(CancellationError())) }
    func fail(_ error: PipelineError) { resolve(.failure(error)) }
    func submit() -> Bool {
        if lock.withLock({ cancelled }) { return false }
        do { completion = try operation.submit(); return true }
        catch { resolve(.failure(error)); return false }
    }
    func poll() -> Bool {
        guard let result = completion?.poll() else { return false }
        resolve(result); completion = nil; return true
    }
    private func resolve(_ result: Result<Value, any Error>) {
        let continuation = lock.withLock { () -> CheckedContinuation<Value, any Error>? in
            guard self.result == nil else { return nil }
            self.result = result; let value = self.continuation; self.continuation = nil; return value
        }
        continuation?.resume(with: result)
    }
}
private struct AnyGPUWork: Sendable {
    let submit: @Sendable () -> Bool
    let poll: @Sendable () -> Bool
    let cancel: @Sendable () -> Void
}
private final class GPUExecutorState: @unchecked Sendable {
    let condition = NSCondition()
    var queue: [AnyGPUWork] = []
    var outstanding = 0
    var submitted = 0
    var completed = 0
    var stopping = false
    var stopped = false
    var started = false
    let capacity: Int
    init(capacity: Int) { self.capacity = capacity }
    func run() {
        var active: [AnyGPUWork] = []
        condition.lock(); started = true; condition.broadcast(); condition.unlock()
        while true {
            condition.lock()
            while queue.isEmpty && active.isEmpty && !stopping { condition.wait() }
            let pending = queue; queue.removeAll(keepingCapacity: true)
            let stopping = self.stopping
            condition.unlock()
            var finished = 0
            for work in pending {
                if stopping { work.cancel() }
                if work.submit() { active.append(work) } else { finished += 1 }
            }
            if stopping { for work in active { work.cancel() } }
            active.removeAll { work in
                if work.poll() { finished += 1; return true }; return false
            }
            condition.lock()
            submitted += pending.count; completed += finished; outstanding -= finished
            if self.stopping && outstanding == 0 {
                stopped = true; condition.broadcast(); condition.unlock(); return
            }
            // A runtime event wakeup can replace this mock-compatible completion polling.
            if queue.isEmpty && !active.isEmpty { _ = condition.wait(until: Date().addingTimeInterval(0.001)) }
            condition.unlock()
        }
    }
}

// One persistent OS thread submits operations and returns completions. Models belong
// to callers; their captured resources stay alive through physical completion.
public final class GPUExecutor: Sendable {
    private let state: GPUExecutorState
    public init(capacity: Int = 64) throws {
        guard capacity > 0 else { throw PipelineError("INVALID_CONFIGURATION", stage: "gpu") }
        state = GPUExecutorState(capacity: capacity)
        let thread = Thread { [state] in state.run() }
        thread.name = "DeskPet.GPU"; thread.start()
    }
    public func run<Value: Sendable>(_ operation: GPUOperation<Value>) async throws -> Value {
        try Task.checkCancellation()
        let work = GPUWork(operation: operation)
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                work.install(continuation)
                state.condition.lock()
                guard !state.stopping, state.outstanding < state.capacity else {
                    let error = PipelineError(state.stopping ? "GPU_STOPPED" : "GPU_BUSY", stage: "gpu")
                    state.condition.unlock(); work.fail(error); return
                }
                state.outstanding += 1
                state.queue.append(AnyGPUWork(submit: { work.submit() }, poll: { work.poll() }, cancel: { work.cancel() }))
                state.condition.signal(); state.condition.unlock()
            }
        } onCancel: { work.cancel() }
    }
    public func snapshot() -> GPUExecutorSnapshot {
        state.condition.lock(); defer { state.condition.unlock() }
        return GPUExecutorSnapshot(outstanding: state.outstanding, submitted: state.submitted,
                                   completed: state.completed, threadCount: state.started ? 1 : 0)
    }
    public func shutdown() async {
        requestStop()
        while !isStopped() { try? await Task.sleep(for: .milliseconds(2)) }
    }
    private func requestStop() {
        state.condition.lock(); state.stopping = true; state.condition.broadcast(); state.condition.unlock()
    }
    private func isStopped() -> Bool {
        state.condition.lock(); defer { state.condition.unlock() }; return state.stopped
    }
    deinit { requestStop() }
}

// Synthetic runtime fixture: completion timing is independent of the submitter thread.
// A real backend replaces this with its event/result handle, without changing modules.
public struct MockGPUCall<Value: Sendable>: Sendable {
    public var delay: Duration
    public var result: Result<Value, PipelineError>
    public init(value: Value, delay: Duration = .zero) { self.delay = delay; result = .success(value) }
    public init(error: PipelineError, delay: Duration = .zero) { self.delay = delay; result = .failure(error) }
    public var operation: GPUOperation<Value> {
        GPUOperation { [delay, result] in
            let ready = ContinuousClock.now.advanced(by: delay)
            return GPUCompletion { ContinuousClock.now >= ready ? result.mapError { $0 as any Error } : nil }
        }
    }
}
