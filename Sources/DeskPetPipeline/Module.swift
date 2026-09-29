import Foundation

public struct ModuleSnapshot: Sendable {
    public var submitting: Int
    public var processingCompletions: Int
    public var inFlight: Int
    public var completedQueue: Int
}

private struct ModuleCompletion<Input: Sendable, Output: Sendable>: Sendable {
    let id: UUID
    let input: Input
    let result: Result<Output, any Error>
}

// A module owns its resource and exactly one submission loop and one completion loop.
// Per-request tasks represent outstanding async calls, not replicated module workers.
protocol ModuleControl: Sendable {
    func start() async
    func stop() async
    func snapshot() async -> ModuleSnapshot
}
actor PipelineModule<Input: Sendable, Resource: Sendable, Output: Sendable>: ModuleControl {
    let resource: Resource
    private let input: Mailbox<Input>?
    private let completions: Mailbox<ModuleCompletion<Input, Output>>
    private let limit: Int
    private let execute: @Sendable (Input, Resource) async throws -> Output
    private let process: @Sendable (Input, Result<Output, any Error>) async -> Void
    private var submissionLoop: Task<Void, Never>?
    private var completionLoop: Task<Void, Never>?
    private var operations: [UUID: Task<Void, Never>] = [:]
    private var slotWaiter: CheckedContinuation<Void, Never>?
    private var drainWaiter: CheckedContinuation<Void, Never>?
    private var stopping = false
    private var stopped = false
    private var submitting = 0
    private var processing = 0

    init(resource: Resource, input: Mailbox<Input>?, limit: Int,
         execute: @escaping @Sendable (Input, Resource) async throws -> Output,
         process: @escaping @Sendable (Input, Result<Output, any Error>) async -> Void) {
        self.resource = resource; self.input = input; self.limit = limit
        completions = Mailbox(capacity: limit)
        self.execute = execute; self.process = process
    }

    func start() {
        guard completionLoop == nil, !stopping, !stopped else { return }
        completionLoop = Task { [weak self, completions] in
            while let completion = await completions.receive() {
                guard let self else { return }
                await self.processCompletion(completion)
            }
        }
        if let input {
            submissionLoop = Task { [weak self, input] in
                while !Task.isCancelled {
                    guard let self, await self.waitForSlot() else { return }
                    guard let value = await input.receive() else { return }
                    do { try await self.submit(value) } catch {
                        await self.process(value, .failure(error))
                    }
                }
            }
        }
    }

    // APIs use this entry directly, without a pending input queue.
    func submit(_ value: Input) throws {
        guard !stopping, !stopped else { throw CancellationError() }
        guard operations.count < limit else { throw PipelineError("MODULE_BUSY", stage: "module") }
        submitting = 1
        let id = UUID()
        let task = Task { [resource, execute, completions] in
            let result: Result<Output, any Error>
            do { result = .success(try await execute(value, resource)) }
            catch { result = .failure(error) }
            // Every admitted operation reserves one completion slot until processed.
            try? await completions.offer(ModuleCompletion(id: id, input: value, result: result))
        }
        operations[id] = task; submitting = 0
    }

    private func waitForSlot() async -> Bool {
        if operations.count >= limit, !stopping, !stopped {
            await withCheckedContinuation { slotWaiter = $0 }
        }
        return !stopping && !stopped && !Task.isCancelled
    }
    private func processCompletion(_ completion: ModuleCompletion<Input, Output>) async {
        guard !stopped else { return }
        processing = 1
        await process(completion.input, completion.result)
        processing = 0; operations.removeValue(forKey: completion.id)
        slotWaiter?.resume(); slotWaiter = nil
        if operations.isEmpty { drainWaiter?.resume(); drainWaiter = nil }
    }
    func snapshot() async -> ModuleSnapshot {
        ModuleSnapshot(submitting: submitting, processingCompletions: processing,
                       inFlight: operations.count, completedQueue: await completions.count)
    }
    func stop() async {
        guard !stopping, !stopped else { return }; stopping = true
        slotWaiter?.resume(); slotWaiter = nil
        if let input { await input.close() }
        submissionLoop?.cancel()
        let pending = Array(operations.values)
        for operation in pending { operation.cancel() }
        if let submissionLoop { await submissionLoop.value }
        for operation in pending { await operation.value }
        // API cancellation can produce a confirmed/unknown result. Process it before
        // discarding queues so stopping never loses external execution status.
        if !operations.isEmpty { await withCheckedContinuation { drainWaiter = $0 } }
        await completions.close()
        if let completionLoop { await completionLoop.value }
        stopped = true; operations.removeAll(); submitting = 0; processing = 0
    }
}
