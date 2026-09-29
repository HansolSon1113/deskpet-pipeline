import Foundation

// Exactly one TTS producer and one playback consumer per output. Backpressure suspends
// that bounded producer instead of accumulating audio or spawning more producer tasks.
actor PCMChannel {
    private var chunks: [PCMChunk] = []
    private let capacity: Int
    private var closed = false
    private var failure: (any Error)?
    private var sender: (PCMChunk, CheckedContinuation<Void, any Error>)?
    private var receiver: CheckedContinuation<PCMChunk?, any Error>?
    init(capacity: Int) { self.capacity = capacity }
    func send(_ chunk: PCMChunk) async throws {
        try Task.checkCancellation()
        if let failure { throw failure }
        guard !closed else { throw CancellationError() }
        if let receiver { self.receiver = nil; receiver.resume(returning: chunk); return }
        if chunks.count < capacity { chunks.append(chunk); return }
        guard sender == nil else { throw PipelineError("MULTIPLE_PCM_PRODUCERS", stage: "tts") }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
                if Task.isCancelled { continuation.resume(throwing: CancellationError()) }
                else { sender = (chunk, continuation) }
            }
        } onCancel: { Task { await self.finish(error: CancellationError()) } }
    }
    func receive() async throws -> PCMChunk? {
        try Task.checkCancellation()
        if let failure { throw failure }
        if !chunks.isEmpty {
            let first = chunks.removeFirst()
            if let sender { self.sender = nil; chunks.append(sender.0); sender.1.resume() }
            return first
        }
        if closed { return nil }
        guard receiver == nil else { throw PipelineError("MULTIPLE_PCM_CONSUMERS", stage: "speaker") }
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                if Task.isCancelled { continuation.resume(throwing: CancellationError()) }
                else { receiver = continuation }
            }
        } onCancel: { Task { await self.finish(error: CancellationError()) } }
    }
    func finish(error: (any Error)? = nil) {
        guard !closed else { return }
        closed = true; failure = error
        if error != nil { chunks.removeAll() }
        if let sender { self.sender = nil; sender.1.resume(throwing: error ?? CancellationError()) }
        if let receiver {
            self.receiver = nil
            if let error { receiver.resume(throwing: error) } else { receiver.resume(returning: nil) }
        }
    }
    var bufferedCount: Int { chunks.count }
}
