import Foundation
import DeskPetPipeline
#if os(macOS)
import Darwin
#else
import Glibc
#endif

/// This mode never admits audio or loads VAD/STT models.
struct KeyboardOnlyRecognizer: SpeechRecognizer {
    func transcribe(_ segment: AudioSegment) async throws -> Transcript {
        throw PipelineError("KEYBOARD_INPUT_ONLY", stage: "input")
    }
}

struct KeyboardReadOnlyEureka: EurekaAdapter {
    let base: any EurekaAdapter
    func execute(_ request: ActionRequest) async throws -> ActionResult {
        guard !request.decision.headOutputs.intention.value.isMutation else {
            throw PipelineError("KEYBOARD_READ_ONLY", stage: "input.keyboard", message: "키보드 입력은 읽기 전용입니다.")
        }
        return try await base.execute(request)
    }
    func reconcile(_ request: ActionRequest) async throws -> ActionResult { try await base.reconcile(request) }
    func reminders() async throws -> [ReminderItem] { try await base.reminders() }
    func reminder(itemId: String) async throws -> ReminderItem? { try await base.reminder(itemId: itemId) }
}

/// Reads stdin on a dispatch queue so waiting for Enter does not block Swift tasks.
/// The queue owns the partial UTF-8 buffer; stream and cancellation are thread safe.
private final class KeyboardLineInput: @unchecked Sendable {
    let lines: AsyncThrowingStream<String, Error>
    private let continuation: AsyncThrowingStream<String, Error>.Continuation
    private let source: DispatchSourceRead
    private var buffer = Data()
    private let maxBytes: Int

    init(maxBytes: Int) {
        self.maxBytes = maxBytes
        let pair = AsyncThrowingStream<String, Error>.makeStream(bufferingPolicy: .bufferingOldest(16))
        lines = pair.stream; continuation = pair.continuation
        source = DispatchSource.makeReadSource(fileDescriptor: STDIN_FILENO,
            queue: DispatchQueue(label: "deskpet.keyboard"))
        source.setEventHandler { [weak self] in self?.readAvailable() }
        continuation.onTermination = { [weak self] _ in self?.source.cancel() }
        source.resume()
    }
    func cancel() { source.cancel(); continuation.finish() }
    private func fail(_ code: String, message: String = "") {
        source.cancel(); continuation.finish(throwing: PipelineError(code, stage: "input.keyboard", message: message))
    }
    private func yieldLine() {
        let line = String(decoding: buffer, as: UTF8.self); buffer.removeAll(keepingCapacity: true)
        if case .dropped = continuation.yield(line) { fail("KEYBOARD_QUEUE_FULL") }
    }
    private func readAvailable() {
        var bytes = [UInt8](repeating: 0, count: 4096)
        let count = bytes.withUnsafeMutableBytes { read(STDIN_FILENO, $0.baseAddress, $0.count) }
        if count == 0 {
            if !buffer.isEmpty { yieldLine() }
            cancel(); return
        }
        if count < 0 {
            let err = errno
            if err != EINTR && err != EAGAIN { fail("KEYBOARD_READ_FAILED", message: String(cString: strerror(err))) }
            return
        }
        for byte in bytes.prefix(count) {
            if byte == 10 { yieldLine() }
            else {
                guard buffer.count < maxBytes else { fail("INPUT_TOO_LARGE"); return }
                buffer.append(byte)
            }
        }
    }
}

extension NativeDemo {
    static func keyboardSession(pipeline: DeskPetPipeline, target: Target, reminder: String?) async throws {
        if let reminder {
            try await pipeline.startScenarioReminder(target: target, text: reminder, conversationId: "native-demo")
        }
        while await pipeline.inFlightCount() > 0 { try await Task.sleep(for: .milliseconds(20)) }
        let input = KeyboardLineInput(maxBytes: 4096)
        defer { input.cancel() }
        let session = Task {
            await NativeConsole.flush()
            FileHandle.standardOutput.write(Data("> ".utf8))
            for try await raw in input.lines {
                try Task.checkCancellation()
                let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
                if ["/quit", "/exit"].contains(text.lowercased()) { break }
                if !text.isEmpty {
                    await pipeline.interact(.speechStarted("native-demo"))
                    _ = try await pipeline.submit(Envelope(conversationId: "native-demo", kind: "stt.final",
                        payload: Transcript(text: text, sttConfidence: 0)))
                    while await pipeline.inFlightCount() > 0 { try await Task.sleep(for: .milliseconds(20)) }
                }
                await NativeConsole.flush()
                FileHandle.standardOutput.write(Data("> ".utf8))
            }
        }
        signal(SIGINT, SIG_IGN)
        let interrupt = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global())
        interrupt.setEventHandler { session.cancel(); input.cancel() }; interrupt.resume()
        defer { interrupt.cancel(); signal(SIGINT, SIG_DFL) }
        try await withTaskCancellationHandler {
            try await session.value
        } onCancel: { session.cancel(); input.cancel() }
    }
}
