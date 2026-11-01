#if os(macOS)
@preconcurrency import AVFoundation
import Foundation

private final class ConverterInput: @unchecked Sendable {
    private let lock = NSLock()
    private var buffer: AVAudioPCMBuffer?
    init(_ buffer: AVAudioPCMBuffer) { self.buffer = buffer }
    func take(_ state: UnsafeMutablePointer<AVAudioConverterInputStatus>) -> AVAudioPCMBuffer? {
        lock.withLock {
            guard let value = buffer else { state.pointee = .noDataNow; return nil }
            buffer = nil; state.pointee = .haveData; return value
        }
    }
}

/// Bounded capture; any overflow invalidates the current speech session through AudioIngress.
public final class MacMicrophoneSource: MicrophoneSource, AudioInputEpochValidator, @unchecked Sendable {
    private let engine = AVAudioEngine()
    private let lock = NSLock()
    private var frames: [AudioFrame] = []
    private var failure: PipelineError?
    private var running = false
    private var epoch: UInt64 = 0
    private var gated = false
    private var resumeAt = Date.distantPast
    private var tapInstalled = false
    public init() {}
    public func start() async throws {
        let allowed = await withCheckedContinuation { continuation in
            AVCaptureDevice.requestAccess(for: .audio) { continuation.resume(returning: $0) }
        }
        guard allowed else { throw PipelineError("MICROPHONE_PERMISSION_DENIED", stage: "audio") }
        let input = engine.inputNode
        let source = input.outputFormat(forBus: 0)
        guard source.sampleRate > 0, source.channelCount > 0,
              let target = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 16_000, channels: 1, interleaved: true),
              let converter = AVAudioConverter(from: source, to: target) else {
            throw PipelineError("MICROPHONE_FORMAT_UNAVAILABLE", stage: "audio")
        }
        lock.withLock { running = true; failure = nil; frames.removeAll() }
        input.installTap(onBus: 0, bufferSize: 1024, format: source) { [weak self] buffer, _ in
            guard let self else { return }
            let capacity = AVAudioFrameCount(ceil(Double(buffer.frameLength) * 16_000 / source.sampleRate)) + 32
            guard let out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else { return }
            let supplied = ConverterInput(buffer); var error: NSError?
            let status = converter.convert(to: out, error: &error) { _, state in
                supplied.take(state)
            }
            if status == .error { self.lock.withLock { self.failure = PipelineError("AUDIO_CONVERSION_FAILED", stage: "audio") }; return }
            guard let bytes = out.int16ChannelData?[0], out.frameLength > 0 else { return }
            let now = Date(); let pcm = Data(bytes: bytes, count: Int(out.frameLength) * 2)
            self.lock.withLock {
                guard self.running, !self.gated, now >= self.resumeAt else { return }
                guard self.frames.count < 128 else { self.frames.removeAll(); self.failure = PipelineError("CAPTURE_OVERFLOW", stage: "audio"); return }
                self.frames.append(AudioFrame(chunk: PCMChunk(data: pcm, sampleRate: 16_000, channels: 1), timestamp: now, inputEpoch: self.epoch))
            }
        }
        tapInstalled = true
        try engine.start()
    }
    public func nextFrame() async throws -> AudioFrame? {
        while true {
            try Task.checkCancellation()
            let state = lock.withLock { () -> (AudioFrame?, PipelineError?, Bool) in
                (frames.isEmpty ? nil : frames.removeFirst(), failure, running)
            }
            if let error = state.1 { throw error }
            if let frame = state.0 { return frame }
            if !state.2 { return nil }
            try await Task.sleep(for: .milliseconds(2))
        }
    }
    /// Capture continues; model input is invalidated at playback boundaries.
    public func playbackGate(_ enabled: Bool) {
        lock.withLock { gated = enabled; epoch &+= 1; frames.removeAll(); if !enabled { resumeAt = Date().addingTimeInterval(0.3) } }
    }
    public func isCurrent(_ inputEpoch: UInt64) -> Bool { lock.withLock { inputEpoch == epoch && !gated && Date() >= resumeAt } }
    public func stop() {
        engine.stop(); if tapInstalled { engine.inputNode.removeTap(onBus: 0); tapInstalled = false }
        lock.withLock { running = false; frames.removeAll() }
    }
    deinit { stop() }
}

public final class MacSpeakerAdapter: SpeakerAdapter, @unchecked Sendable {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let lock = NSLock()
    private let microphone: MacMicrophoneSource
    private var current: UUID?
    private var pending = 0
    private var failure = false
    private var formatRate: Int?
    private var outputId: UUID?
    private let onStarted: @Sendable (UUID) async -> Void
    public init(microphone: MacMicrophoneSource, onStarted: @escaping @Sendable (UUID) async -> Void = { _ in }) {
        self.onStarted = onStarted
        self.microphone = microphone; engine.attach(player)
        engine.mainMixerNode.outputVolume = 0.25
    }
    public func begin(jobId: UUID, event: OutputEvent) async throws {
        try Task.checkCancellation()
        guard lock.withLock({ current == nil }) else { throw PipelineError("SPEAKER_BUSY", stage: "speaker") }
        microphone.playbackGate(true)
        lock.withLock { current = jobId; pending = 0; failure = false; formatRate = nil; outputId = event.outputId }
    }
    public func write(_ chunk: PCMChunk, jobId: UUID) async throws {
        guard lock.withLock({ current == jobId }), chunk.channels == 1, chunk.data.count % 2 == 0,
              let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(chunk.sampleRate), channels: 1, interleaved: false),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(chunk.data.count/2)) else {
            throw PipelineError("SPEAKER_FORMAT", stage: "speaker")
        }
        while lock.withLock({ pending >= 4 }) { try await Task.sleep(for: .milliseconds(2)); try Task.checkCancellation() }
        guard lock.withLock({ current == jobId }) else { throw CancellationError() }
        if let rate = lock.withLock({ formatRate }) {
            guard rate == chunk.sampleRate else { throw PipelineError("SPEAKER_FORMAT_CHANGED", stage: "speaker") }
        } else {
            // The TTS sample rate is known only at the first chunk. Connecting
            // with the device's default rate would play 24 kHz PCM incorrectly.
            engine.stop(); engine.disconnectNodeOutput(player)
            engine.connect(player, to: engine.mainMixerNode, format: format)
            try engine.start(); lock.withLock { formatRate = chunk.sampleRate }
        }
        buffer.frameLength = buffer.frameCapacity
        chunk.data.withUnsafeBytes { raw in
            let bytes = raw.bindMemory(to: UInt8.self)
            for i in 0..<Int(buffer.frameLength) {
                buffer.floatChannelData![0][i] = Float(Int16(bitPattern: UInt16(bytes[2*i]) | UInt16(bytes[2*i+1]) << 8)) / 32768
            }
        }
        lock.withLock { pending += 1 }
        player.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            guard let self else { return }
            self.lock.withLock { if self.current == jobId { self.pending -= 1 } }
        }
        if !player.isPlaying {
            player.play()
            if let id = lock.withLock({ outputId }) { await onStarted(id) }
        }
    }
    public func finish(jobId: UUID) async throws {
        while lock.withLock({ current == jobId && pending > 0 }) { try await Task.sleep(for: .milliseconds(2)) }
        guard lock.withLock({ current == jobId && !failure && formatRate != nil }) else { throw PipelineError("AUDIO_OUTPUT_ERROR", stage: "speaker") }
        player.stop(); lock.withLock { current = nil; outputId = nil }; microphone.playbackGate(false)
    }
    public func stop(jobId: UUID) async {
        guard lock.withLock({ current == jobId }) else { return }
        player.stop(); lock.withLock { current = nil; pending = 0; failure = true }; microphone.playbackGate(false)
    }
    deinit { player.stop(); engine.stop() }
}
#endif
