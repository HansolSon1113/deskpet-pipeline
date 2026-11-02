import Foundation

public struct AudioFrame: Sendable {
    public var chunk: PCMChunk
    public var timestamp: Date
    public var inputEpoch: UInt64?
    public init(chunk: PCMChunk, timestamp: Date = Date(), inputEpoch: UInt64? = nil) {
        self.chunk = chunk; self.timestamp = timestamp; self.inputEpoch = inputEpoch
    }
}
public enum VoiceActivity: Sendable, Equatable { case silence, speech, end }
public protocol MicrophoneSource: Sendable {
    // Pull-based ingress. Device implementations must bound their capture backlog and
    // report lost frames; the pipeline does not retain an unbounded microphone stream.
    func nextFrame() async throws -> AudioFrame?
}
/// Source owns a capture watchdog and can intentionally pause delivery during playback.
public protocol ManagedMicrophoneSource: MicrophoneSource {}
public protocol VoiceActivityDetector: Sendable { func classify(_ frame: AudioFrame) async throws -> VoiceActivity }
public protocol ResettableVoiceActivityDetector: VoiceActivityDetector { func reset() async throws }
public protocol AudioInputEpochValidator: Sendable { func isCurrent(_ inputEpoch: UInt64) -> Bool }
public struct AudioIngressConfiguration: Sendable {
    public var sampleRate = 24_000
    public var channels = 1
    public var maxFrameBytes = 64_000
    public var maxSegmentBytes = 2_000_000
    public var frameTimeout: TimeInterval = 1
    public var microphoneTimeout: TimeInterval? = 1
    public var preRollBytes = 0
    public init() {}
}
// One capture/VAD loop forwards speech frames immediately. It never waits for STT
// inference or speaker playback. Overflow invalidates the utterance explicitly.
public struct AudioIngress: Sendable {
    public var microphone: any MicrophoneSource
    public var vad: any VoiceActivityDetector
    public var configuration: AudioIngressConfiguration
    public init(microphone: any MicrophoneSource, vad: any VoiceActivityDetector,
                configuration: AudioIngressConfiguration = .init()) {
        self.microphone = microphone; self.vad = vad; self.configuration = configuration
    }
    public func run(pipeline: DeskPetPipeline, conversationId: String) async throws {
        let config = configuration
        guard config.maxFrameBytes > 0, config.maxSegmentBytes > 0, config.sampleRate > 0,
              config.channels > 0, config.frameTimeout.isFinite, config.frameTimeout > 0,
              config.preRollBytes >= 0, config.preRollBytes <= config.maxSegmentBytes else {
            throw PipelineError("INVALID_CONFIGURATION", stage: "audio")
        }
        var session: UUID?; var bytes = 0; var damaged = false
        var epoch: UInt64?
        var preRoll: [AudioFrame] = []; var preRollSize = 0
        do {
            while !Task.isCancelled {
                let next: AudioFrame?
                if let limit = config.microphoneTimeout {
                    next = try await withTimeout(limit, stage: "microphone") { [microphone] in try await microphone.nextFrame() }
                } else { next = try await microphone.nextFrame() }
                guard let frame = next else {
                    if let session { await pipeline.cancelSpeech(sessionId: session) }
                    return
                }
                if frame.inputEpoch != epoch {
                    if let session { await pipeline.cancelSpeech(sessionId: session) }
                    session = nil; bytes = 0; damaged = false; epoch = frame.inputEpoch
                    preRoll.removeAll(); preRollSize = 0
                    if let resettable = vad as? any ResettableVoiceActivityDetector { try await resettable.reset() }
                }
                guard frame.chunk.data.count <= config.maxFrameBytes,
                      frame.chunk.sampleRate == config.sampleRate, frame.chunk.channels == config.channels else {
                    throw PipelineError("AUDIO_DEVICE_ERROR", stage: "audio.format")
                }
                if frame.inputEpoch != epoch {
                    if let session { await pipeline.cancelSpeech(sessionId: session) }
                    session = nil; bytes = 0; damaged = false; epoch = frame.inputEpoch
                    if let resettable = vad as? any ResettableVoiceActivityDetector { await resettable.reset() }
                }
                guard await pipeline.acceptsInput(epoch: frame.inputEpoch) else { continue }
                let activity = try await withTimeout(config.frameTimeout, stage: "vad") { [vad] in try await vad.classify(frame) }
                guard await pipeline.acceptsInput(epoch: frame.inputEpoch) else { continue }
                switch activity {
                case .silence:
                    if session == nil, !damaged, config.preRollBytes > 0, !frame.chunk.data.isEmpty {
                        preRoll.append(frame); preRollSize += frame.chunk.data.count
                        while preRollSize > config.preRollBytes || preRoll.count > 32 { preRollSize -= preRoll.removeFirst().chunk.data.count }
                    }
                case .speech:
                    guard !damaged else { continue }
                    if session == nil { bytes += preRollSize }
                    bytes += frame.chunk.data.count
                    if bytes > config.maxSegmentBytes {
                        damaged = true
                        if let session { await pipeline.cancelSpeech(sessionId: session) }
                        session = nil
                        await pipeline.reportAudioError(PipelineError("AUDIO_SEGMENT_OVERFLOW", stage: "vad"))
                        continue
                    }
                    do {
                        if session == nil {
                            session = try await pipeline.beginSpeech(conversationId: conversationId)
                            // Silero confirms speech after a short observation
                            // window. Keep the first consonants/PR number.
                            for saved in preRoll { try await pipeline.submitSpeechFrame(sessionId: session!, frame: saved, activity: .speech) }
                            preRoll.removeAll(); preRollSize = 0
                        }
                        try await pipeline.submitSpeechFrame(sessionId: session!, frame: frame, activity: .speech)
                    } catch {
                        if let session { await pipeline.cancelSpeech(sessionId: session) }
                        session = nil; damaged = true
                        await pipeline.reportAudioError(error as? PipelineError ?? PipelineError("STT_UNAVAILABLE", stage: "stt"))
                    }
                case .end:
                    if let session, !damaged {
                        do { try await pipeline.submitSpeechFrame(sessionId: session, frame: frame, activity: .end) }
                        catch { await pipeline.reportAudioError(error as? PipelineError ?? PipelineError("STT_UNAVAILABLE", stage: "stt")) }
                    }
                    session = nil; bytes = 0; damaged = false; preRoll.removeAll(); preRollSize = 0
                }
            }
            throw CancellationError()
        } catch {
            if let session { await pipeline.cancelSpeech(sessionId: session) }
            throw error
        }
    }
}
