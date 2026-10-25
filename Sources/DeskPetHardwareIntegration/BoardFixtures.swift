import Foundation
import DeskPetPipeline

/// Explicit synthetic fixtures: exercise PCM transport, never claim speech recognition.
public actor BoardVAD: ResettableVoiceActivityDetector {
    private var armed = false
    private var frames = 0
    public init() {}
    public func arm() { armed = true; frames = 0 }
    public func reset() { frames = 0 }
    public func classify(_ frame: AudioFrame) async throws -> VoiceActivity {
        guard armed else { return .silence }
        frames += 1
        if frames == 6 { armed = false; return .end }
        return .speech
    }
}
public actor BoardSTT: StreamingSpeechRecognizer {
    public init() {}
    public func transcribe(_ segment: AudioSegment) async throws -> Transcript { Transcript(text: "보드 연결 시험") }
    public func accept(_ frame: STTFrame) async throws -> Transcript? {
        guard frame.frame.chunk.sampleRate == 16000, frame.frame.chunk.channels == 1, frame.frame.chunk.data.count == 640 else {
            throw PipelineError("FIXTURE_PCM_MISMATCH", stage: "stt")
        }
        return frame.activity == .end ? Transcript(text: "보드 연결 시험") : nil
    }
    public func cancel(sessionId: UUID) async {}
}
