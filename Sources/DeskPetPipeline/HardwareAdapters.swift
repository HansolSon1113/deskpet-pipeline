import Foundation
import DeskPetHardware

public struct HardwareMicrophone: MicrophoneSource, AudioInputEpochValidator {
    private let client: HardwareClient
    private let source: HardwareMicrophoneSource
    public init(client: HardwareClient) { self.client = client; source = HardwareMicrophoneSource(client: client) }
    public func isCurrent(_ inputEpoch: UInt64) -> Bool {
        guard let epoch = Int64(exactly: inputEpoch) else { return false }
        return client.isCurrentInputEpoch(epoch)
    }
    public func start() async throws { try await source.start() }
    public func stop() async throws { try await source.stop() }
    public func nextFrame() async throws -> AudioFrame? {
        guard let frame = try await source.nextFrame() else { return nil }
        return AudioFrame(chunk: PCMChunk(data: frame.chunk.data, sampleRate: frame.chunk.sampleRate,
                                         channels: frame.chunk.channels), timestamp: frame.timestamp, inputEpoch: UInt64(exactly: frame.inputEpoch))
    }
}

public actor HardwareSpeaker: SpeakerAdapter {
    private let speaker: HardwareSpeakerAdapter
    private var pending: [UUID: OutputEvent] = [:]
    public init(client: HardwareClient) { speaker = HardwareSpeakerAdapter(client: client) }
    // Wiki: TTS format is known at the first chunk, so defer native begin until then.
    public func begin(jobId: UUID, event: OutputEvent) { pending[jobId] = event }
    public func write(_ chunk: PCMChunk, jobId: UUID) async throws {
        if let event = pending.removeValue(forKey: jobId) {
            try await speaker.begin(jobId: jobId.uuidString, event: event.hardware,
                                    sampleRate: chunk.sampleRate, channels: chunk.channels)
        }
        try await speaker.write(DeskPetHardware.PCMChunk(data: chunk.data, sampleRate: chunk.sampleRate,
                                                       channels: chunk.channels), jobId: jobId.uuidString)
    }
    public func finish(jobId: UUID) async throws {
        guard pending.removeValue(forKey: jobId) == nil else { throw PipelineError("NO_AUDIO", stage: "speaker") }
        try await speaker.finish(jobId: jobId.uuidString)
    }
    public func stop(jobId: UUID) async {
        if pending.removeValue(forKey: jobId) == nil { try? await speaker.stop(jobId: jobId.uuidString) }
    }
}

/// Share this adapter's revision counter and consume client.events in one place.
public actor HardwareDisplay: DisplayAdapter {
    private let display: HardwareDisplayAdapter
    private var current: OutputEvent?
    private var startedOutputId: String?
    public init(client: HardwareClient) { display = HardwareDisplayAdapter(client: client) }
    public func show(_ event: OutputEvent) async throws {
        current = event
        var view = event.hardware
        // Wiki: begin acknowledgement and TTS generation are not audio.started.
        if event.displayMode == .face && event.activity == .speaking && startedOutputId != event.outputId.uuidString {
            view.activity = "processing"
        }
        try await display.show(view)
    }
    public func idle() async throws { current = nil; startedOutputId = nil; try await display.idle() }
    public func audioStarted(outputId: String) async throws {
        startedOutputId = outputId
        guard let current, current.outputId.uuidString == outputId, current.displayMode == .face else { return }
        var view = current.hardware; view.activity = "speaking"
        try await display.show(view)
    }
}

private extension OutputEvent {
    var hardware: DeskPetHardware.OutputEvent {
        let status = actionStatus.flatMap { [.succeeded, .failed, .unknown].contains($0) ? $0.rawValue : nil }
        let emotion = self.emotion == "cheerful" ? "happy" : self.emotion == "sad" ? "sad" : "neutral"
        return DeskPetHardware.OutputEvent(requestId: requestId.uuidString, outputId: outputId.uuidString,
            activity: activity.rawValue, emotion: emotion, displayState: displayState, displayMode: displayMode.rawValue,
            displayText: displayText, actionStatus: status, reminderKey: reminderKey)
    }
}
