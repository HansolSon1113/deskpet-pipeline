import Foundation
import DeskPetPipeline
import DeskPetHardware
import struct DeskPetPipeline.OutputEvent
import struct DeskPetPipeline.AudioFrame
import struct DeskPetPipeline.PCMChunk

/// One session and one event consumer for all devices. Models remain injectable.
public actor HardwareBridge: SpeakerAdapter, ActivityDisplayAdapter, ManagedMicrophoneSource {
    public nonisolated let client: HardwareClient
    private let screen: HardwareDisplayAdapter
    private let speaker: HardwareSpeakerAdapter
    private let ears: HardwareMotionAdapter
    private let vad: (any ResettableVoiceActivityDetector)?
    private var pipeline: DeskPetPipeline?
    private var eventTask: Task<Void, Never>?
    private var resumeTask: Task<Void, Never>?
    private var pending: (UUID, OutputEvent)?
    private var deviceStarted = false
    private var listening = false
    private var speaking = false
    private var desired: OutputEvent?
    private var requestedActivity: Activity?
    private var invalidatedEpoch: Int64 = -1
    private var generation: UInt64 = 0
    private var notes: [String] = []
    private let motionEnabled: Bool

    public init(client: HardwareClient, vad: (any ResettableVoiceActivityDetector)? = nil, motionEnabled: Bool = false) {
        self.client = client; self.vad = vad; self.motionEnabled = motionEnabled
        screen = HardwareDisplayAdapter(client: client); speaker = HardwareSpeakerAdapter(client: client)
        ears = HardwareMotionAdapter(client: client)
    }
    public static func map(_ event: OutputEvent) -> DeskPetHardware.OutputEvent {
        var emotion = event.emotion == "cheerful" ? "happy" : event.emotion
        switch event.actionStatus {
        case .succeeded: emotion = "happy"
        case .failed: emotion = "sad"
        case .unknown: emotion = "neutral"
        default: break
        }
        return .init(requestId: event.requestId.uuidString, outputId: event.outputId.uuidString,
            activity: event.activity.rawValue, emotion: ["happy", "sad", "neutral"].contains(emotion) ? emotion : "neutral",
            displayState: "default",
            displayMode: event.displayMode.rawValue, displayText: event.displayText,
            actionStatus: event.actionStatus?.rawValue, reminderKey: event.reminderKey)
    }
    private func note(_ value: String) { if notes.count == 128 { notes.removeFirst() }; notes.append(value) }
    public func diagnostics() -> [String] { notes }
    public func start(pipeline: DeskPetPipeline, capture: Bool = true) async throws {
        guard eventTask == nil else { throw PipelineError("ALREADY_RUNNING", stage: "hardware") }
        self.pipeline = pipeline; client.holdInputDelivery()
        eventTask = Task { [weak self, client] in
            do { for try await event in client.events { await self?.handle(event) } }
            catch { await self?.failed(error) }
        }
        _ = try await client.command("touch.subscribe")
        if capture { try await client.startCapture() }
        await invalidate(epoch: client.inputEpoch)
        try await reopen(epoch: client.inputEpoch, generation: generation)
    }
    public func shutdown() async {
        generation &+= 1; resumeTask?.cancel(); resumeTask = nil
        client.holdInputDelivery(); client.close()
        eventTask?.cancel(); eventTask = nil; pipeline = nil
    }
    private func failed(_ error: Error) async {
        client.holdInputDelivery(); listening = false; generation &+= 1
        resumeTask?.cancel(); note("HARDWARE_ERROR: \(error)")
        await pipeline?.invalidateInput(epoch: client.inputEpoch)
        await pipeline?.reportAudioError(PipelineError("HARDWARE_ERROR", stage: "hardware"))
    }
    private func invalidate(epoch: Int64) async {
        guard epoch > invalidatedEpoch else { return }
        invalidatedEpoch = epoch
        await pipeline?.invalidateInput(epoch: epoch)
        await vad?.reset()
    }
    private func reopen(epoch: Int64, generation expected: UInt64) async throws {
        guard expected == generation, pending == nil, client.isCurrentInputEpoch(epoch) else { return }
        // Disable motion before listening; a missing/un-calibrated servo never pretends to move.
        if motionEnabled { try await ears.disable() }
        var view = desired.map(Self.map) ?? DeskPetHardware.OutputEvent(requestId: "input", outputId: "input-ready")
        view.activity = requestedActivity == .awaitingAnswer || desired?.activity == .awaitingAnswer ? "awaiting_answer" : "listening"
        try await screen.show(view)
        guard expected == generation, pending == nil, client.isCurrentInputEpoch(epoch) else { return }
        // A format rejection can occur before HardwareClient advances its epoch.
        await pipeline?.invalidateInput(epoch: epoch)
        guard expected == generation, pending == nil, client.isCurrentInputEpoch(epoch) else { return }
        await pipeline?.resumeInput(epoch: epoch)
        guard expected == generation, pending == nil else { return }
        listening = client.resumeInputDelivery(epoch: epoch)
    }
    private func scheduleReopen() {
        let expected = generation, epoch = client.inputEpoch
        resumeTask?.cancel()
        resumeTask = Task { [weak self] in
            do {
                try await Task.sleep(for: .milliseconds(320))
                await self?.finishReopen(epoch: epoch, generation: expected)
            } catch {}
        }
    }
    private func finishReopen(epoch: Int64, generation expected: UInt64) async {
        guard expected == generation else { return }
        // Once an ack is in flight, use generation checks rather than cancelling its
        // Task: HardwareClient intentionally closes the shared session on cancellation.
        resumeTask = nil
        do { try await reopen(epoch: epoch, generation: expected) }
        catch { await failed(error) }
    }
    private func handle(_ event: HardwareEvent) async {
        do {
            switch event.name {
            case "input.invalidated": await invalidate(epoch: event.fields["inputEpoch"]?.integer ?? client.inputEpoch)
            case "touch.event":
                let t = try TouchEvent(event: event)
                await pipeline?.interact(.touch(.init(phase: t.phase, contactId: t.contactId, x: t.x, y: t.y, timestamp: t.timestamp, simulated: t.simulated)))
                note("touch:\(t.phase):\(t.contactId)")
            case "audio.started":
                guard event.fields["jobId"]?.string == pending?.0.uuidString else { return }
                speaking = true
                if let desired { try await show(desired) }
            case "audio.failed", "audio.cancelled":
                note("\(event.name):\(event.fields["code"]?.string ?? "")")
                if event.fields["stoppedConfirmed"] != .bool(true) {
                    await pipeline?.reportAudioError(PipelineError("AUDIO_STOP_UNCONFIRMED", stage: "speaker"))
                }
            case "device.error": note("device.error:\(event.fields)")
            default: break
            }
        } catch { await failed(error) }
    }
    public func nextFrame() async throws -> AudioFrame? {
        guard let frame = try await client.nextFrame() else { return nil }
        return .init(chunk: .init(data: frame.chunk.data, sampleRate: frame.chunk.sampleRate, channels: frame.chunk.channels),
                     timestamp: frame.timestamp, inputEpoch: frame.inputEpoch)
    }
    public func begin(jobId: UUID, event: OutputEvent) async throws {
        guard pending == nil else { throw PipelineError("BUSY", stage: "speaker") }
        pending = (jobId, event); deviceStarted = false; speaking = false; listening = false
        generation &+= 1; resumeTask?.cancel(); client.holdInputDelivery()
        // Invalidate even before the first PCM chunk establishes the playback format.
        await pipeline?.invalidateInput(epoch: client.inputEpoch)
        await vad?.reset()
    }
    public func write(_ chunk: PCMChunk, jobId: UUID) async throws {
        guard let (id, event) = pending, id == jobId else { throw PipelineError("UNKNOWN_JOB", stage: "speaker") }
        if !deviceStarted {
            await invalidate(epoch: client.inputEpoch + 1)
            try await speaker.begin(jobId: jobId.uuidString, event: Self.map(event), sampleRate: chunk.sampleRate, channels: chunk.channels)
            guard pending?.0 == jobId else { throw CancellationError() }
            deviceStarted = true
        }
        try await speaker.write(.init(data: chunk.data, sampleRate: chunk.sampleRate, channels: chunk.channels), jobId: jobId.uuidString)
    }
    public func finish(jobId: UUID) async throws {
        guard pending?.0 == jobId, deviceStarted else { throw PipelineError("EMPTY_AUDIO", stage: "speaker") }
        try await speaker.finish(jobId: jobId.uuidString)
        guard pending?.0 == jobId else { throw CancellationError() }
        pending = nil; deviceStarted = false; speaking = false; scheduleReopen()
    }
    public func stop(jobId: UUID) async {
        guard pending?.0 == jobId else { return }
        do { if deviceStarted { try await speaker.stop(jobId: jobId.uuidString) } }
        catch { note("STOP_ERROR: \(error)") }
        pending = nil; deviceStarted = false; speaking = false; scheduleReopen()
    }
    public func show(_ event: OutputEvent) async throws {
        desired = event; requestedActivity = nil
        var view = Self.map(event)
        if event.activity == .speaking && !speaking { view.activity = "processing" }
        try await screen.show(view)
        if let motion = event.motion {
            guard ["neutral", "perk", "droop", "wiggle"].contains(motion.name), motion.intensity.isFinite, (0...1).contains(motion.intensity) else {
                note("UNSUPPORTED_MOTION:\(motion.name)"); return
            }
            guard motionEnabled, !listening else { note("MOTION_SUPPRESSED"); return }
            do {
                try await ears.enable()
                try await ears.apply(preset: motion.name, intensity: motion.intensity, requestId: event.requestId.uuidString, outputId: event.outputId.uuidString)
            } catch { note("MOTION_ERROR: \(error)") }
        }
    }
    public func activity(_ state: Activity) async throws {
        requestedActivity = state
        if state == .listening || state == .processing { desired = nil }
        if (state == .listening || state == .awaitingAnswer) && !client.isCurrentInputEpoch(client.inputEpoch) { return }
        var view = desired.map(Self.map) ?? DeskPetHardware.OutputEvent(requestId: "pipeline-state", outputId: "activity")
        view.activity = state.rawValue
        try await screen.show(view)
    }
    public func idle() async throws { desired = nil; requestedActivity = .idle; try await screen.idle() }
}
