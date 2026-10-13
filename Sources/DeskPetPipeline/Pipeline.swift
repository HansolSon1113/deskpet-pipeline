import Foundation

public struct PipelineSnapshot: Sendable {
    public var queued: [String: Int]
    public var modules: [String: ModuleSnapshot]
    public var activeRequests: Int
    public var apiRequests: Int
}
public struct RequestRecord: Sendable {
    public var requestId: UUID
    public var action: ActionResult?
    public var operation: ActionRequest?
    public var delivery: DeliveryResult?
}
private struct Job: Sendable {
    var envelope: Envelope<Transcript>
    var token = UUID()
    var admitted = ContinuousClock.now
    var input: RouterInput?
    var stamp: ContextStamp?
    var decision: RouteDecision?
    var reminder: ReminderItem?
    var output: OutputRequest?
    var event: OutputEvent?
    var terminal = true
    var operationId = UUID()
    var targetResolution: TargetResolution?
    var executionToken: UUID { terminal ? token : (output?.outputId ?? token) }
}

private struct AudioJob: Sendable { var job: Job; var segment: AudioSegment? = nil; var frame: STTFrame? = nil }
private enum DisplayCommand: Sendable { case output(Job), idle }
private struct PlaybackJob: Sendable { var job: Job; var pcm: PCMChannel; var deadline: Date }
private struct DeliveryState: Sendable {
    var job: Job
    var started = ContinuousClock.now
    var audio: DeliveryStatus?
    var display: DeliveryStatus?
    var errors: [PipelineError] = []
    var fallbackPending = false
    var contextStored = false
    var storingContext = false
}

public actor DeskPetPipeline {
    public nonisolated let events: AsyncStream<PipelineEvent>
    private let eventSink: AsyncStream<PipelineEvent>.Continuation
    private let config: PipelineConfiguration
    private let adapters: Adapters
    private let context: RouterContextStore
    private let metrics = Metrics()
    private let sttQueue: Mailbox<AudioJob>
    private let routerQueue: Mailbox<Job>
    private let outputQueue: Mailbox<Job>
    private let ttsQueue: Mailbox<Job>
    private let playbackQueue: Mailbox<PlaybackJob>
    private let displayQueue: Mailbox<DisplayCommand>
    private let webQueue: Mailbox<Job>
    private var deliveries: [UUID: DeliveryState] = [:]
    private var pcmChannels: [UUID: PCMChannel] = [:]
    private var modules: [String: any ModuleControl] = [:]
    private var eurekaModule: PipelineModule<Job, any EurekaAdapter, HarnessResult>?
    private var harnessModule: PipelineModule<Job, any HarnessAdapter, HarnessResult>?
    private var apiInFlight: [UUID: Bool] = [:]
    private var reminderLoop: Task<Void, Never>?
    private var starting = false
    private var running = false
    private var stopped = false
    private var active: [UUID: Job] = [:]
    private var latest: [String: UUID] = [:]
    private var cancellation: [UUID: [UUID: @Sendable () -> Void]] = [:]
    private var progressTimers: [UUID: Task<Void, Never>] = [:]
    private var questionTimer: Task<Void, Never>?
    private var displayTimer: Task<Void, Never>?
    private var records: [UUID: RequestRecord] = [:]
    private var recordOrder: [UUID] = []
    private var reminderHistory: [String: (due: Date, delivered: Bool, snooze: Date?)] = [:]
    private var currentReminder: ReminderItem?
    private var quietUntil = Date.distantPast
    private var waitingQuestion: (conversation: String, request: UUID, output: UUID)?
    private var droppedEvents = 0
    private var speechSessions: [UUID: Job] = [:]

    public init(adapters: Adapters, configuration: PipelineConfiguration = .init()) throws {
        try configuration.validate(); config = configuration; self.adapters = adapters
        context = RouterContextStore(configuration)
        sttQueue = Mailbox(capacity: max(configuration.audioQueueCapacity, configuration.streamingFrameQueueCapacity))
        routerQueue = Mailbox(capacity: configuration.queueCapacity)
        outputQueue = Mailbox(capacity: configuration.queueCapacity)
        ttsQueue = Mailbox(capacity: configuration.queueCapacity)
        playbackQueue = Mailbox(capacity: configuration.audioQueueCapacity)
        displayQueue = Mailbox(capacity: configuration.queueCapacity)
        webQueue = Mailbox(capacity: configuration.queueCapacity)
        let pair = AsyncStream<PipelineEvent>.makeStream(bufferingPolicy: .bufferingOldest(configuration.eventCapacity))
        events = pair.stream; eventSink = pair.continuation
    }
    public func start(automaticReminders: Bool = true) async {
        guard !starting, !running, !stopped else { return }; starting = true
        install("stt", queue: sttQueue, resource: adapters.stt, limit: 1,
            execute: { [weak self] input, model in
                guard let self else { throw CancellationError() }; return try await self.recognize(input, model: model)
            }, process: { [weak self] input, result in await self?.recognized(input, result: result) })
        install("router", queue: routerQueue, resource: adapters.router, limit: config.queueCapacity,
            execute: { [weak self] input, model in
                guard let self else { throw CancellationError() }; return try await self.inferRoute(input, model: model)
            }, process: { [weak self] input, result in await self?.routed(input, result: result) })
        install("output", queue: outputQueue, resource: adapters.output, limit: config.queueCapacity,
            execute: { [weak self] input, model in
                guard let self else { throw CancellationError() }; return try await self.generate(input, model: model)
            }, process: { [weak self] input, result in await self?.generated(input, result: result) })
        install("tts", queue: ttsQueue, resource: adapters.tts, limit: 1,
            execute: { [weak self] input, model in
                guard let self else { throw CancellationError() }; try await self.synthesize(input, model: model)
            }, process: { [weak self] input, result in
                if case .failure = result { await self?.audioCompleted(input, success: false, error: PipelineError("TTS_ERROR", stage: "tts")) }
            })
        install("playback", queue: playbackQueue, resource: adapters.speaker, limit: 1,
            execute: { [weak self] input, speaker in
                guard let self else { throw CancellationError() }; return await self.play(input, speaker: speaker)
            }, process: { [weak self] input, result in
                let success = (try? result.get()) ?? false
                await self?.audioCompleted(input.job, success: success, error: success ? nil : PipelineError("TTS_ERROR", stage: "speaker"))
            })
        install("display", queue: displayQueue, resource: adapters.display, limit: 1,
            execute: { [weak self] input, display in
                guard let self else { throw CancellationError() }; return await self.displayCommand(input, display: display)
            }, process: { [weak self] input, result in
                if case .output(let job) = input {
                    await self?.displayCompleted(job, success: (try? result.get()) ?? false, fallback: job.event?.displayMode == .text)
                }
            })
        if let web = adapters.web {
            install("web", queue: webQueue, resource: web, limit: 1,
                execute: { [weak self] input, web in
                    guard let self else { throw CancellationError() }; try await self.publish(input, web: web)
                }, process: { [weak self] input, result in
                    if case .failure = result { await self?.webFailed(input) }
                })
        }
        let eureka = PipelineModule(resource: adapters.eureka, input: nil, limit: config.maxConcurrentEurekaRequests,
            execute: { [weak self] (job: Job, api: any EurekaAdapter) in
                guard let self else { throw CancellationError() }; return try await self.callAPI(job, eureka: api, harness: nil)
            }, process: { [weak self] input, result in await self?.apiCompleted(input, result: result, harness: false) })
        let harness = PipelineModule(resource: adapters.harness, input: nil, limit: config.maxConcurrentHarnessRequests,
            execute: { [weak self] (job: Job, api: any HarnessAdapter) in
                guard let self else { throw CancellationError() }; return try await self.callAPI(job, eureka: nil, harness: api)
            }, process: { [weak self] input, result in await self?.apiCompleted(input, result: result, harness: true) })
        eurekaModule = eureka; harnessModule = harness; modules["eureka"] = eureka; modules["harness"] = harness
        for module in modules.values { await module.start() }
        starting = false
        guard !stopped else { return }; running = true
        if automaticReminders {
            reminderLoop = Task { [weak self, config] in
                while !Task.isCancelled {
                    await self?.pollReminders()
                    do { try await Task.sleep(for: .seconds(config.reminderPollInterval)) } catch { return }
                }
            }
        }
    }
    private func install<Input: Sendable, Resource: Sendable, Output: Sendable>(
        _ name: String, queue: Mailbox<Input>, resource: Resource, limit: Int,
        execute: @escaping @Sendable (Input, Resource) async throws -> Output,
        process: @escaping @Sendable (Input, Result<Output, any Error>) async -> Void) {
        let module = PipelineModule(resource: resource, input: queue, limit: limit, execute: execute, process: process)
        modules[name] = module
    }
    private func webFailed(_ job: Job) { emit(.error(job.envelope.requestId, PipelineError("WEB_ERROR", stage: "web"))) }
    public func snapshot() async -> PipelineSnapshot {
        let queued = ["stt": await sttQueue.count, "router": await routerQueue.count,
                      "output": await outputQueue.count, "tts": await ttsQueue.count,
                      "playback": await playbackQueue.count, "display": await displayQueue.count,
                      "web": await webQueue.count]
        var states: [String: ModuleSnapshot] = [:]
        for (name, module) in modules { states[name] = await module.snapshot() }
        return PipelineSnapshot(queued: queued, modules: states, activeRequests: active.count, apiRequests: apiInFlight.count)
    }
    public func shutdown() async {
        guard !stopped else { return }; stopped = true; running = false
        questionTimer?.cancel(); displayTimer?.cancel()
        for timer in progressTimers.values { timer.cancel() }; progressTimers.removeAll()
        for operations in cancellation.values { for cancel in operations.values { cancel() } }
        for channel in pcmChannels.values { await channel.finish(error: CancellationError()) }
        reminderLoop?.cancel()
        if let streaming = adapters.stt as? any StreamingSpeechRecognizer {
            for session in speechSessions.keys { await streaming.cancel(sessionId: session) }
        }
        speechSessions.removeAll()
        for module in modules.values { await module.stop() }
        await sttQueue.close(); await routerQueue.close(); await outputQueue.close()
        await ttsQueue.close(); await playbackQueue.close(); await displayQueue.close(); await webQueue.close()
        if let reminderLoop { await reminderLoop.value }
        if let executor = adapters.gpuExecutor { await executor.shutdown() }
        apiInFlight.removeAll()
        active.removeAll(); latest.removeAll(); cancellation.removeAll()
        deliveries.removeAll(); pcmChannels.removeAll(); eventSink.finish()
    }
    private func emit(_ event: PipelineEvent) {
        if case .dropped = eventSink.yield(event) { droppedEvents += 1 }
    }
    public func isAudioInputAllowed() -> Bool { running }
    public func reportAudioError(_ error: PipelineError) { emit(.error(nil, error)) }
    public func droppedEventCount() -> Int { droppedEvents }
    public func record(requestId: UUID) -> RequestRecord? { records[requestId] }
    public func measurements() async -> [String: StageMeasurement] { await metrics.snapshot() }
    public func contextTurns(conversationId: String) async -> [ContextTurn] { await context.turns(conversationId) }
    public func inFlightCount() -> Int { active.count }

    @discardableResult public func submit(_ original: Envelope<Transcript>) async throws -> UUID {
        try await admitTranscript(original, replacingAudio: nil)
    }
    private func admitTranscript(_ original: Envelope<Transcript>, replacingAudio: UUID?) async throws -> UUID {
        guard running else { throw PipelineError("NOT_RUNNING", stage: "admission") }
        guard original.payload.text.utf8.count <= config.maxTranscriptBytes,
              original.conversationId.utf8.count <= 256,
              original.payload.language.utf8.count <= 32 else { throw PipelineError("INPUT_TOO_LARGE", stage: "admission") }
        guard original.payload.isFinal else { emit(.partial(original)); return original.requestId }
        guard !original.payload.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw PipelineError("NO_SPEECH", stage: "admission")
        }
        guard original.deadlineAt.map({ $0 > Date() }) ?? true else { throw PipelineError("DEADLINE_EXCEEDED", stage: "admission") }
        var envelope = original
        let pending = await context.pendingRequest(envelope.conversationId)
        var preResolved: TargetResolution?
        if let pending, records[pending]?.action?.status == .needsClarification {
            let target = await context.activeTarget(envelope.conversationId)
            let resolution = try await withTimeout(config.routerTimeout, stage: "clarification.target") { [adapters, original] in
                try await adapters.router.resolveTarget(in: original.payload)
            }
            preResolved = resolution
            let switchesTarget: Bool
            if case .unique(let resolved) = resolution, let target { switchesTarget = !target.isSameEntity(as: resolved) }
            else { switchesTarget = false }
            if !switchesTarget, await context.pendingRequest(envelope.conversationId) == pending { envelope.requestId = pending }
        }
        if let replacingAudio {
            guard let audio = active[replacingAudio], current(audio) else { throw PipelineError("CANCELLED", stage: "stt.admission") }
        }
        if let existing = records[envelope.requestId]?.action, existing.status != .needsClarification {
            throw PipelineError("DUPLICATE_REQUEST", stage: "admission")
        }
        if active.values.contains(where: { $0.token != replacingAudio && ($0.envelope.turnId == envelope.turnId || $0.envelope.requestId == envelope.requestId) }) {
            throw PipelineError("DUPLICATE_TURN", stage: "admission")
        }
        try reserveRecord(envelope.requestId)
        guard active.count - (replacingAudio == nil ? 0 : 1) < config.maxInFlight else { throw PipelineError("QUEUE_FULL", stage: "admission") }
        envelope.deadlineAt = min(envelope.deadlineAt ?? .distantFuture, Date().addingTimeInterval(config.requestTimeout))
        let reservation = try await routerQueue.reserve()
        let audioStillCurrent = replacingAudio.map { token in active[token].map { current($0) } ?? false } ?? true
        guard running, audioStillCurrent, active.count - (replacingAudio == nil ? 0 : 1) < config.maxInFlight,
              !active.values.contains(where: { $0.token != replacingAudio && $0.envelope.requestId == envelope.requestId }) else {
            await routerQueue.release(reservation)
            throw PipelineError("QUEUE_FULL", stage: "admission")
        }
        var job = Job(envelope: envelope)
        if let replacingAudio, let previous = active[replacingAudio] { job.admitted = previous.admitted }
        job.targetResolution = preResolved
        let previous = latest[envelope.conversationId]
        latest[envelope.conversationId] = job.token; active[job.token] = job
        do { try await routerQueue.commit(job, reservation: reservation) }
        catch {
            active.removeValue(forKey: job.token)
            if latest[envelope.conversationId] == job.token { latest[envelope.conversationId] = previous }
            throw error
        }
        if let previous {
            if previous == replacingAudio { active.removeValue(forKey: previous) }
            else { await cancelToken(previous) }
        }
        await cancelReminder()
        clearQuestion(); displayTimer?.cancel()
        if let operations = cancellation[idleDisplayToken] { for cancel in operations.values { cancel() } }
        emit(.accepted(envelope.requestId)); emit(.activity(envelope.conversationId, .processing))
        return envelope.requestId
    }
    // Admission returns immediately after the bounded STT queue accepts the segment.
    @discardableResult public func submitAudio(_ segment: AudioSegment, conversationId: String) async throws -> UUID {
        guard running else { throw PipelineError("NOT_RUNNING", stage: "stt") }
        guard segment.pcm.count <= config.maxAudioBytes, conversationId.utf8.count <= 256 else {
            throw PipelineError("INPUT_TOO_LARGE", stage: "stt")
        }
        let reservation = try await sttQueue.reserve()
        guard running, active.count < config.maxInFlight,
              active.values.filter({ $0.envelope.kind == "audio.segment" }).count < config.audioQueueCapacity else {
            await sttQueue.release(reservation); throw PipelineError("QUEUE_FULL", stage: "stt")
        }
        let envelope = Envelope(conversationId: conversationId,
            deadlineAt: Date().addingTimeInterval(config.requestTimeout), kind: "audio.segment", payload: Transcript(text: ""))
        do { try reserveRecord(envelope.requestId) }
        catch { await sttQueue.release(reservation); throw error }
        let job = Job(envelope: envelope); let previous = latest[conversationId]
        active[job.token] = job; latest[conversationId] = job.token
        do { try await sttQueue.commit(AudioJob(job: job, segment: segment), reservation: reservation) }
        catch { active.removeValue(forKey: job.token); latest[conversationId] = previous; throw error }
        if let previous { await cancelToken(previous) }
        await cancelReminder(); clearQuestion(); displayTimer?.cancel()
        if let operations = cancellation[idleDisplayToken] { for cancel in operations.values { cancel() } }
        emit(.accepted(envelope.requestId)); return envelope.requestId
    }
    // Streaming sessions retain no PCM in the coordinator. VAD frames are admitted
    // independently of inference completion; the STT module consumes them in order.
    public func beginSpeech(conversationId: String) async throws -> UUID {
        guard running, active.count < config.maxInFlight else { throw PipelineError("QUEUE_FULL", stage: "stt.stream") }
        guard adapters.stt is any StreamingSpeechRecognizer else { throw PipelineError("STREAMING_STT_UNAVAILABLE", stage: "stt.stream") }
        guard conversationId.utf8.count <= 256 else { throw PipelineError("INPUT_TOO_LARGE", stage: "stt.stream") }
        let envelope = Envelope(conversationId: conversationId, deadlineAt: Date().addingTimeInterval(config.requestTimeout),
                                kind: "audio.stream", payload: Transcript(text: ""))
        try reserveRecord(envelope.requestId)
        let job = Job(envelope: envelope); let previous = latest[conversationId]
        active[job.token] = job; latest[conversationId] = job.token; speechSessions[job.token] = job
        if let previous { await cancelToken(previous) }
        await interact(.speechStarted(conversationId)); emit(.accepted(envelope.requestId))
        return job.token
    }
    public func submitSpeechFrame(sessionId: UUID, frame: AudioFrame, activity: VoiceActivity) async throws {
        guard let job = speechSessions[sessionId], current(job) else { throw PipelineError("CANCELLED", stage: "stt.stream") }
        guard frame.chunk.data.count <= config.maxChunkBytes else { throw PipelineError("INPUT_TOO_LARGE", stage: "stt.frame") }
        do { try await sttQueue.offer(AudioJob(job: job, frame: STTFrame(sessionId: sessionId, frame: frame, activity: activity))) }
        catch { await cancelSpeech(sessionId: sessionId); throw error }
    }
    public func cancelSpeech(sessionId: UUID) async { await cancelToken(sessionId) }
    private func recognize(_ audio: AudioJob, model: any SpeechRecognizer) async throws -> Transcript? {
        let job = audio.job
        guard current(job) else { throw CancellationError() }
        let start = ContinuousClock.now
        defer { Task { [metrics] in await metrics.record("stt", since: start) } }
        return try await perform(token: job.token, timeout: timeout(job, config.sttTimeout), stage: "stt") {
            if let frame = audio.frame, let streaming = model as? any StreamingSpeechRecognizer {
                return try await streaming.accept(frame)
            }
            guard let segment = audio.segment else { throw PipelineError("INVALID_AUDIO_INPUT", stage: "stt") }
            return try await model.transcribe(segment)
        }
    }
    private func recognized(_ audio: AudioJob, result: Result<Transcript?, any Error>) async {
        let job = audio.job
        do {
            guard let transcript = try result.get() else {
                if audio.frame?.activity == .end { await cancelSpeech(sessionId: job.token) }
                return
            }
            guard current(job) else { finish(job); return }
            guard transcript.isFinal else {
                var partial = job.envelope; partial.kind = "stt.partial"; partial.payload = transcript; emit(.partial(partial))
                if audio.frame == nil { finish(job) }; return
            }
            speechSessions.removeValue(forKey: job.token)
            var envelope = job.envelope; envelope.kind = "stt.final"; envelope.payload = transcript
            _ = try await admitTranscript(envelope, replacingAudio: job.token)
        } catch {
            if speechSessions[job.token] != nil { await cancelSpeech(sessionId: job.token) }
            reject(job, error: error)
        }
    }
    private func reserveRecord(_ request: UUID) throws {
        guard records[request] == nil else { return }
        if records.count >= config.journalCapacity {
            guard let evict = recordOrder.first(where: { id in
                !active.values.contains(where: { $0.envelope.requestId == id }) && records[id]?.action?.status != .unknown
            }) else { throw PipelineError("JOURNAL_FULL", stage: "admission") }
            records.removeValue(forKey: evict); recordOrder.removeAll { $0 == evict }
        }
        records[request] = RequestRecord(requestId: request); recordOrder.append(request)
    }
    private func current(_ job: Job) -> Bool {
        running && active[job.token] != nil && latest[job.envelope.conversationId] == job.token &&
        (job.envelope.deadlineAt ?? .distantPast) > Date()
    }
    private func timeout(_ job: Job, _ stage: TimeInterval) -> TimeInterval {
        min(stage, (job.envelope.deadlineAt ?? Date()).timeIntervalSinceNow)
    }
    private func perform<T: Sendable>(token: UUID, timeout: TimeInterval, stage: String,
                                     operation: @escaping @Sendable () async throws -> T) async throws -> T {
        let task = Task { try await withTimeout(timeout, stage: stage, operation: operation) }
        let operationId = UUID()
        cancellation[token, default: [:]][operationId] = { task.cancel() }
        defer {
            cancellation[token]?.removeValue(forKey: operationId)
            if cancellation[token]?.isEmpty == true { cancellation.removeValue(forKey: token) }
        }
        return try await withTaskCancellationHandler { try await task.value } onCancel: { task.cancel() }
    }
    public func cancel(conversationId: String) async { await cancelConversation(conversationId); clearQuestion() }
    private func cancelConversation(_ conversation: String) async {
        guard let token = latest.removeValue(forKey: conversation) else { return }
        await cancelToken(token)
    }
    private func cancelToken(_ token: UUID) async {
        if let operations = cancellation[token] { for cancel in operations.values { cancel() } }
        if speechSessions.removeValue(forKey: token) != nil, let streaming = adapters.stt as? any StreamingSpeechRecognizer {
            await streaming.cancel(sessionId: token)
        }
        if let channel = pcmChannels[token] { await channel.finish(error: CancellationError()) }
        progressTimers.removeValue(forKey: token)?.cancel()
        // Local modules can retire the request immediately; their bounded workers still
        // finish cancellation cleanup. API tasks retain it until execution status is known.
        let retiring = active[token]

        await sttQueue.remove { $0.job.token == token }
        await routerQueue.remove { $0.token == token }; await outputQueue.remove { $0.token == token }
        await ttsQueue.remove { $0.token == token }; await playbackQueue.remove { $0.job.token == token }
        await displayQueue.remove { if case .output(let job) = $0 { return job.token == token }; return false }
        let outputs = deliveries.filter { $0.value.job.token == token }
        for (id, state) in outputs {
            if let operations = cancellation[state.job.executionToken] { for cancel in operations.values { cancel() } }
            if let channel = pcmChannels.removeValue(forKey: state.job.executionToken) { await channel.finish(error: CancellationError()) }
            deliveries.removeValue(forKey: id)
        }
        if let retiring, apiInFlight[token] == nil { finish(retiring) }
        // An in-flight mutation is allowed to report unknown/confirmed execution independently of output.
    }
    private func finish(_ job: Job) {
        guard active.removeValue(forKey: job.token) != nil else { return }
        if latest[job.envelope.conversationId] == job.token { latest.removeValue(forKey: job.envelope.conversationId) }
        progressTimers.removeValue(forKey: job.token)?.cancel()
        pcmChannels.removeValue(forKey: job.token)
        for (id, state) in deliveries where state.job.token == job.token { deliveries.removeValue(forKey: id) }
        if let reminder = job.reminder, reminderHistory[reminder.key]?.delivered != true {
            reminderHistory.removeValue(forKey: reminder.key)
        }
        Task { [metrics] in await metrics.record("end_to_end", since: job.admitted) }
    }
    private func reject(_ job: Job, error: Error) {
        let fault = error as? PipelineError ?? PipelineError(error is CancellationError ? "CANCELLED" : "ADAPTER_ERROR", stage: "pipeline")
        emit(.error(job.envelope.requestId, fault)); if job.terminal { finish(job) }
    }
    private func enqueue(_ job: Job, into queue: Mailbox<Job>) async {
        guard current(job) else { if job.terminal { finish(job) }; return }
        do { try await queue.offer(job) } catch { reject(job, error: error) }
    }
    private func inferRoute(_ original: Job, model: any RouterModel) async throws -> (RouterInput, RouteDecision) {
        guard current(original) else { throw CancellationError() }
        let job = original
        let start = ContinuousClock.now
        defer { Task { [metrics] in await metrics.record("router", since: start) } }
        return try await perform(token: job.token, timeout: timeout(job, config.routerTimeout + (adapters.remoteRouter == nil ? 0 : config.remoteTimeout)), stage: "router") { [context, adapters, config, job, model] in
                let resolution: TargetResolution
                if let cached = job.targetResolution { resolution = cached }
                else {
                    resolution = try await withTimeout(config.routerTimeout, stage: "router.target") {
                        try await model.resolveTarget(in: job.envelope.payload)
                    }
                }
                if case .unique(let target) = resolution {
                    guard target.kind.utf8.count <= 64, target.entityId.utf8.count <= 256,
                          target.name.utf8.count <= 1_024 else { throw PipelineError("INVALID_TARGET", stage: "router") }
                }
                let input = await context.prepare(job.envelope, resolution: resolution)
                let heads: HeadOutputs
                do { heads = try await withTimeout(config.routerTimeout, stage: "router.heads") { try await model.heads(for: input) } }
                catch {
                    try Task.checkCancellation()
                    guard let remote = adapters.remoteRouter else { throw error }
                    heads = try await withTimeout(min(config.remoteTimeout, job.envelope.deadlineAt!.timeIntervalSinceNow), stage: "remote.router") {
                        try await remote.heads(for: input, deadlineAt: job.envelope.deadlineAt!)
                    }
                }
                let strings = [heads.target.value, heads.name.value, heads.priority.value, heads.task.value, heads.note.value]
                guard strings.allSatisfy({ ($0?.utf8.count ?? 0) <= config.maxTranscriptBytes }),
                      heads.dueAt.value.map({ (try? DateCodec.epochMilliseconds($0)) != nil }) ?? true else {
                    throw PipelineError("INVALID_HEAD_VALUE", stage: "router")
                }
                let masked = DecisionComposer.mask(heads)
                let provisional = DecisionComposer.compose(heads, confidence: 1, input: input, resolution: resolution,
                                                           threshold: config.routeConfidenceThreshold, sttThreshold: config.sttConfidenceThreshold)
                let score = try await withTimeout(config.routerTimeout, stage: "router.confidence") {
                    try await model.confidence(for: masked, missingSlots: provisional.missingRequiredSlots)
                }
                return (input, DecisionComposer.compose(heads, confidence: score, input: input, resolution: resolution,
                                                       threshold: config.routeConfidenceThreshold, sttThreshold: config.sttConfidenceThreshold))
            }
    }
    private func routed(_ original: Job, result outcome: Result<(RouterInput, RouteDecision), any Error>) async {
        var job = original
        do {
            let result = try outcome.get()
            guard current(job) else { finish(job); return }
            job.input = result.0; job.decision = result.1
            job.stamp = await context.commit(job.envelope, input: result.0, decision: result.1)
            guard job.stamp != nil else { throw PipelineError("STALE_CONTEXT", stage: "router") }
            active[job.token] = job; emit(.routed(job.envelope.requestId, result.1))
            switch result.1.route {
            case .eureka: await dispatchAPI(job, harness: false)
            case .harness:
                // Unsafe STT or missing slots never reach an adapter capable of external writes.
                if !result.1.missingRequiredSlots.isEmpty || job.envelope.payload.sttConfidence < config.sttConfidenceThreshold ||
                    !job.envelope.payload.sttConfidence.isFinite || !(0...1).contains(job.envelope.payload.sttConfidence) {
                    await ask(job, question: "대상과 필요한 내용을 다시 말씀해 주세요.")
                } else { await dispatchAPI(job, harness: true) }
            case .output:
                job.output = OutputRequest(requestId: job.envelope.requestId, requiredText: "안녕하세요. 무엇을 도와드릴까요?")
                await enqueue(job, into: outputQueue)
            case .reminder:
                if result.1.headOutputs.intention.value == .acknowledgeReminder { acknowledgeReminder() }
                else { snoozeReminder(seconds: 600) }
                job.output = OutputRequest(requestId: job.envelope.requestId, requiredText: "알림 요청을 반영했어요.")
                await enqueue(job, into: outputQueue)
            }
        } catch {
            if current(job) {
                // Preserve the final user utterance even when inference fails.
                let input = await context.prepare(job.envelope, resolution: .omitted)
                job.input = input; job.stamp = await context.commit(job.envelope, input: input, decision: nil)
                let result = ActionResult(operationId: job.operationId, status: .failed, action: .unclear,
                                          target: input.activeTarget, error: PipelineError("ROUTER_UNAVAILABLE", stage: "router"))
                storeAction(job, result)
                job.output = OutputRequest(requestId: job.envelope.requestId, action: result, requiredText: "요청을 해석하지 못했어요. 다시 말씀해 주세요.")
                await enqueue(job, into: outputQueue)
            } else { reject(job, error: error) }
        }
    }
    private func ask(_ original: Job, question: String) async {
        var job = original
        let result = ActionResult(operationId: job.operationId, status: .needsClarification,
                                  action: job.decision?.headOutputs.intention.value ?? .unclear, target: job.decision?.resolvedTarget)
        storeAction(job, result)
        if let stamp = job.stamp {
            await context.clarification(job.envelope.conversationId, requestId: job.envelope.requestId,
                                        slots: job.decision?.missingRequiredSlots ?? [], stamp: stamp)
        }
        job.output = OutputRequest(requestId: job.envelope.requestId, action: result, requiredText: question, isQuestion: true)
        await enqueue(job, into: outputQueue)
    }
    private func storeAction(_ job: Job, _ result: ActionResult) {
        records[job.envelope.requestId]?.action = result; emit(.action(job.envelope.requestId, result))
    }
    private func dispatchAPI(_ original: Job, harness: Bool) async {
        guard current(original) else { finish(original); return }
        let limit = harness ? config.maxConcurrentHarnessRequests : config.maxConcurrentEurekaRequests
        guard apiInFlight.values.filter({ $0 == harness }).count < limit else {
            // No pending API queue: overload is rejected before an external call is dispatched.
            var job = original
            let error = PipelineError("API_BUSY", stage: harness ? "harness" : "eureka")
            let result = ActionResult(operationId: job.operationId, status: .failed,
                                      action: job.decision?.headOutputs.intention.value ?? .unclear,
                                      target: job.decision?.resolvedTarget, error: error)
            storeAction(job, result); emit(.error(job.envelope.requestId, error))
            job.output = OutputRequest(requestId: job.envelope.requestId, action: result,
                                       requiredText: "동시에 처리 중인 요청이 많아요. 잠시 후 다시 말씀해 주세요.")
            await enqueue(job, into: outputQueue); return
        }
        apiInFlight[original.token] = harness
        do {
            if harness { try await harnessModule?.submit(original) }
            else { try await eurekaModule?.submit(original) }
        } catch {
            let fault = PipelineError("API_BUSY", stage: harness ? "harness" : "eureka")
            emit(.error(original.envelope.requestId, fault))
            let response = HarnessResult(action: ActionResult(operationId: original.operationId, status: .failed,
                action: original.decision?.headOutputs.intention.value ?? .unclear,
                target: original.decision?.resolvedTarget, error: fault))
            await apiCompleted(original, result: .success(response), harness: harness)
        }
    }
    private func callAPI(_ original: Job, eureka: (any EurekaAdapter)?, harness api: (any HarnessAdapter)?) async throws -> HarnessResult {
        let harness = api != nil
        guard current(original), let decision = original.decision, let input = original.input else { throw CancellationError() }
        let job = original
        let request = ActionRequest(operationId: job.operationId, requestId: job.envelope.requestId,
                                    decision: decision, deadlineAt: job.envelope.deadlineAt!)
        records[job.envelope.requestId]?.operation = request
        records[job.envelope.requestId]?.action = nil
        if harness {
            progressTimers[job.token] = Task { [weak self, config, job] in
                do { try await Task.sleep(for: .seconds(config.progressDelay)); await self?.progress(job) } catch {}
            }
        }
        let start = ContinuousClock.now
        let response: HarnessResult
        do {
            response = try await perform(token: job.token, timeout: timeout(job, harness ? config.harnessTimeout : config.eurekaTimeout), stage: harness ? "harness" : "eureka") { [job] in
                if let api {
                    return try await api.run(HarnessRequest(action: request, transcript: job.envelope.payload, context: input))
                }
                guard let eureka else { throw CancellationError() }
                let result = try await eureka.execute(request)
                return HarnessResult(action: result, question: result.facts["question"], resultText: result.facts["resultText"])
            }
        } catch {
            // Once dispatched, lost mutation replies and cancellation are not confirmed failure.
            let confirmedFailure = ["EUREKA_AUTH_ERROR", "EUREKA_NOT_FOUND", "EUREKA_CONFLICT", "MCP_AUTH_ERROR", "MCP_TOOL_UNAVAILABLE"].contains((error as? PipelineError)?.code ?? "")
            let status: ActionStatus = decision.headOutputs.intention.value.isMutation && !confirmedFailure ? .unknown : .failed
            response = HarnessResult(action: ActionResult(operationId: request.operationId, status: status,
                action: decision.headOutputs.intention.value, target: decision.resolvedTarget,
                error: PipelineError(status == .unknown ? (harness ? "MCP_RESULT_UNKNOWN" : "EUREKA_TIMEOUT_RESULT_UNKNOWN") : "ADAPTER_ERROR",
                                     stage: harness ? "harness" : "eureka")))
        }
        progressTimers.removeValue(forKey: job.token)?.cancel()
        await metrics.record(harness ? "harness" : "eureka", since: start)
        return response
    }
    private func apiCompleted(_ original: Job, result: Result<HarnessResult, any Error>, harness: Bool) async {
        defer { apiInFlight.removeValue(forKey: original.token) }
        var job = original
        guard let decision = job.decision else { finish(job); return }
        let request = ActionRequest(operationId: job.operationId, requestId: job.envelope.requestId,
                                    decision: decision, deadlineAt: job.envelope.deadlineAt!)
        let response: HarnessResult
        do { response = try result.get() }
        catch { reject(job, error: error); return }
        let targetMatches = decision.resolvedTarget.map { expected in
            response.action.target.map { expected.isSameEntity(as: $0) } ?? (response.action.status != .succeeded)
        } ?? true
        guard response.action.operationId == request.operationId,
              harness || (response.action.action == decision.headOutputs.intention.value && targetMatches) else {
            let unknown = ActionResult(operationId: request.operationId, status: decision.headOutputs.intention.value.isMutation ? .unknown : .failed,
                                       action: decision.headOutputs.intention.value, target: decision.resolvedTarget,
                                       error: PipelineError("CORRELATION_MISMATCH", stage: "action"))
            storeAction(job, unknown); reject(job, error: unknown.error!); return
        }
        guard response.action.facts.count <= 32,
              response.action.facts.reduce(0, { $0 + $1.key.utf8.count + $1.value.utf8.count }) <= config.maxOutputBytes / 2,
              (response.resultText?.utf8.count ?? 0) <= config.maxOutputBytes / 2,
              (response.question?.utf8.count ?? 0) <= config.maxOutputBytes / 2,
              (response.action.target?.name.utf8.count ?? 0) <= 1_024,
              (response.action.target?.entityId.utf8.count ?? 0) <= 256,
              (response.action.target?.kind.utf8.count ?? 0) <= 64,
              (response.action.error?.message.utf8.count ?? 0) <= 1_024,
              (response.action.error?.code.utf8.count ?? 0) <= 128,
              (response.action.error?.stage.utf8.count ?? 0) <= 128 else {
            // Preserve confirmed status, but never retain an oversized adapter payload.
            let fault = PipelineError("RESULT_TOO_LARGE", stage: "action")
            storeAction(job, ActionResult(operationId: response.action.operationId, status: response.action.status,
                                          action: response.action.action, error: fault))
            reject(job, error: fault); return
        }
        storeAction(job, response.action)
        guard current(job) else { finish(job); return }
        if response.action.status == .needsClarification { await ask(job, question: response.question ?? "어느 대상을 말씀하시나요?"); return }
        await context.clearPending(job.envelope.conversationId)
        let text = factualText(response.action, resultText: response.resultText)
        job.output = OutputRequest(requestId: job.envelope.requestId, action: response.action, requiredText: text)
        await enqueue(job, into: outputQueue)
    }
    private func factualText(_ action: ActionResult, resultText: String?) -> String {
        switch action.status {
        case .succeeded:
            let name = action.target?.name ?? "요청"
            switch action.action {
            case .postpone:
                if let value = action.facts["dueAt"], let date = ISO8601DateFormatter().date(from: value) {
                    return "\(name) 기한을 \(DateCodec.humanDateTime(date))로 변경했어요."
                }
                return "\(name) 변경은 완료됐지만 새 기한 정보는 확인되지 않았어요."
            case .complete: return "\(name) 업무 완료가 확인됐어요."
            case .register: return "\(name) 업무를 등록했어요."
            case .updatePriority:
                let labels = ["high": "높음", "normal": "보통", "low": "낮음"]
                if let value = action.facts["priority"] { return "\(name) 우선순위를 \(labels[value] ?? value)으로 변경했어요." }
                return "\(name) 우선순위를 변경했어요."
            case .query:
                if let resultText { return resultText }
                var text = action.facts["description"] ?? "\(name) 정보를 확인했어요."
                if let value = action.facts["dueAt"], let date = ISO8601DateFormatter().date(from: value) {
                    text += " 마감은 \(DateCodec.humanDateTime(date))이에요."
                }
                return text
            default: return resultText ?? "\(name) 처리 결과를 확인했어요."
            }
        case .failed: return "요청을 처리하지 못했어요. \(action.error?.code ?? "오류")"
        case .unknown: return "요청이 반영됐는지 아직 확인하지 못했어요. 다시 실행하지 않고 상태 확인이 필요해요."
        case .cancelled: return "요청 취소가 확인됐어요."
        case .needsClarification: return "필요한 내용을 다시 말씀해 주세요."
        }
    }
    private func progress(_ original: Job) async {
        guard current(original), progressTimers[original.token] != nil else { return }
        var job = original; job.terminal = false
        job.output = OutputRequest(requestId: job.envelope.requestId, requiredText: "아직 처리 중이에요")
        await enqueue(job, into: outputQueue)
    }
    private func generate(_ job: Job, model: any OutputModel) async throws -> String {
        guard current(job), let request = job.output else { throw CancellationError() }
        if !job.terminal, records[job.envelope.requestId]?.action != nil { throw CancellationError() }
        let start = ContinuousClock.now
        defer { Task { [metrics] in await metrics.record("output", since: start) } }
        return try await perform(token: job.terminal ? job.token : request.outputId, timeout: timeout(job, config.outputTimeout), stage: "output") {
            try await model.generate(request)
        }
    }
    private func generated(_ original: Job, result: Result<String, any Error>) async {
        guard let request = original.output else { if original.terminal { finish(original) }; return }
        var job = original
        var text = request.requiredText
        do {
            let candidate = try result.get()
            if !candidate.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
               candidate.utf8.count <= config.maxOutputBytes,
               (request.action == nil && request.reminder == nil && !request.isQuestion && job.terminal || candidate == request.requiredText) {
                text = candidate
            }
        } catch { emit(.error(job.envelope.requestId, PipelineError("OUTPUT_MODEL_ERROR", stage: "output"))) }
        guard current(job) else { if job.terminal { finish(job) }; return }
        guard text.utf8.count <= config.maxOutputBytes else { reject(job, error: PipelineError("OUTPUT_TOO_LARGE", stage: "output")); return }
        let status = request.action?.status
        job.event = OutputEvent(requestId: request.requestId, outputId: request.outputId, text: text,
            emotion: status == .succeeded ? "cheerful" : "neutral", activity: job.terminal ? .speaking : .processing,
            displayState: status?.rawValue ?? (job.reminder == nil ? "normal" : "reminder"), displayMode: .face,
            displayText: text, speech: SpeechPolicy(), motion: nil, actionStatus: status, reminderKey: request.reminder?.key)
        await fanOut(job)
    }
    private func fanOut(_ job: Job) async {
        guard current(job), let event = job.event else { if job.terminal { finish(job) }; return }
        if !job.terminal, records[job.envelope.requestId]?.action != nil { return }
        if let reminder = job.reminder {
            guard active.count == 1, waitingQuestion == nil, Date() >= quietUntil else { finish(job); return }
            do {
                let fresh = try await perform(token: job.token, timeout: timeout(job, config.eurekaTimeout), stage: "reminder.check") { [adapters] in
                    try await adapters.eureka.reminder(itemId: reminder.target.entityId)
                }
                guard let fresh, fresh.key == reminder.key, fresh.description == reminder.description, fresh.eligible,
                      fresh.dueAt > Date(), current(job) else { finish(job); return }
            } catch { reject(job, error: error); return }
        }
        deliveries[event.outputId] = DeliveryState(job: job)
        emit(.output(event))
        // Independent consumers. A web/display acknowledgement never gates TTS admission.
        do { try await displayQueue.offer(.output(job)) }
        catch { await displayCompleted(job, success: false, fallback: false) }
        if adapters.web != nil {
            do { try await webQueue.offer(job) }
            catch { emit(.error(event.requestId, PipelineError("QUEUE_FULL", stage: "web"))) }
        }
        do { try await ttsQueue.offer(job) }
        catch { await audioCompleted(job, success: false, error: PipelineError("QUEUE_FULL", stage: "tts")) }
    }
    private func publish(_ job: Job, web: any WebConsumer) async throws {
        guard running, let event = job.event else { throw CancellationError() }
        if !job.terminal, records[job.envelope.requestId]?.action != nil { return }
        try await perform(token: event.outputId, timeout: config.outputTimeout, stage: "web") { try await web.publish(event) }
    }
    private func displayCommand(_ command: DisplayCommand, display: any DisplayAdapter) async -> Bool {
        switch command {
        case .output(let job): return await show(job, display: display)
        case .idle:
            guard running, active.isEmpty else { return false }
            do {
                try await perform(token: idleDisplayToken, timeout: config.outputTimeout, stage: "display.idle") { try await display.idle() }
                return true
            } catch { emit(.error(nil, PipelineError("DISPLAY_ERROR", stage: "display.idle"))); return false }
        }
    }
    private let idleDisplayToken = UUID()
    private func show(_ job: Job, display: any DisplayAdapter) async -> Bool {
        guard current(job), let event = job.event else { return false }
        let fallback = event.displayMode == .text
        var success = false
        do {
            try await perform(token: job.executionToken, timeout: timeout(job, config.outputTimeout), stage: fallback ? "display.fallback" : "display") {
                try await display.show(event)
            }
            success = true
        } catch {}
        return success
    }
    private func synthesize(_ job: Job, model: any SpeechSynthesizer) async throws {
        guard current(job), let event = job.event else { throw CancellationError() }
        if !job.terminal, records[job.envelope.requestId]?.action != nil { return }
        let channel = PCMChannel(capacity: config.pcmQueueCapacity)
        pcmChannels[job.executionToken] = channel
        let deadline = min(job.envelope.deadlineAt!, Date().addingTimeInterval(config.playbackTimeout))
        do {
            try await perform(token: job.executionToken, timeout: deadline.timeIntervalSinceNow, stage: "tts") { [model, config, playbackQueue] in
                let first: (any AudioSource, PCMChunk) = try await withTimeout(config.ttsFirstChunkTimeout, stage: "tts.first_chunk") {
                    let source = try await model.audio(for: event)
                    let value = try await source.nextChunk()
                    guard let value else { throw PipelineError("TTS_ERROR", stage: "tts.empty") }
                    return (source, value)
                }
                func validate(_ chunk: PCMChunk) throws {
                    guard chunk.data.count <= config.maxChunkBytes, chunk.sampleRate > 0, chunk.channels > 0 else {
                        throw PipelineError("AUDIO_BUFFER_ERROR", stage: "tts")
                    }
                }
                try validate(first.1); try await channel.send(first.1)
                // One producer waits for playback capacity without discarding speech.
                try await playbackQueue.send(PlaybackJob(job: job, pcm: channel, deadline: deadline))
                while let chunk = try await first.0.nextChunk() {
                    try Task.checkCancellation(); try validate(chunk); try await channel.send(chunk)
                }
                await channel.finish()
            }
        } catch {
            await channel.finish(error: error)
            throw error
        }
    }
    private func play(_ audio: PlaybackJob, speaker: any SpeakerAdapter) async -> Bool {
        let job = audio.job
        guard current(job), let event = job.event else {
            await audio.pcm.finish(error: CancellationError()); pcmChannels.removeValue(forKey: job.executionToken); return false
        }
        if !job.terminal, records[job.envelope.requestId]?.action != nil {
            await audio.pcm.finish(error: CancellationError()); pcmChannels.removeValue(forKey: job.executionToken); return false
        }
        let jobId = UUID(); var success = false
        do {
            try await perform(token: job.executionToken, timeout: audio.deadline.timeIntervalSinceNow, stage: "speaker") {
                try await speaker.begin(jobId: jobId, event: event)
                while let chunk = try await audio.pcm.receive() {
                    try Task.checkCancellation(); try await speaker.write(chunk, jobId: jobId)
                }
                try await speaker.finish(jobId: jobId)
            }
            success = true
        } catch {
            await audio.pcm.finish(error: error); await speaker.stop(jobId: jobId)
        }
        pcmChannels.removeValue(forKey: job.executionToken)
        return success
    }
    private func audioCompleted(_ job: Job, success: Bool, error: PipelineError?) async {
        guard let event = job.event, var state = deliveries[event.outputId], state.audio == nil else { return }
        state.audio = success ? .completed : .failed
        if let error { state.errors.append(error) }
        if !success, current(job) { state.fallbackPending = true; state.display = nil }
        deliveries[event.outputId] = state
        if success { await storeDelivered(job, textFallback: false) }
        else if current(job) {
            var fallback = job; fallback.event?.displayMode = .text
            do { try await displayQueue.offer(.output(fallback)) }
            catch { await displayCompleted(fallback, success: false, fallback: true) }
        }
        await finalizeDelivery(event.outputId)
    }
    private func displayCompleted(_ job: Job, success: Bool, fallback: Bool) async {
        guard let event = job.event, var state = deliveries[event.outputId] else { return }
        if !success { state.errors.append(PipelineError("DISPLAY_ERROR", stage: fallback ? "display.fallback" : "display")) }
        if fallback { state.display = success ? .completed : .failed; state.fallbackPending = false }
        else if !state.fallbackPending { state.display = success ? .suppressed : .failed }
        deliveries[event.outputId] = state
        if fallback, success { await storeDelivered(job, textFallback: true) }
        await finalizeDelivery(event.outputId)
    }
    private func storeDelivered(_ job: Job, textFallback: Bool) async {
        guard current(job), job.terminal, let event = job.event,
              var state = deliveries[event.outputId], !state.contextStored else { return }
        state.contextStored = true; state.storingContext = true; deliveries[event.outputId] = state
        defer { deliveries[event.outputId]?.storingContext = false }
        await context.delivered(event, conversation: job.envelope.conversationId,
                                target: job.reminder?.target ?? job.decision?.resolvedTarget,
                                stamp: job.stamp, reminder: job.reminder != nil)
        guard current(job) else { return }
        if let reminder = job.reminder {
            reminderHistory[reminder.key] = (reminder.dueAt, true, nil); currentReminder = reminder
        }
        quietUntil = Date().addingTimeInterval(config.reminderQuietInterval + (textFallback ? config.textDisplayDuration : 0))
        if job.output?.isQuestion == true { beginQuestion(job, outputId: event.outputId) }
        if textFallback { beginDisplayTimer() }
    }
    private func finalizeDelivery(_ outputId: UUID) async {
        guard let state = deliveries[outputId], let audio = state.audio, let display = state.display,
              !state.fallbackPending, !state.storingContext else { return }
        deliveries.removeValue(forKey: outputId)
        let job = state.job
        let result = DeliveryResult(outputId: outputId, audio: audio, display: display, errors: state.errors)
        if job.terminal { records[job.envelope.requestId]?.delivery = result }
        emit(.delivery(result)); await metrics.record("delivery", since: state.started)
        if job.terminal {
            finish(job)
            if audio == .completed, job.output?.isQuestion != true {
                emit(.activity(job.envelope.conversationId, .idle))
            }
        }
    }
    private func beginQuestion(_ job: Job, outputId: UUID) {
        clearQuestion()
        waitingQuestion = (job.envelope.conversationId, job.envelope.requestId, outputId)
        emit(.activity(job.envelope.conversationId, .awaitingAnswer))
        questionTimer = Task { [weak self, config] in
            do { try await Task.sleep(for: .seconds(config.questionWait)); await self?.expireQuestion(outputId) } catch {}
        }
    }
    private func clearQuestion() { questionTimer?.cancel(); questionTimer = nil; waitingQuestion = nil }
    private func expireQuestion(_ output: UUID) async {
        guard let question = waitingQuestion, question.output == output else { return }
        emit(.questionWaitExpired(question.request, output)); clearQuestion()
        await context.clearPending(question.conversation); emit(.activity(question.conversation, .idle))
    }
    private func beginDisplayTimer() {
        displayTimer?.cancel()
        displayTimer = Task { [weak self, config] in
            do { try await Task.sleep(for: .seconds(config.textDisplayDuration)); await self?.displayIdle() } catch {}
        }
    }
    private func displayIdle() async {
        guard running, active.isEmpty else { return }
        do { try await displayQueue.offer(.idle) }
        catch { emit(.error(nil, PipelineError("QUEUE_FULL", stage: "display.idle"))) }
    }
    public func interact(_ event: InteractionEvent) async {
        switch event {
        case .speechStarted(let conversation):
            clearQuestion(); displayTimer?.cancel(); quietUntil = Date().addingTimeInterval(config.reminderQuietInterval)
            await cancelReminder(); emit(.activity(conversation, .listening))
        case .cancel(let conversation): await cancel(conversationId: conversation)
        case .acknowledgeReminder: acknowledgeReminder()
        case .snoozeReminder(let seconds): snoozeReminder(seconds: seconds)
        }
    }
    public func acknowledgeReminder() { currentReminder = nil }
    public func snoozeReminder(seconds: TimeInterval) {
        guard let item = currentReminder, seconds.isFinite, seconds > 0 else { return }
        reminderHistory[item.key] = (item.dueAt, true, Date().addingTimeInterval(seconds)); currentReminder = nil
    }
    private func cancelReminder() async {
        for job in active.values where job.reminder != nil { await cancelConversation(job.envelope.conversationId) }
    }
    public func pollReminders(conversationId: String = "deskpet") async {
        guard running, active.isEmpty, waitingQuestion == nil, Date() >= quietUntil,
              cancellation[pollToken] == nil else { return }
        do {
            let items = try await perform(token: pollToken, timeout: config.eurekaTimeout, stage: "reminder.poll") { [adapters] in try await adapters.eureka.reminders() }
            guard items.count <= config.maxReminderItems else { throw PipelineError("INPUT_TOO_LARGE", stage: "reminder.poll") }
            guard active.isEmpty, waitingQuestion == nil, running else { return }
            let now = Date()
            reminderHistory = reminderHistory.filter { $0.value.due > now }
            let candidates = items.filter { item in
                guard item.eligible, !item.description.isEmpty, item.description.utf8.count <= config.maxOutputBytes / 2,
                      item.dueAt > now, item.dueAt <= now.addingTimeInterval(86_400), TimeZone(identifier: item.timeZone) != nil else { return false }
                guard let history = reminderHistory[item.key] else { return true }
                return history.snooze.map { $0 <= now } ?? false
            }.sorted { $0.dueAt < $1.dueAt }
            guard let item = candidates.first, reminderHistory.count < config.maxReminderItems || reminderHistory[item.key] != nil else { return }
            let request = UUID(); try reserveRecord(request)
            var job = Job(envelope: Envelope(requestId: request, conversationId: conversationId,
                deadlineAt: now.addingTimeInterval(config.requestTimeout), kind: "reminder", payload: Transcript(text: "")))
            job.reminder = item
            job.output = OutputRequest(requestId: request, reminder: item,
                requiredText: "\(item.description) 마감은 \(DateCodec.reminderDeadline(item))이에요.")
            reminderHistory[item.key] = (item.dueAt, false, nil)
            active[job.token] = job; latest[conversationId] = job.token
            await enqueue(job, into: outputQueue)
        } catch { emit(.error(nil, error as? PipelineError ?? PipelineError("REMINDER_POLL_ERROR", stage: "reminder.poll"))) }
    }
    private let pollToken = UUID()
    // Reconciliation is read-only. The caller retains the ActionRequest / operation ID.
    public func reconcile(_ request: ActionRequest) async throws -> ActionResult {
        guard records[request.requestId]?.action?.status == .unknown,
              records[request.requestId]?.operation?.operationId == request.operationId else { throw PipelineError("NOT_UNKNOWN", stage: "reconcile") }
        let result = try await withTimeout(config.eurekaTimeout, stage: "reconcile") { [adapters] in
            if request.decision.route == .harness { return try await adapters.harness.reconcile(request) }
            return try await adapters.eureka.reconcile(request)
        }
        guard result.operationId == request.operationId else { throw PipelineError("CORRELATION_MISMATCH", stage: "reconcile") }
        guard result.facts.count <= 32,
              result.facts.reduce(0, { $0 + $1.key.utf8.count + $1.value.utf8.count }) <= config.maxOutputBytes / 2,
              (result.target?.name.utf8.count ?? 0) <= 1_024,
              (result.target?.entityId.utf8.count ?? 0) <= 256,
              (result.target?.kind.utf8.count ?? 0) <= 64,
              (result.error?.message.utf8.count ?? 0) <= 1_024,
              (result.error?.code.utf8.count ?? 0) <= 128,
              (result.error?.stage.utf8.count ?? 0) <= 128 else { throw PipelineError("RESULT_TOO_LARGE", stage: "reconcile") }
        records[request.requestId]?.action = result; emit(.action(request.requestId, result)); return result
    }
}
