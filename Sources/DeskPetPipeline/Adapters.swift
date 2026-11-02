import Foundation

// All adapters must cooperate with Task cancellation. Blocking native inference belongs
// on a dedicated executor/process with a bounded buffer and explicit interrupt support.
public protocol SpeechRecognizer: Sendable { func transcribe(_ segment: AudioSegment) async throws -> Transcript }
public struct STTFrame: Sendable {
    public let sessionId: UUID
    public let frame: AudioFrame
    public let activity: VoiceActivity
    public init(sessionId: UUID, frame: AudioFrame, activity: VoiceActivity) {
        self.sessionId = sessionId; self.frame = frame; self.activity = activity
    }
}
public protocol StreamingSpeechRecognizer: SpeechRecognizer {
    func accept(_ frame: STTFrame) async throws -> Transcript?
    func cancel(sessionId: UUID) async
}
public protocol RouterModel: Sendable {
    func resolveTarget(in transcript: Transcript) async throws -> TargetResolution
    func heads(for input: RouterInput) async throws -> HeadOutputs
    func confidence(for maskedHeads: HeadOutputs, missingSlots: [String]) async throws -> Double
}
public protocol EurekaAdapter: Sendable {
    func execute(_ request: ActionRequest) async throws -> ActionResult
    func reconcile(_ request: ActionRequest) async throws -> ActionResult
    func reminders() async throws -> [ReminderItem]
    func reminder(itemId: String) async throws -> ReminderItem?
}
public protocol HarnessAdapter: Sendable {
    func run(_ request: HarnessRequest) async throws -> HarnessResult
    func reconcile(_ request: ActionRequest) async throws -> ActionResult
}
public extension HarnessAdapter {
    func reconcile(_ request: ActionRequest) async throws -> ActionResult {
        throw PipelineError("RECONCILIATION_UNAVAILABLE", stage: "harness")
    }
}
public protocol OutputModel: Sendable { func generate(_ request: OutputRequest) async throws -> String }
public protocol EmotionClassifier: Sendable { func emotion(for text: String) async throws -> String }
public struct HarnessPipelineEvent: Sendable {
    public var requestId: UUID
    public var kind: String
    public var pendingId: String?
    public var outputId: String?
    public var deliveredText: String?
    public var channel: String?
    public init(requestId: UUID, kind: String, pendingId: String? = nil, outputId: String? = nil,
                deliveredText: String? = nil, channel: String? = nil) {
        self.requestId = requestId; self.kind = kind; self.pendingId = pendingId; self.outputId = outputId
        self.deliveredText = deliveredText; self.channel = channel
    }
}
public protocol HarnessEventConsumer: Sendable { func report(_ event: HarnessPipelineEvent) async throws }
public protocol HarnessNoticeSource: Sendable { var notices: AsyncStream<Data> { get async } }
// Pulling one chunk at a time bounds PCM memory. finish returns only after physical playback.
public protocol AudioSource: Sendable { func nextChunk() async throws -> PCMChunk? }
public protocol SpeechSynthesizer: Sendable { func audio(for event: OutputEvent) async throws -> any AudioSource }
public protocol SpeakerAdapter: Sendable {
    func begin(jobId: UUID, event: OutputEvent) async throws
    func write(_ chunk: PCMChunk, jobId: UUID) async throws
    func finish(jobId: UUID) async throws
    func stop(jobId: UUID) async
}
public protocol DisplayAdapter: Sendable {
    // Returns only after displayed acknowledgement, not command acceptance.
    func show(_ event: OutputEvent) async throws
    func idle() async throws
}
public protocol WebConsumer: Sendable { func publish(_ event: OutputEvent) async throws }
public protocol RemoteRouter: Sendable { func heads(for input: RouterInput, deadlineAt: Date) async throws -> HeadOutputs }
public struct Adapters: Sendable {
    public var stt: any SpeechRecognizer
    public var router: (any RouterModel)?
    public var eureka: any EurekaAdapter
    public var harness: any HarnessAdapter
    public var output: any OutputModel
    public var tts: any SpeechSynthesizer
    public var speaker: any SpeakerAdapter
    public var display: any DisplayAdapter
    public var web: (any WebConsumer)?
    public var remoteRouter: (any RemoteRouter)?
    public var gpuExecutor: GPUExecutor?
    public var emotion: (any EmotionClassifier)?
    public var inputEpochValidator: (any AudioInputEpochValidator)?
    public var taskRefs: [HarnessTaskRef] = []
    public init(stt: any SpeechRecognizer, router: (any RouterModel)? = nil, eureka: any EurekaAdapter,
                harness: any HarnessAdapter, output: any OutputModel, tts: any SpeechSynthesizer,
                speaker: any SpeakerAdapter, display: any DisplayAdapter,
                web: (any WebConsumer)? = nil, remoteRouter: (any RemoteRouter)? = nil,
                gpuExecutor: GPUExecutor? = nil) {
        self.stt = stt; self.router = router; self.eureka = eureka; self.harness = harness
        self.output = output; self.tts = tts; self.speaker = speaker; self.display = display
        self.web = web; self.remoteRouter = remoteRouter; self.gpuExecutor = gpuExecutor
    }
}
