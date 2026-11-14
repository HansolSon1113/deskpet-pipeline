import Foundation
import Testing
@testable import DeskPetPipeline

private actor SpyRouter: RouterModel {
    private var calls = 0
    func count() -> Int { calls }
    func resolveTarget(in transcript: Transcript) async throws -> TargetResolution {
        calls += 1; return .omitted
    }
    func heads(for input: RouterInput) async throws -> HeadOutputs {
        calls += 1; return HeadOutputs(intent: .chat)
    }
    func confidence(for maskedHeads: HeadOutputs, missingSlots: [String]) async throws -> Double {
        calls += 1; return 1
    }
}

private actor RerouteTalkHarness: HarnessAdapter {
    var calls = 0
    func run(_ request: HarnessRequest) async throws -> HarnessResult {
        calls += 1
        return HarnessResult(
            action: ActionResult(
                operationId: request.action.operationId,
                status: .succeeded,
                action: .unclear,
                facts: ["reroute": "talk"]
            ),
            resultText: "일상 대화로 이어갈게요."
        )
    }
    func count() -> Int { calls }
}

private actor RecordingChatOutputModel: OutputModel {
    var captured: [OutputRequest] = []
    let reply: String
    init(reply: String) { self.reply = reply }
    func generate(_ request: OutputRequest) async throws -> String {
        captured.append(request)
        if request.action == nil && request.reminder == nil && !request.isQuestion && !request.messages.isEmpty {
            return reply
        }
        return request.requiredText
    }
    func requests() -> [OutputRequest] { captured }
}

private actor InterruptedHarness: HarnessAdapter {
    func run(_ request: HarnessRequest) async throws -> HarnessResult {
        let json = #"{"timedOutIdle":true,"tasks":[{"goal":"PR 12 리뷰"}]}"#
        return HarnessResult(
            action: ActionResult(
                operationId: request.action.operationId,
                status: .succeeded,
                action: .harnessTask,
                facts: ["interrupted": json]
            ),
            resultText: nil
        )
    }
}

private actor DuplicateNoticeHarness: HarnessAdapter, HarnessNoticeSource {
    let notices: AsyncStream<Data>
    private let sink: AsyncStream<Data>.Continuation
    private var lastRequest: HarnessRequest?

    init() {
        let pair = AsyncStream<Data>.makeStream()
        notices = pair.stream
        sink = pair.continuation
    }

    func run(_ request: HarnessRequest) async throws -> HarnessResult {
        lastRequest = request
        return HarnessResult(
            action: ActionResult(operationId: request.action.operationId, status: .unknown, facts: ["inProgress": "true"]),
            resultText: "작업을 확인하고 있어요."
        )
    }

    func sendDuplicateNotice(id: String) throws {
        guard let req = lastRequest else { return }
        let data = try JSONSerialization.data(withJSONObject: [
            "type": "notice",
            "resultText": "지연 완료 결과입니다.",
            "notice": [
                "noticeId": id,
                "requestId": req.action.requestId.uuidString,
                "pipelineAction": ["status": "succeeded", "facts": ["done": "true"]]
            ]
        ])
        sink.yield(data)
    }
}

private func waitForIdleState(_ pipeline: DeskPetPipeline) async throws {
    let end = ContinuousClock.now.advanced(by: .seconds(3))
    while await pipeline.inFlightCount() > 0 {
        guard ContinuousClock.now < end else { throw PipelineError("TEST_TIMEOUT", stage: "test") }
        try await Task.sleep(for: .milliseconds(2))
    }
}

struct ExternalInjectionTests {
    @Test func lowSttConfidenceEmitsErrorAndNeverCallsRouterOrHarness() async throws {
        let router = SpyRouter()
        let eureka = MockEureka()
        let harness = MockHarness()
        var adapters = Adapters.mocks(eureka: eureka, harness: harness)
        adapters.router = router
        let pipeline = try DeskPetPipeline(adapters: adapters)
        await pipeline.start(automaticReminders: false)

        let collector: Task<[PipelineError], Never> = Task {
            var collected: [PipelineError] = []
            for await event in pipeline.events {
                if case .error(_, let error) = event {
                    collected.append(error)
                }
            }
            return collected
        }

        let id = try await pipeline.submit(Envelope(
            conversationId: "low-stt",
            kind: "stt.final",
            payload: Transcript(text: "어쩌고 저쩌고", sttConfidence: 0.2)
        ))
        try await waitForIdleState(pipeline)
        let record = await pipeline.record(requestId: id)
        await pipeline.shutdown()
        let errors = await collector.value

        #expect(errors.contains { $0.code == "STT_CONFIDENCE_LOW" })
        #expect(await router.count() == 0)
        #expect(await eureka.count() == 0)
        #expect(await harness.count() == 0)
        #expect(record?.action?.action != .unclear)
        #expect(record?.action?.error?.code == "STT_CONFIDENCE_LOW")
    }

    @Test func rerouteTalkEnqueuesChatOutputWithoutRoutingHintLeak() async throws {
        let router = MockRouter(scripts: [
            "오늘 날씨 어때": MockRoute(heads: HeadOutputs(intent: .unclear))
        ])
        let harness = RerouteTalkHarness()
        let outputModel = RecordingChatOutputModel(reply: "오늘은 맑고 선선해요.")
        let speaker = MockSpeaker()
        var adapters = Adapters.mocks(router: router, speaker: speaker)
        adapters.harness = harness
        adapters.output = outputModel
        let pipeline = try DeskPetPipeline(adapters: adapters)
        await pipeline.start(automaticReminders: false)

        _ = try await pipeline.submit(Envelope(
            conversationId: "reroute-talk",
            kind: "stt.final",
            payload: Transcript(text: "오늘 날씨 어때")
        ))
        try await waitForIdleState(pipeline)
        let spoken = await speaker.spoken()
        let captured = await outputModel.requests()
        await pipeline.shutdown()

        #expect(await harness.count() == 1)
        #expect(captured.count == 1)
        let req = try #require(captured.first)
        #expect(req.action == nil)
        #expect(req.reminder == nil)
        #expect(!req.isQuestion)
        #expect(req.messages.last?.content == "오늘 날씨 어때")
        #expect(spoken == ["오늘은 맑고 선선해요."])
        #expect(!spoken.joined().contains("일상 대화로 이어갈게요"))
    }

    @Test func interruptedTasksInjectedViaModuleReachSpeakerCleanly() async throws {
        let router = MockRouter(scripts: [
            "작업해줘": MockRoute(heads: HeadOutputs(intent: .harnessTask))
        ])
        let harness = InterruptedHarness()
        let speaker = MockSpeaker()
        var adapters = Adapters.mocks(router: router, speaker: speaker)
        adapters.harness = harness
        let pipeline = try DeskPetPipeline(adapters: adapters)
        await pipeline.start(automaticReminders: false)

        _ = try await pipeline.submit(Envelope(
            conversationId: "interrupted",
            kind: "stt.final",
            payload: Transcript(text: "작업해줘")
        ))
        try await waitForIdleState(pipeline)
        let spoken = await speaker.spoken()
        await pipeline.shutdown()

        #expect(spoken.count == 1)
        #expect(spoken.first?.contains("PR 12 리뷰") == true)
    }

    @Test func externalNoticeDeduplicatesAndRejectsStaleTurns() async throws {
        let harness = DuplicateNoticeHarness()
        let speaker = MockSpeaker()
        var adapters = Adapters.mocks(router: MockRouter(scripts: [
            "리뷰": MockRoute(heads: HeadOutputs(intent: .reviewPR))
        ]))
        adapters.harness = harness
        adapters.speaker = speaker

        let pipeline = try DeskPetPipeline(adapters: adapters)
        await pipeline.start(automaticReminders: false)

        let id = try await pipeline.submit(Envelope(conversationId: "dup-notice", kind: "stt.final", payload: Transcript(text: "리뷰")))
        try await waitForIdleState(pipeline)

        // Same noticeId yielded twice should only inject once into outputQueue.
        try await harness.sendDuplicateNotice(id: "notice-1")
        try await harness.sendDuplicateNotice(id: "notice-1")
        let end = ContinuousClock.now.advanced(by: .seconds(1))
        while await pipeline.record(requestId: id)?.action?.status != .succeeded, ContinuousClock.now < end {
            try await Task.sleep(for: .milliseconds(2))
        }
        try await waitForIdleState(pipeline)

        let spokenAfterNotice = await speaker.spoken()
        #expect(spokenAfterNotice.filter { $0 == "지연 완료 결과입니다." }.count == 1)

        // New user turn supersedes previous turn; delayed notice with another id on old requestId is rejected.
        _ = try await pipeline.submit(Envelope(conversationId: "dup-notice", kind: "stt.final", payload: Transcript(text: "안녕")))
        try await waitForIdleState(pipeline)
        try await harness.sendDuplicateNotice(id: "notice-2")
        try await Task.sleep(for: .milliseconds(30))
        try await waitForIdleState(pipeline)
        let finalSpoken = await speaker.spoken()
        #expect(finalSpoken.filter { $0 == "지연 완료 결과입니다." }.count == 1)

        await pipeline.shutdown()
    }

    @Test func routingHintTextGeneratedByOutputModelIsBlocked() async throws {
        let speaker = MockSpeaker()
        let output = MockOutput(replacement: "일상 대화로 이어갈게요.")
        let pipeline = try DeskPetPipeline(adapters: .mocks(output: output, speaker: speaker))
        await pipeline.start(automaticReminders: false)

        _ = try await pipeline.submit(Envelope(conversationId: "sanitize", kind: "stt.final", payload: Transcript(text: "안녕")))
        try await waitForIdleState(pipeline)
        let spoken = await speaker.spoken()
        await pipeline.shutdown()

        #expect(spoken == ["안녕하세요. 무엇을 도와드릴까요?"])
        #expect(!spoken.joined().contains("일상 대화로 이어갈게요"))
    }
}
