import Foundation
import Testing
@testable import DeskPetPipeline

private actor ConfirmEureka: EurekaAdapter {
    var calls: [ActionRequest] = []
    func execute(_ request: ActionRequest) async throws -> ActionResult {
        calls.append(request)
        if !request.mutationApproved {
            return ActionResult(
                operationId: request.operationId,
                status: .needsClarification,
                action: request.decision.headOutputs.intention.value,
                facts: ["questionKind": "confirmation", "question": "새 업무를 등록할까요?"]
            )
        }
        return ActionResult(
            operationId: request.operationId,
            status: .succeeded,
            action: request.decision.headOutputs.intention.value,
            facts: ["resultText": "'새 업무' 업무를 등록했어요."]
        )
    }
    func reconcile(_ request: ActionRequest) async throws -> ActionResult { ActionResult(operationId: request.operationId, status: .unknown) }
    func reminders() async throws -> [ReminderItem] { [] }
    func reminder(itemId: String) async throws -> ReminderItem? { nil }
}

private actor SlowAsyncHarness: HarnessAdapter {
    var calls: [HarnessRequest] = []
    func run(_ request: HarnessRequest) async throws -> HarnessResult {
        calls.append(request)
        try await Task.sleep(for: .milliseconds(80))
        return HarnessResult(
            action: ActionResult(operationId: request.action.operationId, status: .succeeded, action: .harnessTask),
            resultText: "백그라운드 작업이 끝났어요."
        )
    }
}

private actor CueRecorder: SoundCuePlayer {
    var cues: [SoundCue] = []
    func play(_ cue: SoundCue) async { cues.append(cue) }
}

struct RuleBypassAndAsyncTests {
    @Test func ruleBypassMatchesConcreteScenarios() {
        #expect(RuleBypass.match("오늘 할일 알려줘") == .eurekaFilter("today"))
        #expect(RuleBypass.match("마감 지난 업무 뭐 있어") == .eurekaFilter("overdue"))
        #expect(RuleBypass.match("급한 일 보여줘") == .eurekaFilter("urgent"))
        #expect(RuleBypass.match("오늘 브리핑 해줘") == .eurekaFilter("briefing"))
        #expect(RuleBypass.match("타이머 3분 맞춰줘") == .setTimer(seconds: 180, label: "3분"))
        #expect(RuleBypass.match("3분 타이머 맞춰줘") == nil)
        #expect(RuleBypass.match("타이머 삼분") == .setTimer(seconds: 180, label: "3분"))
        #expect(RuleBypass.match("타이머 오 분 맞춰줘") == .setTimer(seconds: 300, label: "5분"))
        #expect(RuleBypass.match("타이머 십오분") == .setTimer(seconds: 900, label: "15분"))
        #expect(RuleBypass.match("타이머 일분 삼십초") == .setTimer(seconds: 90, label: "1분 30초"))
        #expect(RuleBypass.match("타이머 이십 오 초") == .setTimer(seconds: 25, label: "25초"))
        #expect(RuleBypass.match("타이머 한 시간 반") == .setTimer(seconds: 5400, label: "1시간 30분"))
        #expect(RuleBypass.match("타이머 일곱 분") == .setTimer(seconds: 420, label: "7분"))
        #expect(RuleBypass.match("타임머 삼분") == .setTimer(seconds: 180, label: "3분"))
        #expect(RuleBypass.match("타이머 남은 시간") == .queryTimer)
        #expect(RuleBypass.match("타이머 얼마 남았어") == .queryTimer)
        #expect(RuleBypass.match("타임머 얼마 남았어") == .queryTimer)
        #expect(RuleBypass.match("타이머 언제 끝나") == .queryTimer)
        #expect(RuleBypass.match("타이머 취소해줘") == .cancelTimer)
        #expect(RuleBypass.match("타임머 취소해줘") == .cancelTimer)
        #expect(RuleBypass.match("알림 꺼줘") == .acknowledgeReminder)
        #expect(RuleBypass.match("그만 말해") == .stopSpeaking)
        #expect(RuleBypass.match("다시 말해줘") == .repeatLastReply)
        #expect(RuleBypass.confirmationAnswer("응 등록해줘") == true)
        #expect(RuleBypass.confirmationAnswer("아니 취소해") == false)
    }

    @Test func eurekaConfirmationAndFilterBypassWorkInPipeline() async throws {
        let eureka = ConfirmEureka()
        let speaker = MockSpeaker()
        var adapters = Adapters.mocks(
            router: MockRouter(scripts: [
                "새 업무 등록해줘": MockRoute(heads: HeadOutputs(intent: .create, task: "새 업무"))
            ])
        )
        adapters.eureka = eureka
        adapters.speaker = speaker
        var config = PipelineConfiguration()
        config.ruleBypassEnabled = true
        let pipeline = try DeskPetPipeline(adapters: adapters, configuration: config)
        await pipeline.start(automaticReminders: false)

        _ = try await pipeline.submit(Envelope(conversationId: "c1", kind: "stt.final", payload: Transcript(text: "오늘 할 일")))
        while await pipeline.inFlightCount() > 0 { try await Task.sleep(for: .milliseconds(5)) }
        #expect(await eureka.calls.last?.decision.headOutputs.filter?.value == "today")

        _ = try await pipeline.submit(Envelope(conversationId: "c1", kind: "stt.final", payload: Transcript(text: "새 업무 등록해줘")))
        while await pipeline.inFlightCount() > 0 { try await Task.sleep(for: .milliseconds(5)) }
        #expect(await speaker.spoken().last == "새 업무를 등록할까요?")

        _ = try await pipeline.submit(Envelope(conversationId: "c1", kind: "stt.final", payload: Transcript(text: "응")))
        while await pipeline.inFlightCount() > 0 { try await Task.sleep(for: .milliseconds(5)) }
        #expect(await eureka.calls.last?.mutationApproved == true)
        #expect(await speaker.spoken().last == "'새 업무' 업무를 등록했어요.")
        await pipeline.shutdown()
    }

    @Test func asyncHarnessAckSpeaksImmediatelyAndCompletesAfterSubsequentTurn() async throws {
        let harness = SlowAsyncHarness()
        let speaker = MockSpeaker()
        var adapters = Adapters.mocks(
            router: MockRouter(scripts: [
                "코드 분석해줘": MockRoute(heads: HeadOutputs(intent: .harnessTask)),
                "안녕": MockRoute(heads: HeadOutputs(intent: .chat))
            ])
        )
        adapters.harness = harness
        adapters.speaker = speaker
        adapters.output = MockOutput(replacement: "안녕하세요.")
        var config = PipelineConfiguration()
        config.asyncHarnessAck = true
        let pipeline = try DeskPetPipeline(adapters: adapters, configuration: config)
        await pipeline.start(automaticReminders: false)

        _ = try await pipeline.submit(Envelope(conversationId: "c2", kind: "stt.final", payload: Transcript(text: "코드 분석해줘")))
        try await Task.sleep(for: .milliseconds(25))
        _ = try await pipeline.submit(Envelope(conversationId: "c2", kind: "stt.final", payload: Transcript(text: "안녕")))
        try await Task.sleep(for: .milliseconds(150))

        let spoken = await speaker.spoken()
        #expect(spoken.contains("확인해 볼게요."))
        #expect(spoken.contains("백그라운드 작업이 끝났어요."))
        await pipeline.shutdown()
    }

    @Test func outputContextIncludesLocalTimeAndSoundCueRecordsStart() async throws {
        let ctx = OutputContextProvider()
        let line = await ctx.currentContextLine(now: Date(timeIntervalSince1970: 1_700_000_000))
        #expect(line.contains("현재 시각:"))

        let cue = CueRecorder()
        var adapters = Adapters.mocks()
        adapters.soundCue = cue
        let pipeline = try DeskPetPipeline(adapters: adapters)
        await pipeline.start(automaticReminders: false)
        await pipeline.interact(.speechStarted("c3"))
        try await Task.sleep(for: .milliseconds(20))
        #expect(await cue.cues == [.listenStart])
        await pipeline.shutdown()
    }

    @Test func nativeRouterContractNormalizesKoreanDueDates() {
        var cal = Calendar(identifier: .gregorian)
        let seoul = TimeZone(identifier: "Asia/Seoul")!
        cal.timeZone = seoul
        // 2026-11-25 16:44 KST (Wednesday)
        let ref = cal.date(from: DateComponents(timeZone: seoul, year: 2026, month: 11, day: 25, hour: 16, minute: 44))!

        func ymdhm(_ date: Date?) -> [Int]? {
            guard let date else { return nil }
            let c = cal.dateComponents([.year, .month, .day, .hour, .minute], from: date)
            return [c.year!, c.month!, c.day!, c.hour!, c.minute!]
        }

        #expect(ymdhm(NativeRouterContract.date("오늘로 설정", relativeTo: ref)) == [2026, 11, 25, 23, 59])
        #expect(ymdhm(NativeRouterContract.date("내일 오후 3시까지", relativeTo: ref)) == [2026, 11, 26, 15, 0])
        #expect(ymdhm(NativeRouterContract.date("모레", relativeTo: ref)) == [2026, 11, 27, 23, 59])
        #expect(ymdhm(NativeRouterContract.date("글피", relativeTo: ref)) == [2026, 11, 28, 23, 59])
        #expect(ymdhm(NativeRouterContract.date("3일 뒤", relativeTo: ref)) == [2026, 11, 28, 23, 59])
        #expect(ymdhm(NativeRouterContract.date("다음 주 금요일", relativeTo: ref)) == [2026, 12, 4, 23, 59])
        #expect(ymdhm(NativeRouterContract.date("12월 5일까지", relativeTo: ref)) == [2026, 12, 5, 23, 59])
        #expect(ymdhm(NativeRouterContract.date("오후 5시 반까지", relativeTo: ref)) == [2026, 11, 25, 17, 30])
        #expect(NativeRouterContract.date("다음에 시간 날 때", relativeTo: ref) == nil)
    }
}
