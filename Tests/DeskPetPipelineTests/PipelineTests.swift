import Foundation
import Testing
@testable import DeskPetPipeline

private let report = Target(entityId: "report", name: "보고서 제출")
private let meeting = Target(entityId: "meeting", name: "회의 준비")
private func item(_ target: Target = report, due: Date = Date().addingTimeInterval(3_600)) -> ReminderItem {
    ReminderItem(target: target, description: "분석 결과를 비교하고 보고서를 제출해야 해요.", dueAt: due)
}
private func fixtureRouter() -> MockRouter {
    MockRouter(scripts: [
        "보고서 완료": MockRoute(heads: HeadOutputs(intent: .complete, name: report.name), resolution: .unique(report)),
        "회의 완료": MockRoute(heads: HeadOutputs(intent: .complete, name: meeting.name), resolution: .unique(meeting)),
        "기한 변경": MockRoute(heads: HeadOutputs(intent: .postpone, name: report.name, dueAt: Date().addingTimeInterval(7_200)), resolution: .unique(report)),
        "그거 완료": MockRoute(heads: HeadOutputs(intent: .complete)),
        "기한은 내일": MockRoute(heads: HeadOutputs(intent: .postpone, dueAt: Date().addingTimeInterval(86_400))),
        "높여줘": MockRoute(heads: HeadOutputs(intent: .updatePriority, priority: "high")),
        "모호": MockRoute(heads: HeadOutputs(intent: .complete)),
        "동명이인": MockRoute(heads: HeadOutputs(intent: .complete, name: "보고서"), resolution: .ambiguous),
        "외부 조회": MockRoute(heads: HeadOutputs(intent: .mcp, task: "read only fixture")),
        "낮은 확신": MockRoute(heads: HeadOutputs(intent: .complete, name: report.name, confidence: 0.2), resolution: .unique(report))
    ])
}
private func send(_ pipeline: DeskPetPipeline, _ text: String, confidence: Double = 1, conversation: String = "deskpet") async throws -> UUID {
    try await pipeline.submit(Envelope(conversationId: conversation, kind: "stt.final", payload: Transcript(text: text, sttConfidence: confidence)))
}
private func idle(_ pipeline: DeskPetPipeline) async throws {
    let end = ContinuousClock.now.advanced(by: .seconds(3))
    while await pipeline.inFlightCount() > 0 {
        guard ContinuousClock.now < end else { throw PipelineError("TEST_TIMEOUT", stage: "test") }
        try await Task.sleep(for: .milliseconds(2))
    }
}
private func started(_ eureka: MockEureka) async throws {
    let end = ContinuousClock.now.advanced(by: .seconds(1))
    while await eureka.count() == 0 {
        guard ContinuousClock.now < end else { throw PipelineError("TEST_TIMEOUT", stage: "test") }
        try await Task.sleep(for: .milliseconds(2))
    }
}

struct PipelineTests {
    @Test func successAndSameTargetContext() async throws {
        let eureka = MockEureka(items: [item(), item(meeting)])
        let pipeline = try DeskPetPipeline(adapters: .mocks(router: fixtureRouter(), eureka: eureka))
        await pipeline.start(automaticReminders: false)
        for text in ["기한 변경", "기한은 내일", "높여줘"] { _ = try await send(pipeline, text); try await idle(pipeline) }
        #expect(await eureka.count() == 3)
        let turns = await pipeline.contextTurns(conversationId: "deskpet")
        #expect(turns.filter { $0.role == "user" }.count == 3)
        #expect(turns.filter { $0.role == "assistant" }.count == 3)
        await pipeline.shutdown()
    }
    @Test func targetSwitchDiscardsPriorContext() async throws {
        let eureka = MockEureka(items: [item(), item(meeting)])
        let pipeline = try DeskPetPipeline(adapters: .mocks(router: fixtureRouter(), eureka: eureka))
        await pipeline.start(automaticReminders: false)
        _ = try await send(pipeline, "보고서 완료"); try await idle(pipeline)
        _ = try await send(pipeline, "회의 완료"); try await idle(pipeline)
        let turns = await pipeline.contextTurns(conversationId: "deskpet")
        #expect(turns.count == 2)
        #expect(turns.first?.content == "회의 완료")
        #expect(!turns.contains { $0.content.contains("보고서") })
        await pipeline.shutdown()
    }
    @Test func partialAndEmptyNeverExecute() async throws {
        let eureka = MockEureka(items: [item()])
        let pipeline = try DeskPetPipeline(adapters: .mocks(router: fixtureRouter(), eureka: eureka))
        await pipeline.start(automaticReminders: false)
        _ = try await pipeline.submit(Envelope(conversationId: "deskpet", kind: "stt.partial", payload: Transcript(text: "보고서 완료", isFinal: false)))
        await #expect(throws: PipelineError.self) { _ = try await send(pipeline, "  ") }
        #expect(await eureka.count() == 0)
        #expect(await pipeline.contextTurns(conversationId: "deskpet").isEmpty)
        await pipeline.shutdown()
    }
    @Test func sttConfidenceIsIndependentAndPreventsWrites() async throws {
        let eureka = MockEureka(items: [item()]); let harness = MockHarness()
        let pipeline = try DeskPetPipeline(adapters: .mocks(router: fixtureRouter(), eureka: eureka, harness: harness))
        await pipeline.start(automaticReminders: false)
        let request = try await send(pipeline, "보고서 완료", confidence: 0.1); try await idle(pipeline)
        #expect(await eureka.count() == 0); #expect(await harness.count() == 0)
        #expect(await pipeline.record(requestId: request)?.action?.status == .needsClarification)
        await pipeline.shutdown()
    }
    @Test func ambiguityAndMissingSlotsDoNotInvokeExternalAdapters() async throws {
        let eureka = MockEureka(items: [item()]); let harness = MockHarness()
        let pipeline = try DeskPetPipeline(adapters: .mocks(router: fixtureRouter(), eureka: eureka, harness: harness))
        await pipeline.start(automaticReminders: false)
        _ = try await send(pipeline, "동명이인"); try await idle(pipeline)
        #expect(await eureka.count() == 0); #expect(await harness.count() == 0)
        await pipeline.shutdown()
    }
    @Test func clarificationFollowupKeepsRequestId() async throws {
        let eureka = MockEureka(items: [item()])
        let pipeline = try DeskPetPipeline(adapters: .mocks(router: fixtureRouter(), eureka: eureka))
        await pipeline.start(automaticReminders: false)
        let question = try await send(pipeline, "모호"); try await idle(pipeline)
        let answer = try await send(pipeline, "보고서 완료"); try await idle(pipeline)
        #expect(answer == question)
        #expect(await pipeline.record(requestId: answer)?.action?.status == .succeeded)
        await pipeline.shutdown()
    }
    @Test func chatBypassesEurekaAndHarness() async throws {
        let eureka = MockEureka(); let harness = MockHarness()
        let pipeline = try DeskPetPipeline(adapters: .mocks(eureka: eureka, harness: harness))
        await pipeline.start(automaticReminders: false)
        _ = try await send(pipeline, "안녕"); try await idle(pipeline)
        #expect(await eureka.count() == 0); #expect(await harness.count() == 0)
        await pipeline.shutdown()
    }
    @Test func mcpUsesHarness() async throws {
        let eureka = MockEureka(); let harness = MockHarness()
        let pipeline = try DeskPetPipeline(adapters: .mocks(router: fixtureRouter(), eureka: eureka, harness: harness))
        await pipeline.start(automaticReminders: false)
        let id = try await send(pipeline, "외부 조회"); try await idle(pipeline)
        #expect(await harness.count() == 1); #expect(await eureka.count() == 0)
        #expect(await pipeline.record(requestId: id)?.action?.status == .succeeded)
        await pipeline.shutdown()
    }
    @Test func lostMutationReplyIsUnknownAndNotRetried() async throws {
        let eureka = MockEureka(items: [item()], loseMutationReply: true)
        let speaker = MockSpeaker()
        let pipeline = try DeskPetPipeline(adapters: .mocks(router: fixtureRouter(), eureka: eureka, speaker: speaker))
        await pipeline.start(automaticReminders: false)
        let id = try await send(pipeline, "기한 변경"); try await idle(pipeline)
        #expect(await eureka.count() == 1)
        let record = await pipeline.record(requestId: id)
        #expect(record?.action?.status == .unknown)
        #expect(await speaker.spoken().first?.contains("확인하지 못") == true)
        let decision = RouteDecision(headOutputs: HeadOutputs(intent: .postpone), overallConfidence: 1,
                                     missingRequiredSlots: [], route: .eureka, sourceTurnIds: [], resolvedTarget: report)
        let result = try await pipeline.reconcile(ActionRequest(operationId: record!.action!.operationId, requestId: id,
                                                               decision: decision, deadlineAt: Date().addingTimeInterval(1)))
        #expect(result.status == .succeeded); #expect(await eureka.count() == 1)
        await pipeline.shutdown()
    }
    @Test func writeTimeoutIsUnknown() async throws {
        var config = PipelineConfiguration(); config.eurekaTimeout = 0.01
        let eureka = MockEureka(items: [item()], behavior: .init(delay: 1))
        let pipeline = try DeskPetPipeline(adapters: .mocks(router: fixtureRouter(), eureka: eureka), configuration: config)
        await pipeline.start(automaticReminders: false)
        let id = try await send(pipeline, "보고서 완료"); try await idle(pipeline)
        #expect(await pipeline.record(requestId: id)?.action?.status == .unknown)
        await pipeline.shutdown()
    }
    @Test func authFailureIsConfirmedFailure() async throws {
        let eureka = MockEureka(items: [item()], behavior: .init(error: PipelineError("EUREKA_AUTH_ERROR", stage: "eureka")))
        let pipeline = try DeskPetPipeline(adapters: .mocks(router: fixtureRouter(), eureka: eureka))
        await pipeline.start(automaticReminders: false)
        let id = try await send(pipeline, "보고서 완료"); try await idle(pipeline)
        #expect(await pipeline.record(requestId: id)?.action?.status == .failed)
        await pipeline.shutdown()
    }
    @Test func ttsFailureUsesLCDWithoutRepeatingWork() async throws {
        let eureka = MockEureka(items: [item()]); let display = MockDisplay()
        let pipeline = try DeskPetPipeline(adapters: .mocks(router: fixtureRouter(), eureka: eureka,
            tts: MockTTS(behavior: .init(error: PipelineError("TTS_ERROR", stage: "tts"))), display: display))
        await pipeline.start(automaticReminders: false)
        let id = try await send(pipeline, "보고서 완료"); try await idle(pipeline)
        #expect(await eureka.count() == 1)
        let record = await pipeline.record(requestId: id)
        #expect(record?.action?.status == .succeeded)
        #expect(record?.delivery?.audio == .failed); #expect(record?.delivery?.display == .completed)
        #expect(await display.shown().last?.displayMode == .text)
        #expect(await pipeline.contextTurns(conversationId: "deskpet").count == 2)
        await pipeline.shutdown()
    }
    @Test func bothOutputFailuresNeverAddAssistantTurn() async throws {
        let failure = MockBehavior(error: PipelineError("MOCK_FAILURE", stage: "mock"))
        let pipeline = try DeskPetPipeline(adapters: .mocks(router: fixtureRouter(), eureka: MockEureka(items: [item()]),
            tts: MockTTS(behavior: failure), display: MockDisplay(behavior: failure)))
        await pipeline.start(automaticReminders: false)
        _ = try await send(pipeline, "보고서 완료"); try await idle(pipeline)
        #expect(await pipeline.contextTurns(conversationId: "deskpet").count == 1)
        await pipeline.shutdown()
    }
    @Test func outputModelCannotChangeFacts() async throws {
        let speaker = MockSpeaker()
        let pipeline = try DeskPetPipeline(adapters: .mocks(router: fixtureRouter(), eureka: MockEureka(items: [item()]),
            output: MockOutput(replacement: "모든 업무가 삭제됐어요"), speaker: speaker))
        await pipeline.start(automaticReminders: false)
        _ = try await send(pipeline, "기한 변경"); try await idle(pipeline)
        #expect(await speaker.spoken().first?.contains("기한을") == true)
        #expect(await speaker.spoken().first?.contains("삭제") == false)
        await pipeline.shutdown()
    }
    @Test func outputTimeoutFallsBack() async throws {
        var config = PipelineConfiguration(); config.outputTimeout = 0.01
        let speaker = MockSpeaker()
        let pipeline = try DeskPetPipeline(adapters: .mocks(output: MockOutput(behavior: .init(delay: 1)), speaker: speaker), configuration: config)
        await pipeline.start(automaticReminders: false)
        _ = try await send(pipeline, "안녕"); try await idle(pipeline)
        #expect(await speaker.spoken().count == 1)
        await pipeline.shutdown()
    }
    @Test func reminderIsSingleDeduplicatedAndBecomesContext() async throws {
        var config = PipelineConfiguration(); config.reminderQuietInterval = 0
        let speaker = MockSpeaker(); let eureka = MockEureka(items: [item()])
        let pipeline = try DeskPetPipeline(adapters: .mocks(router: fixtureRouter(), eureka: eureka, speaker: speaker), configuration: config)
        await pipeline.start(automaticReminders: false)
        await pipeline.pollReminders(); try await idle(pipeline)
        await pipeline.pollReminders(); try await idle(pipeline)
        let spoken = await speaker.spoken()
        #expect(spoken.count == 1); #expect(spoken[0].contains("Asia/Seoul")); #expect(spoken[0].contains(item().description))
        let turns = await pipeline.contextTurns(conversationId: "deskpet")
        #expect(turns.first?.role == "assistant"); #expect(turns.first?.content == spoken.first)
        let id = try await send(pipeline, "그거 완료"); try await idle(pipeline)
        #expect(await pipeline.record(requestId: id)?.action?.target?.entityId == report.entityId)
        await pipeline.shutdown()
    }
    @Test func reminderDefersDuringQuestionAndSnoozeDoesNotChangeDueDate() async throws {
        var config = PipelineConfiguration(); config.reminderQuietInterval = 0
        let original = item(); let speaker = MockSpeaker(); let eureka = MockEureka(items: [original])
        let pipeline = try DeskPetPipeline(adapters: .mocks(router: fixtureRouter(), eureka: eureka, speaker: speaker), configuration: config)
        await pipeline.start(automaticReminders: false)
        await pipeline.pollReminders(); try await idle(pipeline)
        await pipeline.snoozeReminder(seconds: 600)
        await pipeline.pollReminders(); try await idle(pipeline)
        #expect(await speaker.spoken().count == 1)
        #expect(try await eureka.reminder(itemId: report.entityId)?.dueAt == original.dueAt)
        _ = try await send(pipeline, "동명이인"); try await idle(pipeline)
        await pipeline.pollReminders(); try await idle(pipeline)
        #expect(await speaker.spoken().count == 2)
        await pipeline.shutdown()
    }
    @Test func completedOverdueOrIncompleteRemindersAreIgnored() async throws {
        var done = item(); done.eligible = false
        var empty = item(meeting); empty.description = ""
        let speaker = MockSpeaker()
        let pipeline = try DeskPetPipeline(adapters: .mocks(eureka: MockEureka(items: [done, empty,
            item(Target(entityId: "overdue", name: "overdue"), due: Date().addingTimeInterval(-1))]), speaker: speaker))
        await pipeline.start(automaticReminders: false); await pipeline.pollReminders(); try await idle(pipeline)
        #expect(await speaker.spoken().isEmpty)
        await pipeline.shutdown()
    }
    @Test func expiredOversizeAndDuplicateRequestsAreRejected() async throws {
        let pipeline = try DeskPetPipeline(adapters: .mocks(router: fixtureRouter(), eureka: MockEureka(items: [item()])))
        await pipeline.start(automaticReminders: false)
        await #expect(throws: PipelineError.self) {
            _ = try await pipeline.submit(Envelope(conversationId: "deskpet", deadlineAt: Date().addingTimeInterval(-1), kind: "stt.final", payload: Transcript(text: "보고서 완료")))
        }
        await #expect(throws: PipelineError.self) { _ = try await send(pipeline, String(repeating: "x", count: 4_097)) }
        let envelope = Envelope(conversationId: "deskpet", kind: "stt.final", payload: Transcript(text: "보고서 완료"))
        _ = try await pipeline.submit(envelope); try await idle(pipeline)
        await #expect(throws: PipelineError.self) { _ = try await pipeline.submit(envelope) }
        await pipeline.shutdown()
    }
    @Test func boundedMailboxRejectsOverflowAndRetainsOrder() async throws {
        let mailbox = Mailbox<Int>(capacity: 2)
        try await mailbox.offer(1); try await mailbox.offer(2)
        await #expect(throws: PipelineError.self) { try await mailbox.offer(3) }
        #expect(await mailbox.pop() == 1); #expect(await mailbox.pop() == 2)
        await mailbox.close()
        await #expect(throws: PipelineError.self) { try await mailbox.offer(4) }
    }
    @Test func inFlightAdmissionIsBounded() async throws {
        var config = PipelineConfiguration(); config.maxInFlight = 1
        let pipeline = try DeskPetPipeline(adapters: .mocks(router: MockRouter(behavior: .init(delay: 0.1))), configuration: config)
        await pipeline.start(automaticReminders: false)
        _ = try await send(pipeline, "a", conversation: "a")
        await #expect(throws: PipelineError.self) { _ = try await send(pipeline, "b", conversation: "b") }
        await pipeline.shutdown(); #expect(await pipeline.inFlightCount() == 0)
    }
    @Test func contextStampsRejectLateOldTargetOutput() async throws {
        let config = PipelineConfiguration(); let store = RouterContextStore(config)
        let envelope = Envelope(conversationId: "test", kind: "stt.final", payload: Transcript(text: "보고서 완료"))
        let input = await store.prepare(envelope, resolution: .unique(report))
        let stamp = await store.commit(envelope, input: input, decision: nil)
        let next = Envelope(conversationId: "test", kind: "stt.final", payload: Transcript(text: "회의 완료"))
        _ = await store.prepare(next, resolution: .unique(meeting))
        let output = OutputEvent(requestId: envelope.requestId, outputId: UUID(), text: "늦은 보고서 안내", emotion: "neutral",
            activity: .speaking, displayState: "normal", displayMode: .face, displayText: "", speech: SpeechPolicy())
        await store.delivered(output, conversation: "test", target: report, stamp: stamp, reminder: false)
        #expect(await store.turns("test").isEmpty)
        #expect(await store.commit(envelope, input: input, decision: nil) == nil)
    }
    @Test func dateRoundTripPreservesMillisecondsAndTimeZone() throws {
        let date = ISO8601DateFormatter().date(from: "2026-10-02T23:59:00+09:00")!
        #expect(DateCodec.rfc3339(date) == "2026-10-02T14:59:00Z")
        #expect(Date(timeIntervalSince1970: Double(try DateCodec.epochMilliseconds(date)) / 1_000) == date)
        #expect(DateCodec.reminderDeadline(item(due: date)).contains("23시 59분"))
    }
    @Test func maskingDoesNotRequireDueDateForChat() {
        let heads = HeadOutputs(intent: .chat, dueAt: Date(), priority: "high")
        let masked = DecisionComposer.mask(heads)
        #expect(masked.dueAt.value == nil); #expect(masked.priority.value == nil)
    }
    @Test func invalidConfigurationFailsAtStartup() throws {
        var config = PipelineConfiguration(); config.queueCapacity = 0
        #expect(throws: PipelineError.self) { _ = try DeskPetPipeline(adapters: .mocks(), configuration: config) }
    }
    @Test func webFailureDoesNotChangeExecution() async throws {
        let pipeline = try DeskPetPipeline(adapters: .mocks(router: fixtureRouter(), eureka: MockEureka(items: [item()]),
            web: MockWeb(behavior: .init(error: PipelineError("WEB_ERROR", stage: "web")))))
        await pipeline.start(automaticReminders: false)
        let id = try await send(pipeline, "보고서 완료"); try await idle(pipeline)
        #expect(await pipeline.record(requestId: id)?.action?.status == .succeeded)
        #expect(await pipeline.record(requestId: id)?.delivery?.audio == .completed)
        await pipeline.shutdown()
    }
}

extension PipelineTests {
    @Test func cancelledHarnessKeepsUnknownAndSuppressesStaleOutput() async throws {
        let harness = MockHarness(behavior: .init(delay: 1)); let speaker = MockSpeaker()
        let pipeline = try DeskPetPipeline(adapters: .mocks(router: fixtureRouter(), harness: harness, speaker: speaker))
        await pipeline.start(automaticReminders: false)
        let old = try await send(pipeline, "외부 조회")
        let limit = ContinuousClock.now.advanced(by: .seconds(1))
        while await harness.count() == 0 {
            #expect(ContinuousClock.now < limit); try await Task.sleep(for: .milliseconds(2))
        }
        await pipeline.cancel(conversationId: "deskpet")
        _ = try await send(pipeline, "새 대화"); try await idle(pipeline)
        #expect(await pipeline.record(requestId: old)?.action?.status == .unknown)
        #expect(await speaker.spoken().count == 1)
        #expect(await speaker.spoken().first?.contains("mock 도구") == false)
        await pipeline.shutdown()
    }
    @Test func unresolvedJournalCannotBeEvicted() async throws {
        var config = PipelineConfiguration(); config.journalCapacity = 1
        let pipeline = try DeskPetPipeline(adapters: .mocks(router: fixtureRouter(), eureka: MockEureka(items: [item()], loseMutationReply: true)), configuration: config)
        await pipeline.start(automaticReminders: false)
        let id = try await send(pipeline, "보고서 완료"); try await idle(pipeline)
        await #expect(throws: PipelineError.self) { _ = try await send(pipeline, "안녕") }
        #expect(await pipeline.record(requestId: id)?.action?.status == .unknown)
        await pipeline.shutdown()
    }
    @Test func questionExpiryDetachesLateAnswer() async throws {
        var config = PipelineConfiguration(); config.questionWait = 0.01
        let pipeline = try DeskPetPipeline(adapters: .mocks(router: fixtureRouter(), eureka: MockEureka(items: [item()])), configuration: config)
        await pipeline.start(automaticReminders: false)
        let original = try await send(pipeline, "모호"); try await idle(pipeline)
        try await Task.sleep(for: .milliseconds(30))
        let next = try await send(pipeline, "보고서 완료"); try await idle(pipeline)
        #expect(next != original)
        await pipeline.shutdown()
    }
    @Test func routerFailureRetainsOnlyUserAndDeliveredFailure() async throws {
        let pipeline = try DeskPetPipeline(adapters: .mocks(router: MockRouter(behavior: .init(error: PipelineError("INFERENCE_ERROR", stage: "router")))))
        await pipeline.start(automaticReminders: false)
        let id = try await send(pipeline, "요청"); try await idle(pipeline)
        #expect(await pipeline.record(requestId: id)?.action?.status == .failed)
        #expect(await pipeline.record(requestId: id)?.action?.error?.message.contains("INFERENCE_ERROR") == true)
        #expect(await pipeline.contextTurns(conversationId: "deskpet").first?.content == "요청")
        await pipeline.shutdown()
    }
    @Test func envelopeSerializationUsesRFC3339Dates() throws {
        let date = Date(timeIntervalSince1970: 1_700_000_000)
        var value = Envelope(conversationId: "fixture", kind: "stt.final", payload: Transcript(text: "hello", audioStartedAt: date, audioEndedAt: date))
        value.createdAt = date
        let bytes = try ContractCodec.encoder().encode(value)
        let text = String(decoding: bytes, as: UTF8.self)
        #expect(text.contains("2023-11-14T22:13:20Z")); #expect(text.contains("schemaVersion"))
        let decoded = try ContractCodec.decoder().decode(Envelope<Transcript>.self, from: bytes)
        #expect(decoded.requestId == value.requestId); #expect(decoded.createdAt == date)
    }
    @Test func slowFirstAudioChunkFallsBackAndWebIsIndependent() async throws {
        var config = PipelineConfiguration(); config.ttsFirstChunkTimeout = 0.01
        let pipeline = try DeskPetPipeline(adapters: .mocks(tts: MockTTS(behavior: .init(delay: 1))), configuration: config)
        await pipeline.start(automaticReminders: false)
        let id = try await send(pipeline, "안녕"); try await idle(pipeline)
        #expect(await pipeline.record(requestId: id)?.delivery?.display == .completed)
        #expect(await pipeline.record(requestId: id)?.delivery?.audio == .failed)
        await pipeline.shutdown()
    }
}

private actor FrameFixture: MicrophoneSource {
    var frames: [AudioFrame]
    init(_ frames: [AudioFrame]) { self.frames = frames }
    func nextFrame() async throws -> AudioFrame? { frames.isEmpty ? nil : frames.removeFirst() }
}
private struct VADFixture: VoiceActivityDetector {
    func classify(_ frame: AudioFrame) async throws -> VoiceActivity { frame.chunk.data.isEmpty ? .end : .speech }
}
extension PipelineTests {
    @Test func audioIngressConnectsVADAndSTT() async throws {
        let pipeline = try DeskPetPipeline(adapters: .mocks())
        await pipeline.start(automaticReminders: false)
        let capture = AudioIngress(microphone: FrameFixture([
            AudioFrame(chunk: PCMChunk(data: Data("hello".utf8))), AudioFrame(chunk: PCMChunk(data: Data()))
        ]), vad: VADFixture())
        try await capture.run(pipeline: pipeline, conversationId: "audio"); try await idle(pipeline)
        #expect(await pipeline.contextTurns(conversationId: "audio").first?.content == "hello")
        await pipeline.shutdown()
    }
    @Test func damagedAudioNeverExecutesTruncatedCommand() async throws {
        var config = AudioIngressConfiguration(); config.maxSegmentBytes = 4
        let eureka = MockEureka(items: [item()])
        let pipeline = try DeskPetPipeline(adapters: .mocks(router: fixtureRouter(), eureka: eureka))
        await pipeline.start(automaticReminders: false)
        let capture = AudioIngress(microphone: FrameFixture([
            AudioFrame(chunk: PCMChunk(data: Data("보고서 완료".utf8))), AudioFrame(chunk: PCMChunk(data: Data()))
        ]), vad: VADFixture(), configuration: config)
        try await capture.run(pipeline: pipeline, conversationId: "audio")
        #expect(await eureka.count() == 0)
        #expect(await pipeline.contextTurns(conversationId: "audio").isEmpty)
        await pipeline.shutdown()
    }
    @Test func newExplicitTargetDetachesOldClarificationRequest() async throws {
        let router = MockRouter(scripts: [
            "보고서 기한": MockRoute(heads: HeadOutputs(intent: .postpone), resolution: .unique(report)),
            "회의 완료": MockRoute(heads: HeadOutputs(intent: .complete), resolution: .unique(meeting))
        ])
        let pipeline = try DeskPetPipeline(adapters: .mocks(router: router, eureka: MockEureka(items: [item(), item(meeting)])))
        await pipeline.start(automaticReminders: false)
        let original = try await send(pipeline, "보고서 기한"); try await idle(pipeline)
        let next = try await send(pipeline, "회의 완료"); try await idle(pipeline)
        #expect(next != original)
        #expect(await pipeline.record(requestId: next)?.action?.target?.entityId == meeting.entityId)
        #expect(await pipeline.record(requestId: original)?.action?.status == .needsClarification)
        await pipeline.shutdown()
    }
    @Test func contextMemoryAndHistoryStayBounded() async throws {
        var config = PipelineConfiguration(); config.maxContextTurns = 4
        let pipeline = try DeskPetPipeline(adapters: .mocks(router: fixtureRouter(), eureka: MockEureka(items: [item()])), configuration: config)
        await pipeline.start(automaticReminders: false)
        _ = try await send(pipeline, "기한 변경"); try await idle(pipeline)
        for _ in 0..<4 { _ = try await send(pipeline, "높여줘"); try await idle(pipeline) }
        let turns = await pipeline.contextTurns(conversationId: "deskpet")
        #expect(turns.count == 4); #expect(turns.reduce(0, { $0 + $1.content.utf8.count }) <= config.contextBytes)
        await pipeline.shutdown()
    }
}

extension PipelineTests {
    @Test func invalidDatesFailWithoutIntegerConversionTrap() throws {
        #expect(throws: PipelineError.self) { _ = try DateCodec.epochMilliseconds(Date(timeIntervalSince1970: .infinity)) }
        #expect(item(due: Date(timeIntervalSince1970: .infinity)).key.contains("invalid"))
    }
    @Test func duplicateMockFixtureIdsDoNotTrap() async throws {
        let original = item(); var replacement = original; replacement.description = "updated"
        let eureka = MockEureka(items: [original, replacement])
        #expect(try await eureka.reminders().count == 1)
        #expect(try await eureka.reminder(itemId: report.entityId)?.description == "updated")
    }
}

private struct OversizedHarness: HarnessAdapter {
    func run(_ request: HarnessRequest) async throws -> HarnessResult {
        HarnessResult(action: ActionResult(operationId: request.action.operationId, status: .succeeded,
            action: .mcp, facts: ["oversized": String(repeating: "x", count: 100_000)]))
    }
}
extension PipelineTests {
    @Test func oversizedAdapterResultIsNotRetainedInJournal() async throws {
        var adapters = Adapters.mocks(router: fixtureRouter()); adapters.harness = OversizedHarness()
        let pipeline = try DeskPetPipeline(adapters: adapters)
        await pipeline.start(automaticReminders: false)
        let request = try await send(pipeline, "외부 조회"); try await idle(pipeline)
        let record = await pipeline.record(requestId: request)
        #expect(record?.action?.status == .succeeded)
        #expect(record?.action?.facts.isEmpty == true)
        #expect(record?.action?.error?.code == "RESULT_TOO_LARGE")
        await pipeline.shutdown()
    }
}

private actor BlockingAPIFixture: EurekaAdapter, HarnessAdapter {
    private var started = 0
    private var active = 0
    func counts() -> (started: Int, active: Int) { (started, active) }
    private func wait(_ request: ActionRequest) async throws -> ActionResult {
        started += 1; active += 1
        defer { active -= 1 }
        try await Task.sleep(for: .seconds(5))
        return ActionResult(operationId: request.operationId, status: .succeeded,
                            action: request.decision.headOutputs.intention.value, target: request.decision.resolvedTarget)
    }
    func execute(_ request: ActionRequest) async throws -> ActionResult { try await wait(request) }
    func run(_ request: HarnessRequest) async throws -> HarnessResult { HarnessResult(action: try await wait(request.action)) }
    func reconcile(_ request: ActionRequest) async throws -> ActionResult {
        ActionResult(operationId: request.operationId, status: .unknown)
    }
    func reminders() async throws -> [ReminderItem] { [] }
    func reminder(itemId: String) async throws -> ReminderItem? { nil }
}
private func waitForAPICalls(_ api: BlockingAPIFixture, _ count: Int) async throws {
    let end = ContinuousClock.now.advanced(by: .seconds(1))
    while await api.counts().started < count {
        guard ContinuousClock.now < end else { throw PipelineError("TEST_TIMEOUT", stage: "api concurrency") }
        try await Task.sleep(for: .milliseconds(2))
    }
}
extension PipelineTests {
    @Test(arguments: [false, true]) func apiRepliesDoNotSerializeLaterRequests(harness: Bool) async throws {
        let api = BlockingAPIFixture(); var adapters = Adapters.mocks(router: fixtureRouter())
        adapters.eureka = api; adapters.harness = api
        let pipeline = try DeskPetPipeline(adapters: adapters)
        await pipeline.start(automaticReminders: false)
        let first = try await send(pipeline, harness ? "외부 조회" : "보고서 완료", conversation: "a")
        try await waitForAPICalls(api, 1)
        let second = try await send(pipeline, harness ? "외부 조회" : "회의 완료", conversation: "b")
        try await waitForAPICalls(api, 2)
        #expect(await api.counts().active == 2)
        // Both suspended API operations must be cancelled and joined on shutdown.
        await pipeline.shutdown()
        #expect(await api.counts().active == 0)
        #expect(await pipeline.record(requestId: first)?.action?.status == .unknown)
        #expect(await pipeline.record(requestId: second)?.action?.status == .unknown)
        #expect(await pipeline.inFlightCount() == 0)
    }
    @Test(arguments: [false, true]) func apiCapacityRejectsWithoutQueueingOrDispatch(harness: Bool) async throws {
        let api = BlockingAPIFixture(); var adapters = Adapters.mocks(router: fixtureRouter())
        adapters.eureka = api; adapters.harness = api
        var config = PipelineConfiguration()
        config.maxConcurrentEurekaRequests = 1; config.maxConcurrentHarnessRequests = 1
        let pipeline = try DeskPetPipeline(adapters: adapters, configuration: config)
        await pipeline.start(automaticReminders: false)
        _ = try await send(pipeline, harness ? "외부 조회" : "보고서 완료", conversation: "a")
        try await waitForAPICalls(api, 1)
        let second = try await send(pipeline, harness ? "외부 조회" : "회의 완료", conversation: "b")
        let end = ContinuousClock.now.advanced(by: .seconds(1))
        while await pipeline.record(requestId: second)?.action == nil {
            guard ContinuousClock.now < end else { throw PipelineError("TEST_TIMEOUT", stage: "api capacity") }
            try await Task.sleep(for: .milliseconds(2))
        }
        #expect(await pipeline.record(requestId: second)?.action?.status == .failed)
        #expect(await pipeline.record(requestId: second)?.action?.error?.code == "API_BUSY")
        #expect(await pipeline.record(requestId: second)?.operation == nil)
        #expect(await api.counts().started == 1)
        await pipeline.shutdown()
        #expect(await api.counts().started == 1)
    }
    @Test func fullHarnessCapacityDoesNotBlockEureka() async throws {
        let api = BlockingAPIFixture(); var adapters = Adapters.mocks(router: fixtureRouter())
        adapters.eureka = api; adapters.harness = api
        var config = PipelineConfiguration(); config.maxConcurrentHarnessRequests = 1
        let pipeline = try DeskPetPipeline(adapters: adapters, configuration: config)
        await pipeline.start(automaticReminders: false)
        _ = try await send(pipeline, "외부 조회", conversation: "harness")
        try await waitForAPICalls(api, 1)
        _ = try await send(pipeline, "보고서 완료", conversation: "eureka")
        try await waitForAPICalls(api, 2)
        #expect(await api.counts().active == 2)
        await pipeline.shutdown()
    }
}
