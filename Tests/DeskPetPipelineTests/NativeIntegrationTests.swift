import Foundation
import Testing
@testable import DeskPetPipeline

private func waitForIdle(_ pipeline: DeskPetPipeline) async throws {
    let end = ContinuousClock.now.advanced(by: .seconds(2))
    while await pipeline.inFlightCount() > 0 {
        guard ContinuousClock.now < end else { throw PipelineError("TEST_TIMEOUT", stage: "test") }
        try await Task.sleep(for: .milliseconds(2))
    }
}

@Test func opaqueNativeBackendPreservesPhysicalLifetimeAndFixedText() async throws {
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let temp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: temp) }
    let libraryPath = temp.appendingPathComponent("fixture.dylib").path
    let compiler = Process(); compiler.executableURL = URL(fileURLWithPath: "/usr/bin/clang")
    compiler.arguments = ["-shared", "-fPIC", "-I", root.appendingPathComponent("Sources/CNativeInference/include").path,
                          root.appendingPathComponent("native/tests/fixture.c").path, "-o", libraryPath]
    try compiler.run(); compiler.waitUntilExit(); #expect(compiler.terminationStatus == 0)
    let library = try NativeBackendLibrary(path: libraryPath)
    #expect(throws: PipelineError.self) {
        _ = try NativeInferenceModel(library: library, configuration: NativeModelConfiguration(role: "output", device: "cpu", paths: [:]))
    }
    #expect(throws: PipelineError.self) {
        _ = try NativeInferenceModel(library: library, configuration: NativeModelConfiguration(backend: "missing-backend", role: "output", device: "metal", paths: [:]))
    }
    let model = try NativeInferenceModel(library: library, configuration: NativeModelConfiguration(role: "output", device: "metal", paths: [:]))
    let executor = try GPUExecutor(capacity: 1)
    let output = NativeOutputModel(model: model, executor: executor)
    #expect(!output.shortOutput)
    let userMessages = [ChatMessage(role: "user", content: "파스타 레시피 알려줘")]
    #expect(NativeOutputModel.preparedMessages(for: userMessages, shortOutput: false) == userMessages)
    let shortMessages = NativeOutputModel.preparedMessages(for: userMessages, shortOutput: true)
    #expect(shortMessages.count == 2)
    #expect(shortMessages.first?.role == "system")
    #expect(shortMessages.first?.content == NativeOutputModel.shortOutputPrompt)
    #expect(shortMessages.last == userMessages.first)
    let rawSpeech = "**재료(1인분)**: *스파게티 면*, 올리브유(3큰술), 마늘 (5쪽)을 준비합니다.\n- 면을 삶습니다 (약 8분(중불)).\n- 마늘을 볶아 완성합니다!"
    #expect(NativeOutputModel.sanitizeForSpeech(rawSpeech) == "재료: 스파게티 면, 올리브유, 마늘을 준비합니다. 면을 삶습니다. 마늘을 볶아 완성합니다!")
    let shortOutputModel = NativeOutputModel(model: model, executor: executor, shortOutput: true)
    let question = OutputRequest(requestId: UUID(), requiredText: "PR 1번을 승인할까요?", isQuestion: true, messages: [.init(role: "user", content: "바꿔 말해줘")])
    #expect(try await output.generate(question) == question.requiredText)
    #expect(try await shortOutputModel.generate(question) == question.requiredText)
    #expect(executor.snapshot().submitted == 0)
    let request = OutputRequest(requestId: UUID(), requiredText: "", messages: [.init(role: "user", content: "안녕")])
    let task = Task { try await output.generate(request) }
    while executor.snapshot().submitted == 0 { try await Task.sleep(for: .milliseconds(1)) }
    task.cancel()
    await #expect(throws: CancellationError.self) { _ = try await task.value }
    #expect(executor.snapshot().outstanding == 1)
    await executor.shutdown()
    #expect(executor.snapshot().outstanding == 0)
    #expect(executor.snapshot().completed == 1)
}

@Test func nativeRouterMetadataKeepsUnicodeAndWireHeadShape() throws {
    let text = "  내일 👩‍💻 검토"
    var heads = HeadOutputs(intent: .harnessTask)
    heads.modelIntention = Head(.llm); heads.modelConfidence = 0.9
    heads.spans = ["DueAt": ModelSpan(message: 0, start: 2, end: 4, text: "내일")]
    try NativeRouterContract.validate(heads, messages: [.init(role: "user", content: text)])
    var invalid = heads; invalid.spans?["DueAt"]?.end = 5
    #expect(throws: PipelineError.self) { try NativeRouterContract.validate(invalid, messages: [.init(role: "user", content: text)]) }
    #expect(NativeRouterContract.answer("네, 승인해줘") == true)
    #expect(NativeRouterContract.answer("취소해줘") == false)
    #expect(NativeRouterContract.answer("네, 그런데 다른 PR을 검토해줘") == nil)
    #expect(NativeRouterContract.date("다음에 시간 날 때", relativeTo: Date()) == nil)
    let server = try ServerHarness(url: URL(string: "ws://127.0.0.1:1")!, token: "unused-test-token")
    let decision = RouteDecision(headOutputs: heads, overallConfidence: 0.9, missingRequiredSlots: [], route: .harness, sourceTurnIds: [])
    let action = try server.wireAction(ActionRequest(requestId: UUID(), decision: decision, deadlineAt: Date().addingTimeInterval(1)))
    let wire = action["decision"] as! [String: Any]
    let wireHeads = wire["headOutputs"] as! [String: Any]
    #expect(wireHeads["modelConfidence"] == nil && wireHeads["spans"] == nil)
    #expect(wireHeads.values.allSatisfy { ($0 as? [String: Any])?["confidence"] is Double })
    #expect((wire["modelMetadata"] as? [String: Any])?["spans"] != nil)
    let ref = HarnessTaskRef(itemId: "item", stageId: "stage", repository: [:], prNumber: 1)
    #expect(ref.source.kind == "eureka" && !ref.source.ref.isEmpty)
}

private actor ConfirmationHarness: HarnessAdapter, HarnessEventConsumer {
    var calls: [HarnessRequest] = []; var reports: [HarnessPipelineEvent] = []
    func report(_ event: HarnessPipelineEvent) { reports.append(event) }
    func snapshot() -> ([HarnessRequest], [HarnessPipelineEvent]) { (calls, reports) }
    func run(_ request: HarnessRequest) -> HarnessResult {
        calls.append(request)
        let n = calls.count
        if n == 3 { return HarnessResult(action: ActionResult(operationId: request.action.operationId, status: .succeeded), resultText: "Eureka 단계 완료를 확인했어요.") }
        return HarnessResult(action: ActionResult(operationId: request.action.operationId, status: .needsClarification,
            facts: ["pendingId": "pending-\(n)", "outputId": "output-\(n)"]),
            question: n == 1 ? "PR 1번을 승인할까요?" : "연결된 Eureka 단계를 완료할까요?")
    }
}

@Test func scenarioStartsWithDeliveredReminderAndKeepsPRContext() async throws {
    let pr = Target(kind: "github_pr", entityId: "owner/repo#1", name: "PR 1", repository: ["owner": "owner", "name": "repo"], prNumber: 1)
    let harness = ConfirmationHarness(); let speaker = MockSpeaker(); let display = MockDisplay()
    var adapters = Adapters.mocks(router: MockRouter(scripts: ["그거 리뷰해줘": MockRoute(heads: HeadOutputs(intent: .reviewPR))]))
    adapters.harness = harness; adapters.speaker = speaker; adapters.display = display
    adapters.output = MockOutput(replacement: "모델이 바꾼 문장")
    let pipeline = try DeskPetPipeline(adapters: adapters); await pipeline.start(automaticReminders: false)
    let text = "PR 1번 검토를 잊지 마세요. 변경 내용을 확인해 드릴까요?"
    let opening = try await pipeline.startScenarioReminder(target: pr, text: text, conversationId: "opening")
    try await waitForIdle(pipeline)
    #expect(await speaker.spoken() == [text])
    #expect(await display.shown().first?.displayState == "reminder")
    #expect(await pipeline.record(requestId: opening)?.operation == nil)
    #expect(await harness.snapshot().0.isEmpty)
    let turns = await pipeline.contextTurns(conversationId: "opening")
    #expect(turns.count == 1 && turns[0].role == "assistant" && turns[0].content == text)
    await pipeline.interact(.speechStarted("opening"))
    #expect(await harness.snapshot().1.isEmpty)
    let reply = try await pipeline.submit(Envelope(conversationId: "opening", kind: "stt.final", payload: Transcript(text: "그거 리뷰해줘")))
    try await waitForIdle(pipeline)
    #expect(await pipeline.record(requestId: reply)?.operation?.decision.resolvedTarget == pr)
    #expect(await harness.snapshot().0.count == 1)
    #expect(await harness.snapshot().0.first?.mutationApproved == false)
    await pipeline.shutdown()
}

@Test func undeliveredScenarioReminderCreatesNoConversation() async throws {
    var adapters = Adapters.mocks()
    let failure = MockBehavior(error: PipelineError("DEVICE_FAILED", stage: "test"))
    adapters.speaker = MockSpeaker(behavior: failure); adapters.display = MockDisplay(behavior: failure)
    let pipeline = try DeskPetPipeline(adapters: adapters); await pipeline.start(automaticReminders: false)
    _ = try await pipeline.startScenarioReminder(target: Target(entityId: "pr", name: "PR 1"), text: "PR 1번을 검토할까요?", conversationId: "undelivered-opening")
    try await waitForIdle(pipeline)
    #expect(await pipeline.contextTurns(conversationId: "undelivered-opening").isEmpty)
    await pipeline.shutdown()
}

private func confirmationAdapters(_ harness: ConfirmationHarness) -> Adapters {
    let pr = Target(kind: "github_pr", entityId: "owner/repo#1", name: "PR 1", repository: ["owner": "owner", "name": "repo"], prNumber: 1)
    var adapters = Adapters.mocks(router: MockRouter(scripts: [
        "승인 요청": MockRoute(heads: HeadOutputs(intent: .approvePR), resolution: .unique(pr)),
        "네": MockRoute(heads: HeadOutputs(intent: .confirmApprove))
    ]))
    adapters.harness = harness; return adapters
}

@Test func relaxedRoutingSendsModerateConfidenceChatToOutput() async throws {
    for (threshold, score, expected) in [(0.8, 0.45, Route.harness), (0.3, 0.45, Route.output), (0.3, 0.2, Route.harness)] {
        let harness = MockHarness(); let eureka = MockEureka(); let speaker = MockSpeaker()
        var adapters = Adapters.mocks(router: MockRouter(scripts: [
            "안녕": MockRoute(heads: HeadOutputs(intent: .chat, confidence: score))
        ]), eureka: eureka, harness: harness)
        adapters.speaker = speaker; adapters.output = MockOutput(replacement: "반가워요.")
        var config = PipelineConfiguration(); config.routeConfidenceThreshold = threshold
        let pipeline = try DeskPetPipeline(adapters: adapters, configuration: config)
        await pipeline.start(automaticReminders: false)
        let request = try await pipeline.submit(Envelope(conversationId: "relaxed-chat", kind: "stt.final",
            payload: Transcript(text: "안녕", sttConfidence: 0.9)))
        try await waitForIdle(pipeline)
        let decision = await pipeline.record(requestId: request)?.operation?.decision
        if expected == .output {
            #expect(await harness.count() == 0)
            #expect(await speaker.spoken() == ["반가워요."])
        } else {
            #expect(await harness.count() == 1)
            #expect(decision?.overallConfidence == score)
        }
        #expect(await eureka.count() == 0)
        await pipeline.shutdown()
    }
}

@Test func relaxedRoutingKeepsSTTAndApprovalConfidenceGates() async throws {
    var config = PipelineConfiguration(); config.routeConfidenceThreshold = 0.3
    for (sttScore, routerScore) in [(0.69, 0.99), (0.84, 0.99), (0.9, 0.79)] {
        let harness = ConfirmationHarness()
        var adapters = confirmationAdapters(harness)
        let pr = Target(kind: "github_pr", entityId: "owner/repo#1", name: "PR 1", repository: ["owner": "owner", "name": "repo"], prNumber: 1)
        adapters.router = MockRouter(scripts: [
            "승인 요청": MockRoute(heads: HeadOutputs(intent: .approvePR), resolution: .unique(pr)),
            "네": MockRoute(heads: HeadOutputs(intent: .confirmApprove, confidence: routerScore))
        ])
        let pipeline = try DeskPetPipeline(adapters: adapters, configuration: config)
        await pipeline.start(automaticReminders: false)
        _ = try await pipeline.submit(Envelope(conversationId: "relaxed-approval", kind: "stt.final", payload: Transcript(text: "승인 요청")))
        try await waitForIdle(pipeline)
        _ = try await pipeline.submit(Envelope(conversationId: "relaxed-approval", kind: "stt.final", payload: Transcript(text: "네", sttConfidence: sttScore)))
        try await waitForIdle(pipeline)
        let calls = await harness.snapshot().0
        #expect(calls.count == (sttScore < 0.7 ? 1 : 2))
        #expect(calls.allSatisfy { !$0.mutationApproved })
        await pipeline.shutdown()
    }
}

@Test func relaxedSTTDeliversUnchangedLlmRequestToHarness() async throws {
    for sttThreshold in [0.7, 0.3] {
        let harness = ConfirmationHarness()
        var heads = HeadOutputs(intent: .harnessTask, confidence: 0.5463101267814636)
        heads.modelIntention = Head(.llm, confidence: 0.9806195196976736)
        heads.modelConfidence = 0.5463101267814636
        var adapters = confirmationAdapters(harness)
        adapters.router = MockRouter(scripts: ["그 PR 승인해줘.": MockRoute(heads: heads)])
        var config = PipelineConfiguration()
        config.routeConfidenceThreshold = 0.3; config.sttConfidenceThreshold = sttThreshold
        let pipeline = try DeskPetPipeline(adapters: adapters, configuration: config)
        await pipeline.start(automaticReminders: false)
        _ = try await pipeline.submit(Envelope(conversationId: "llm-stt", kind: "stt.final",
            payload: Transcript(text: "그 PR 승인해줘.", sttConfidence: 0.55)))
        try await waitForIdle(pipeline)
        let calls = await harness.snapshot().0
        if sttThreshold == 0.7 { #expect(calls.isEmpty) }
        else {
            let call = try #require(calls.first)
            #expect(calls.count == 1)
            #expect(call.action.decision.headOutputs.intention.value == .harnessTask)
            #expect(call.action.decision.headOutputs.modelIntention?.value == .llm)
            #expect(call.action.decision.overallConfidence == 0.5463101267814636)
            #expect(call.transcript.sttConfidence == 0.55)
            #expect(!call.mutationApproved)
        }
        await pipeline.shutdown()
    }
}

@Test func githubAndEurekaConfirmationsKeepSeparateDeliveryReceipts() async throws {
    let harness = ConfirmationHarness(); let pipeline = try DeskPetPipeline(adapters: confirmationAdapters(harness))
    await pipeline.start(automaticReminders: false)
    for text in ["승인 요청", "네", "네"] {
        _ = try await pipeline.submit(Envelope(conversationId: "approval", kind: "stt.final", payload: Transcript(text: text)))
        try await waitForIdle(pipeline)
    }
    let (calls, reports) = await harness.snapshot()
    #expect(calls.count == 3 && reports.count == 2)
    #expect(calls.first?.mutationApproved == false)
    #expect(calls.dropFirst().allSatisfy { $0.mutationApproved })
    #expect(calls[1].pendingId == "pending-1" && calls[2].pendingId == "pending-2")
    #expect(reports.map(\.outputId) == ["output-1", "output-2"])
    #expect(reports.allSatisfy { $0.kind == "question_delivered" && $0.channel == "speech" })
    #expect(Set(calls.map { $0.action.requestId }).count == 1)
    await pipeline.shutdown()
}

@Test func failedPlaybackAndDisplayNeverReportQuestionDelivered() async throws {
    let harness = ConfirmationHarness(); var adapters = confirmationAdapters(harness)
    let failure = MockBehavior(error: PipelineError("DEVICE_FAILED", stage: "test"))
    adapters.speaker = MockSpeaker(behavior: failure); adapters.display = MockDisplay(behavior: failure)
    let pipeline = try DeskPetPipeline(adapters: adapters); await pipeline.start(automaticReminders: false)
    _ = try await pipeline.submit(Envelope(conversationId: "undelivered", kind: "stt.final", payload: Transcript(text: "승인 요청")))
    try await waitForIdle(pipeline)
    #expect(await harness.snapshot().1.isEmpty)
    await pipeline.shutdown()
}

private actor DeferredHarness: HarnessAdapter, HarnessNoticeSource {
    let notices: AsyncStream<Data>
    let sink: AsyncStream<Data>.Continuation
    var request: HarnessRequest?
    var calls = 0
    init() { let pair = AsyncStream<Data>.makeStream(); notices = pair.stream; sink = pair.continuation }
    func run(_ value: HarnessRequest) -> HarnessResult {
        calls += 1; request = value
        return HarnessResult(action: ActionResult(operationId: value.action.operationId, status: .unknown), resultText: "결과를 확인하고 있어요.")
    }
    func count() -> Int { calls }
    func complete() throws {
        guard let request else { return }
        sink.yield(try JSONSerialization.data(withJSONObject: ["type": "notice", "resultText": "지연된 완료를 확인했어요.",
            "notice": ["requestId": request.action.requestId.uuidString,
                       "pipelineAction": ["status": "succeeded", "facts": ["confirmed": "true"]]]]))
    }
}

@Test func recoveryNoticeUpdatesUnknownWithoutRepeatingExternalCall() async throws {
    let harness = DeferredHarness(); let speaker = MockSpeaker()
    var adapters = Adapters.mocks(router: MockRouter(scripts: ["리뷰": MockRoute(heads: HeadOutputs(intent: .reviewPR))]))
    adapters.harness = harness; adapters.speaker = speaker
    let pipeline = try DeskPetPipeline(adapters: adapters); await pipeline.start(automaticReminders: false)
    let id = try await pipeline.submit(Envelope(conversationId: "notice", kind: "stt.final", payload: Transcript(text: "리뷰")))
    try await waitForIdle(pipeline); #expect(await pipeline.record(requestId: id)?.action?.status == .unknown)
    try await harness.complete()
    let end = ContinuousClock.now.advanced(by: .seconds(1))
    while await pipeline.record(requestId: id)?.action?.status != .succeeded, ContinuousClock.now < end { try await Task.sleep(for: .milliseconds(2)) }
    try await waitForIdle(pipeline)
    #expect(await harness.count() == 1)
    #expect(await pipeline.record(requestId: id)?.action?.status == .succeeded)
    #expect(await speaker.spoken().contains("지연된 완료를 확인했어요."))
    await pipeline.shutdown()
}

@Test func expiryReportsTheHarnessPendingID() async throws {
    let harness = ConfirmationHarness(); var config = PipelineConfiguration(); config.questionWait = 0.02
    let pipeline = try DeskPetPipeline(adapters: confirmationAdapters(harness), configuration: config)
    await pipeline.start(automaticReminders: false)
    _ = try await pipeline.submit(Envelope(conversationId: "expiry", kind: "stt.final", payload: Transcript(text: "승인 요청")))
    try await waitForIdle(pipeline); try await Task.sleep(for: .milliseconds(35))
    let reports = await harness.snapshot().1
    #expect(reports.contains { $0.kind == "question_wait_expired" && $0.pendingId == "pending-1" })
    await pipeline.shutdown()
}
