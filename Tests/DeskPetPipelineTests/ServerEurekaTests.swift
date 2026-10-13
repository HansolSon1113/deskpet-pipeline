import Foundation
import Testing
@testable import DeskPetPipeline

// Opt-in: run against LLM_test's unchanged `pnpm web:fixture` server.
@Test(.enabled(if: ProcessInfo.processInfo.environment["DESKPET_SERVER_TESTS"] == "1"))
func existingEurekaServerConnectsToPipeline() async throws {
    let env = ProcessInfo.processInfo.environment
    let server = try ServerEureka(url: URL(string: env["HARNESS_WS_URL"] ?? "ws://127.0.0.1:8788")!,
                                  token: env["HARNESS_WS_TOKEN"] ?? "fixture-pipeline-token-123")
    let target = Target(entityId: "2001", name: "보고서 제출")
    let decision = RouteDecision(headOutputs: HeadOutputs(intent: .query, name: target.name), overallConfidence: 0.99,
                                 missingRequiredSlots: [], route: .eureka, sourceTurnIds: [], resolvedTarget: target)
    let request = ActionRequest(requestId: UUID(), decision: decision, deadlineAt: Date().addingTimeInterval(10))
    let status = try await server.execute(request)
    #expect(status.status == .succeeded && status.operationId == request.operationId)
    #expect(status.facts["itemId"] == "2001" && status.facts["itemStatus"] == "doing")
    let reminders = try await server.reminders()
    #expect(reminders.contains { $0.target.entityId == "2001" && !$0.description.isEmpty && $0.timeZone == "Asia/Seoul" })
    let reminder = try await server.reminder(itemId: "2001")
    #expect(reminder?.dueAt == reminders.first { $0.target.entityId == "2001" }?.dueAt)
    #expect(try await server.reminder(itemId: "not-a-fixture") == nil)
    let reconciled = try await server.reconcile(request)
    #expect(reconciled.status == .unknown && reconciled.operationId == request.operationId)

    var adapters = Adapters.mocks(router: MockRouter(scripts: ["상태": MockRoute(heads: decision.headOutputs, resolution: .unique(target))]))
    adapters.eureka = server
    let pipeline = try DeskPetPipeline(adapters: adapters)
    await pipeline.start(automaticReminders: false)
    let id = try await pipeline.submit(Envelope(conversationId: "eureka-connection-test", kind: "stt.final", payload: Transcript(text: "상태")))
    let deadline = ContinuousClock.now.advanced(by: .seconds(3))
    while await pipeline.inFlightCount() > 0 && ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(5)) }
    let record = await pipeline.record(requestId: id)
    let turns = await pipeline.contextTurns(conversationId: "eureka-connection-test")
    await pipeline.shutdown()
    #expect(record?.action?.status == .succeeded)
    #expect(record?.action?.facts["itemId"] == "2001")
    #expect(record?.delivery?.delivered == true)
    #expect(turns.contains { $0.role == "assistant" && $0.content == record?.action?.facts["resultText"] })
}

@Test(.enabled(if: ProcessInfo.processInfo.environment["DESKPET_SERVER_TESTS"] == "1"))
func existingHarnessServerConnectsToPipeline() async throws {
    let env = ProcessInfo.processInfo.environment
    let server = try ServerHarness(url: URL(string: env["HARNESS_WS_URL"] ?? "ws://127.0.0.1:8788")!,
                                   token: env["HARNESS_WS_TOKEN"] ?? "fixture-pipeline-token-123")
    let target = Target(kind: "github_pr", entityId: "NewLine/DeskPet#42", name: "DeskPet PR 42",
                        repository: ["owner": "NewLine", "name": "DeskPet"], prNumber: 42)
    var adapters = Adapters.mocks(router: MockRouter(scripts: [
        "리뷰": MockRoute(heads: HeadOutputs(intent: .reviewPR, name: target.name), resolution: .unique(target))]))
    adapters.harness = server
    let pipeline = try DeskPetPipeline(adapters: adapters)
    await pipeline.start(automaticReminders: false)
    let id = try await pipeline.submit(Envelope(conversationId: "harness-connection-test", kind: "stt.final", payload: Transcript(text: "리뷰")))
    let deadline = ContinuousClock.now.advanced(by: .seconds(3))
    while await pipeline.inFlightCount() > 0 && ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(5)) }
    let record = await pipeline.record(requestId: id)
    let turns = await pipeline.contextTurns(conversationId: "harness-connection-test")
    await pipeline.shutdown()
    #expect(record?.operation?.decision.route == .harness)
    #expect(record?.action?.status == .succeeded && record?.action?.facts["action"] == "get_review_context")
    #expect(record?.action?.facts["repository"] == "NewLine/DeskPet" && record?.action?.facts["prNumber"] == "42")
    #expect(record?.action?.facts["headSha"] == "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678")
    #expect(turns.contains { $0.role == "assistant" && $0.content.contains("PR 42번") })
    #expect(record?.delivery?.delivered == true)
}
