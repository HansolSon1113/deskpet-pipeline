import Foundation
import DeskPetPipeline

/// Opt-in GPU evaluation, run on the user's host. Never changes a model or
/// substitutes CPU execution when either reference or INT4 inference fails.
enum RouterEvaluation {
    struct Input: Codable, Sendable { let messages: [ChatMessage] }
    static func run(library: NativeBackendLibrary, configuration: NativeModelConfiguration, executor: GPUExecutor,
                    dataset: String, split: String, reference: String, report: String, limit: Int?) async throws {
        let indices = try JSONDecoder().decode([String: [Int]].self, from: Data(contentsOf: URL(fileURLWithPath: split)))
        guard let test = indices["test"], !test.isEmpty else { throw PipelineError("EVALUATION_SPLIT_MISSING", stage: "evaluation") }
        let selected = Array(test.prefix(limit ?? test.count)); let wanted = Set(selected)
        var inputs: [Int: Input] = [:]; var index = 0
        // Lines are read incrementally; the full training corpus is not retained.
        let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: dataset)); defer { try? handle.close() }
        var pending = Data()
        while let bytes = try handle.read(upToCount: 64*1024), !bytes.isEmpty {
            pending.append(bytes)
            while let end = pending.firstIndex(of: 10) {
                let line = Data(pending[..<end]); pending.removeSubrange(...end)
                if !line.isEmpty {
                    if wanted.contains(index) { inputs[index] = try JSONDecoder().decode(Input.self, from: line) }
                    index += 1
                }
            }
        }
        if !pending.isEmpty, wanted.contains(index) { inputs[index] = try JSONDecoder().decode(Input.self, from: pending) }
        guard inputs.count == selected.count else { throw PipelineError("EVALUATION_DATASET_MISMATCH", stage: "evaluation") }
        func evaluate(_ path: String) async throws -> [HeadOutputs] {
            var config = configuration; config.paths["model"] = path
            let model = try NativeInferenceModel(library: library, configuration: config)
            var outputs: [HeadOutputs] = []
            for id in selected {
                let input = inputs[id]!
                let result = try await model.gpu(input, executor: executor, as: HeadOutputs.self)
                try NativeRouterContract.validate(result, messages: input.messages); outputs.append(result)
            }
            return outputs
        }
        let baseline = try await evaluate(reference)
        let quantized = try await evaluate(configuration.paths["model"]!)
        let encoder = JSONEncoder(); var rows: [[String: Any]] = []; var agreements: [String: Int] = [:]
        var sumDelta = 0.0, maxDelta = 0.0
        for i in selected.indices {
            let a = try JSONSerialization.jsonObject(with: encoder.encode(baseline[i])) as! [String: Any]
            let b = try JSONSerialization.jsonObject(with: encoder.encode(quantized[i])) as! [String: Any]
            for head in ["modelIntention", "operation", "target", "name", "checkpoint", "dueAt", "priority", "task", "note"] {
                let av = (a[head] as? [String: Any])?["value"] ?? NSNull()
                let bv = (b[head] as? [String: Any])?["value"] ?? NSNull()
                if NSDictionary(dictionary: ["v": av]).isEqual(to: ["v": bv]) { agreements[head, default: 0] += 1 }
            }
            let delta = abs(baseline[i].modelConfidence! - quantized[i].modelConfidence!)
            sumDelta += delta; maxDelta = max(maxDelta, delta)
            rows.append(["index": selected[i], "reference": a, "quantized": b, "absoluteConfidenceDelta": delta])
        }
        let result: [String: Any] = ["schemaVersion": 1, "split": "test", "indices": selected,
            "referenceModel": URL(fileURLWithPath: reference).lastPathComponent,
            "quantizedModel": URL(fileURLWithPath: configuration.paths["model"]!).lastPathComponent,
            "headValueAgreementCounts": agreements, "meanAbsoluteConfidenceDelta": sumDelta/Double(selected.count),
            "maximumAbsoluteConfidenceDelta": maxDelta, "samples": rows]
        try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: report), options: .atomic)
        print("동일 test split 비교 완료: \(selected.count)건; 보고서: \(report)")
    }
}
