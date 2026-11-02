import Foundation

/// Console output is limited to readable results, never audio or frame streams.
public enum NativeConsole {
    private static let writer = Writer()

    public static func modelResult(_ role: String, payload: Data) async {
        // These model contracts contain text, labels, spans and scalar scores.
        // STT is logged only by the final-utterance adapter; VAD and TTS are silent.
        guard ["router", "output", "emotion"].contains(role) else { return }
        await formatted(role, payload: payload)
    }
    public static func write(_ label: String, payload: Data) async {
        var record = Data((payload.contains(10) ? "[\(label)]\n" : "[\(label)] ").utf8)
        record.append(payload); record.append(10)
        await writer.enqueue(record)
    }
    public static func value<T: Encodable & Sendable>(_ label: String, _ value: T) async {
        do { await formatted(label, payload: try ContractCodec.encoder().encode(value)) }
        catch { await write("error", payload: Data(String(describing: error).utf8)) }
    }
    private static func formatted(_ label: String, payload: Data) async {
        do {
            let value = try JSONSerialization.jsonObject(with: payload, options: [.fragmentsAllowed])
            let text: String
            if label == "router", let object = value as? [String: Any] {
                text = router(object)
            } else if label == "output", let object = value as? [String: Any], let output = object["text"] as? String {
                text = output
            } else { text = lines(value).joined(separator: "\n") }
            await write(label, payload: Data(text.utf8))
        } catch { await write("error", payload: Data(String(describing: error).utf8)) }
    }
    private static func scalar(_ value: Any) -> String {
        if value is NSNull { return "null" }
        if let number = value as? NSNumber {
            if String(cString: number.objCType) == "c" { return number.boolValue ? "true" : "false" }
            return number.stringValue
        }
        return String(describing: value)
    }
    private static func router(_ object: [String: Any]) -> String {
        var result: [String] = []
        func head(_ title: String, _ value: Any?) {
            guard let head = value as? [String: Any] else { return }
            guard let rawValue = head["value"] else { return }
            let value = scalar(rawValue)
            let score = head["confidence"].map { "  (신뢰도: \(scalar($0)))" } ?? ""
            result.append("  \(title): \(value.replacingOccurrences(of: "\n", with: "\n    "))\(score)")
        }
        head("의도", object["modelIntention"] ?? object["intention"])
        head("Operation", object["operation"])
        if let score = object["modelConfidence"] { result.append("  종합 신뢰도: \(scalar(score))") }
        head("실행 의도", object["intention"])
        for (key, title) in [("target", "업무"), ("name", "단계"), ("checkpoint", "체크포인트"),
                             ("dueAt", "기한"), ("priority", "우선순위"), ("task", "이름/요약"), ("note", "설명")] {
            head(title, object[key])
        }
        if let spans = object["spans"] as? [String: Any], !spans.isEmpty {
            result.append("  원문 위치:")
            for name in spans.keys.sorted() {
                guard let span = spans[name] as? [String: Any] else { continue }
                let source = ["message", "start", "end"].compactMap { key in span[key].map { "\(key)=\(scalar($0))" } }.joined(separator: ", ")
                let text = span["text"].map(scalar) ?? ""
                result.append("    \(name): \(text.replacingOccurrences(of: "\n", with: "\n      "))  (\(source))")
            }
        }
        if let finished = object["generationFinished"] as? [String: Any], !finished.isEmpty {
            result.append("  생성 종료:")
            for key in finished.keys.sorted() { if let value = finished[key] { result.append("    \(key): \(scalar(value))") } }
        }
        return result.joined(separator: "\n")
    }
    private static let names = [
        "text": "텍스트", "confidence": "신뢰도", "positive": "긍정 확률",
        "status": "상태", "action": "동작", "target": "대상", "facts": "결과",
        "error": "오류", "code": "코드", "stage": "발생 위치", "message": "내용",
        "retryable": "재시도 가능", "operationId": "작업 ID", "entityId": "대상 ID", "name": "이름",
    ]
    private static func lines(_ value: Any, indent: String = "") -> [String] {
        if let object = value as? [String: Any] {
            let priority = ["status", "action", "text", "confidence", "positive", "error", "target", "facts"]
            let keys = priority.filter { object[$0] != nil } + object.keys.filter { !priority.contains($0) }.sorted()
            return keys.flatMap { key -> [String] in
                guard let original = object[key] else { return [] }
                // Media, buffers and internal numerical arrays never belong in the console.
                if ["pcm", "base64", "audio", "tensor", "tensors", "logits", "hiddenStates", "buffer", "buffers"].contains(key) { return [] }
                var child = original
                if ["questionReasons", "timings", "wikiExcerpts"].contains(key), let text = original as? String,
                   let data = text.data(using: .utf8), let parsed = try? JSONSerialization.jsonObject(with: data) { child = parsed }
                let title = names[key] ?? key
                if child is [String: Any] || child is [Any] {
                    let nested = lines(child, indent: indent + "  ")
                    return nested.isEmpty ? ["\(indent)\(title): 없음"] : ["\(indent)\(title):"] + nested
                }
                return ["\(indent)\(title): \(scalar(child).replacingOccurrences(of: "\n", with: "\n" + indent + "  "))"]
            }
        }
        if let array = value as? [Any] {
            return array.flatMap { child -> [String] in
                if child is [String: Any] || child is [Any] { return ["\(indent)-"] + lines(child, indent: indent + "  ") }
                return ["\(indent)- \(scalar(child))"]
            }
        }
        return [indent + scalar(value)]
    }
    public static func flush() async { await writer.flush() }

    private actor Writer {
        private let io = DispatchQueue(label: "deskpet.console.stdout")
        private var records: [Data] = []
        private var bytes = 0
        private var writing = false
        private var space: [CheckedContinuation<Void, Never>] = []
        private var drains: [CheckedContinuation<Void, Never>] = []

        func enqueue(_ record: Data) async {
            while records.count + (writing ? 1 : 0) >= 32 ||
                (bytes > 0 && bytes + record.count > 256 * 1024) {
                await withCheckedContinuation { space.append($0) }
            }
            records.append(record); bytes += record.count
            startNext()
        }
        func flush() async {
            if !writing && records.isEmpty { return }
            await withCheckedContinuation { drains.append($0) }
        }
        private func startNext() {
            guard !writing, !records.isEmpty else { return }
            writing = true
            let record = records.removeFirst()
            io.async {
                try? FileHandle.standardOutput.write(contentsOf: record)
                Task { await self.completed(record.count) }
            }
        }
        private func completed(_ count: Int) {
            bytes -= count; writing = false
            let waiting = space; space.removeAll()
            for continuation in waiting { continuation.resume() }
            startNext()
            if !writing && records.isEmpty {
                let waiting = drains; drains.removeAll()
                for continuation in waiting { continuation.resume() }
            }
        }
    }
}
