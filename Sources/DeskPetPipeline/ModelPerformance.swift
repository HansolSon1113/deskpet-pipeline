import Foundation
import CNativeInference

/// File-only telemetry. Memory belongs to this process, including concurrent
/// models and shared/unified allocations; it is never labelled model VRAM.
public final class ModelPerformance: @unchecked Sendable {
    public struct Span: Sendable {
        fileprivate let id: UUID
        fileprivate let started: UInt64
        fileprivate let role: String
        fileprivate let backend: String
        fileprivate let device: String
        fileprivate let phase: String
        fileprivate let before: Memory
    }
    fileprivate struct Memory: Sendable {
        var rss: UInt64?
        var peakRSS: UInt64?
        var footprint: UInt64?
        static func read() -> Self {
            let value = dp_read_process_memory()
            return Self(rss: value.resident_available != 0 ? value.resident_bytes : nil,
                        peakRSS: value.peak_available != 0 ? value.peak_resident_bytes : nil,
                        footprint: value.footprint_available != 0 ? value.footprint_bytes : nil)
        }
        mutating func include(_ other: Self) {
            func highest(_ a: UInt64?, _ b: UInt64?) -> UInt64? {
                if let a, let b { return max(a, b) }; return a ?? b
            }
            rss = highest(rss, other.rss); footprint = highest(footprint, other.footprint)
        }
    }
    private struct Aggregate {
        var count = 0, failed = 0
        var total = 0.0, maximum = 0.0
        var recent: [Double] = []
        var peak = Memory()
        mutating func add(_ ms: Double, status: String, memory: Memory) {
            count += 1; if status != "ok" { failed += 1 }
            total += ms; maximum = max(maximum, ms); peak.include(memory)
            if recent.count == 1_024 { recent.removeFirst() }; recent.append(ms)
        }
    }
    public let directory: URL
    private let lock = NSLock()
    private let io = DispatchQueue(label: "deskpet.performance.files", qos: .utility)
    private let calls: FileHandle
    private let memory: FileHandle
    private let timer: DispatchSourceTimer
    private let epoch = DispatchTime.now().uptimeNanoseconds
    private var active: [UUID: Memory] = [:]
    private var closed = false
    private var pending = 0
    private var dropped = 0
    // Accessed only on io.
    private var aggregates: [String: Aggregate] = [:]
    private var writeError: String?

    public init(parent: URL, mode: String, models: [String: NativeModelConfiguration],
                keyboard: Bool, routerContext: Bool) throws {
        let date = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let outputDirectory = parent.appendingPathComponent("\(date)-\(mode)-\(UUID().uuidString.prefix(8))")
        directory = outputDirectory
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true,
                                               attributes: [.posixPermissions: 0o700])
        func file(_ name: String, header: String) throws -> FileHandle {
            let url = outputDirectory.appendingPathComponent(name)
            guard FileManager.default.createFile(atPath: url.path, contents: Data((header + "\n").utf8),
                                                  attributes: [.posixPermissions: 0o600]) else {
                throw PipelineError("PERFORMANCE_FILE_FAILED", stage: "performance", message: url.path)
            }
            return try FileHandle(forWritingTo: url).atEnd()
        }
        calls = try file("calls.csv", header: "id,elapsed_ms,role,backend,device,phase,status,duration_ms,process_rss_before_bytes,process_rss_after_bytes,process_rss_delta_bytes,process_rss_sampled_peak_bytes,process_footprint_before_bytes,process_footprint_after_bytes,process_footprint_sampled_peak_bytes,process_lifetime_peak_rss_bytes")
        memory = try file("memory.csv", header: "elapsed_ms,process_rss_bytes,process_footprint_bytes,process_lifetime_peak_rss_bytes,active_calls")
        struct Manifest: Encodable {
            var schema = 1
            var mode: String; var keyboard: Bool; var routerContext: Bool
            var os = ProcessInfo.processInfo.operatingSystemVersionString
            var processors = ProcessInfo.processInfo.processorCount
            var physicalMemoryBytes = ProcessInfo.processInfo.physicalMemory
            var sampleIntervalMilliseconds = 100
            var memoryScope = "pipeline_process_shared; not_per_model_or_remote_gpu_memory"
            var latencyScope = "native_request_including_queue_poll_and_serialization; harness_roundtrip_including_network_and_tools"
            var models: [String: NativeModelConfiguration]
        }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(Manifest(mode: mode, keyboard: keyboard, routerContext: routerContext, models: models))
            .write(to: outputDirectory.appendingPathComponent("run.json"), options: .atomic)
        timer = DispatchSource.makeTimerSource(queue: io)
        timer.setEventHandler { [weak self] in self?.sample() }
        timer.schedule(deadline: .now(), repeating: .milliseconds(100))
        timer.resume()
    }
    deinit { timer.cancel(); try? calls.close(); try? memory.close() }

    public func begin(role: String, backend: String, device: String, phase: String) -> Span? {
        let before = Memory.read()
        let span = Span(id: UUID(), started: DispatchTime.now().uptimeNanoseconds,
                        role: role, backend: backend, device: device, phase: phase, before: before)
        return lock.withLock {
            guard !closed else { return nil }
            active[span.id] = before
            return span
        }
    }
    public func end(_ span: Span?, status: String) {
        guard let span else { return }
        let ended = DispatchTime.now().uptimeNanoseconds
        let after = Memory.read()
        lock.lock()
        guard !closed else { lock.unlock(); return }
        var peak = active.removeValue(forKey: span.id) ?? span.before; peak.include(after)
        // Do not let a stalled disk build an unbounded telemetry queue.
        guard pending < 512 else { dropped += 1; lock.unlock(); return }
        pending += 1
        let duration = Double(ended - span.started) / 1e6
        let elapsed = Double(span.started - epoch) / 1e6
        let delta = span.before.rss.flatMap { a in after.rss.map { String(Int64($0) - Int64(a)) } } ?? ""
        let fields = [span.id.uuidString, String(elapsed), span.role, span.backend, span.device, span.phase, status,
                      String(duration), number(span.before.rss), number(after.rss), delta, number(peak.rss),
                      number(span.before.footprint), number(after.footprint), number(peak.footprint), number(after.peakRSS)]
        let capturedPeak = peak
        io.async { [self] in
            write(fields, to: calls)
            let key = [span.role, span.backend, span.device, span.phase].joined(separator: "\t")
            aggregates[key, default: Aggregate()].add(duration, status: status, memory: capturedPeak)
            lock.withLock { pending -= 1 }
        }
        lock.unlock()
    }
    private func number(_ value: UInt64?) -> String { value.map(String.init) ?? "" }
    private func sample() {
        let value = Memory.read()
        let count: Int? = lock.withLock {
            guard !closed else { return nil }
            for id in Array(active.keys) { active[id]?.include(value) }
            return active.count
        }
        guard let count else { return }
        write([String(Double(DispatchTime.now().uptimeNanoseconds - epoch) / 1e6),
               number(value.rss), number(value.footprint), number(value.peakRSS), String(count)], to: memory)
    }
    private func write(_ fields: [String], to file: FileHandle) {
        let line = fields.map { "\"" + $0.replacingOccurrences(of: "\"", with: "\"\"") + "\"" }.joined(separator: ",") + "\n"
        do { try file.write(contentsOf: Data(line.utf8)) }
        catch { report(error) }
    }
    private func report(_ error: any Error) {
        guard writeError == nil else { return }; writeError = String(describing: error)
        try? FileHandle.standardError.write(contentsOf: Data("성능 기록 오류: \(error)\n".utf8))
    }
    /// Called after pipeline shutdown; drains queued records before closing.
    public func close() {
        let shouldClose = lock.withLock {
            if closed { return false }; closed = true; return true
        }
        guard shouldClose else { return }; timer.cancel()
        io.sync {
            do {
                let url = directory.appendingPathComponent("summary.csv")
                FileManager.default.createFile(atPath: url.path, contents: nil, attributes: [.posixPermissions: 0o600])
                let summary = try FileHandle(forWritingTo: url); defer { try? summary.close() }
                write(["role", "backend", "device", "phase", "count", "failed_or_cancelled", "mean_ms", "p50_recent_ms", "p95_recent_ms", "max_ms", "process_rss_sampled_peak_bytes", "process_footprint_sampled_peak_bytes"], to: summary)
                for key in aggregates.keys.sorted() {
                    guard let value = aggregates[key] else { continue }
                    let sorted = value.recent.sorted()
                    func quantile(_ q: Double) -> String { String(sorted[max(0, Int(ceil(Double(sorted.count) * q)) - 1)]) }
                    write(key.components(separatedBy: "\t") + [String(value.count), String(value.failed),
                        String(value.total / Double(value.count)), quantile(0.5), quantile(0.95), String(value.maximum),
                        number(value.peak.rss), number(value.peak.footprint)], to: summary)
                }
                let state: [String: Any] = ["droppedRecords": lock.withLock { dropped }, "writeError": writeError.map { $0 as Any } ?? NSNull()]
                try JSONSerialization.data(withJSONObject: state, options: [.prettyPrinted, .sortedKeys])
                    .write(to: directory.appendingPathComponent("recording-status.json"), options: .atomic)
                try calls.synchronize(); try memory.synchronize()
            } catch { report(error) }
            try? calls.close(); try? memory.close()
        }
    }
}

private extension FileHandle {
    func atEnd() throws -> FileHandle { _ = try seekToEnd(); return self }
}
