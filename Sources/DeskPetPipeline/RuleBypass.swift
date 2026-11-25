import Foundation

/// Temporary rule-based bypass for concrete scenarios not directly supported by the current Router.
/// Designed as a self-contained module to be removed once Router retraining covers these cases.
public enum RuleBypassAction: Sendable, Equatable {
    case eurekaFilter(String) // today, tomorrow, this_week, overdue, urgent, briefing
    case setTimer(seconds: Int, label: String)
    case queryTimer
    case cancelTimer
    case acknowledgeReminder
    case snoozeReminder(seconds: Int)
    case stopSpeaking
    case repeatLastReply
}

public enum RuleBypass {
    public static func confirmationAnswer(_ raw: String) -> Bool? {
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.range(of: "^(응|네|예|그래|좋아|오케이|알겠어|맞아)([ ,.!~]*(등록|추가|진행|승인|완료|변경|삭제|반영)?(해(줘|요)?)?)?[.!~ ]*$|^(등록|추가|진행|승인|완료|변경|삭제|반영)(해(줘|요)?)?[.!~ ]*$", options: .regularExpression) != nil {
            return true
        }
        if text.range(of: "^(아니(요)?|싫어)([ ,.!~]*(거절|취소)(해(줘|요)?)?|[ ,.!~]*(하지[ ]?마(세요)?|안[ ]?해|그만둬))?[.!~ ]*$|^(거절(해(줘|요)?)?|취소(해(줘|요)?)?|하지[ ]?마(세요)?|안[ ]?해|그만둬)[.!~ ]*$", options: .regularExpression) != nil {
            return false
        }
        return nil
    }

    public static func match(_ raw: String) -> RuleBypassAction? {
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return nil }

        // 1. Stop speaking / cancel
        if text.range(of: "^(그만[ ]?말해|말[ ]?그만|멈춰|조용히[ ]?해|스톱|stop)[.! ]*$", options: [.regularExpression, .caseInsensitive]) != nil {
            return .stopSpeaking
        }

        // 2. Repeat last reply
        if text.range(of: "(다시[ ]?말해(줘|요)?|뭐라고[ ]?했(어|어요|지)|방금[ ]?뭐라)", options: .regularExpression) != nil {
            return .repeatLastReply
        }

        // 3. Reminder acknowledge / snooze
        if text.range(of: "(알림|알람)[ ]?(꺼(줘|요)?|확인|끄기|지워)", options: .regularExpression) != nil {
            return .acknowledgeReminder
        }
        if text.range(of: "(나중에[ ]?다시[ ]?알려|나중에[ ]?알려|알림[ ]?미뤄)", options: .regularExpression) != nil {
            let secs = parseDurationSeconds(text) ?? 600
            return .snoozeReminder(seconds: secs)
        }

        // 4. Local timer: cancel / query / set
        if text.range(of: "(타이머|타임머).*(취소|꺼|끄|중지|삭제|없애)", options: .regularExpression) != nil {
            return .cancelTimer
        }
        if text.range(of: "(타이머|타임머).*(남은|남았|얼마나|얼마[ ]?남|몇[ ]?(분|초)|언제[ ]?끝|확인|상태)", options: .regularExpression) != nil {
            return .queryTimer
        }
        if let timerRange = text.range(of: "타이머|타임머", options: .regularExpression) {
            let afterTimer = String(text[timerRange.upperBound...])
            if let seconds = parseDurationSeconds(afterTimer), seconds > 0, seconds <= 86_400 {
                return .setTimer(seconds: seconds, label: formatDuration(seconds))
            }
        }
        if text.range(of: "(뒤에[ ]?알려|후에[ ]?알려)", options: .regularExpression) != nil,
           let seconds = parseDurationSeconds(text), seconds > 0, seconds <= 86_400 {
            return .setTimer(seconds: seconds, label: formatDuration(seconds))
        }

        // 5. Concrete Eureka filtered queries
        if text.range(of: "(오늘[ ]?브리핑|하루[ ]?요약|오늘[ ]?일정[ ]?정리|아침[ ]?브리핑)", options: .regularExpression) != nil {
            return .eurekaFilter("briefing")
        }
        if text.range(of: "(오늘[ ]?(할[ ]?일|마감|끝낼|끝내야)|오늘까지[ ]?(할|마감|업무))", options: .regularExpression) != nil {
            return .eurekaFilter("today")
        }
        if text.range(of: "(내일[ ]?(할[ ]?일|마감|끝낼|끝내야)|내일까지[ ]?(할|마감|업무))", options: .regularExpression) != nil {
            return .eurekaFilter("tomorrow")
        }
        if text.range(of: "(이번[ ]?주[ ]?(할[ ]?일|마감|업무))", options: .regularExpression) != nil {
            return .eurekaFilter("this_week")
        }
        if text.range(of: "(밀린[ ]?(업무|일)|마감[ ]?지난|지연된[ ]?(업무|일)|늦은[ ]?(업무|일))", options: .regularExpression) != nil {
            return .eurekaFilter("overdue")
        }
        if text.range(of: "(급한[ ]?(일|업무)|긴급[ ]?(업무|일)|우선순위[ ]?높은)", options: .regularExpression) != nil {
            return .eurekaFilter("urgent")
        }

        return nil
    }

    public static func parseDurationSeconds(_ text: String) -> Int? {
        let numberPattern = "(?:[0-9]+|(?:(?:스물|열)\\s*)?(?:하나|한|둘|두|셋|세|넷|네|다섯|여섯|일곱|여덟|아홉)|스물|스무|열|(?:[이삼사오육칠팔구]?백\\s*)?(?:[이삼사오육칠팔구]?십\\s*)?[일이삼사오육칠팔구]|(?:[이삼사오육칠팔구]?백\\s*)?[이삼사오육칠팔구]?십|[이삼사오육칠팔구]?백)"
        let pattern = "(\(numberPattern))\\s*(시간|분|초)(\\s*반)?"
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let ns = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: ns.length))
        guard !matches.isEmpty else { return nil }
        var total = 0
        for m in matches {
            guard let val = parseKoreanNumber(ns.substring(with: m.range(at: 1))) else { continue }
            let unit = ns.substring(with: m.range(at: 2))
            let hasHalf = m.range(at: 3).location != NSNotFound
            switch unit {
            case "시간":
                total += val * 3600 + (hasHalf ? 1800 : 0)
            case "분":
                total += val * 60 + (hasHalf ? 30 : 0)
            case "초":
                total += val
            default:
                break
            }
        }
        return total > 0 ? total : nil
    }

    private static func parseKoreanNumber(_ raw: String) -> Int? {
        let s = String(raw.filter { !$0.isWhitespace })
        if let num = Int(s) {
            return num
        }

        let nativeUnits: [String: Int] = [
            "하나": 1, "한": 1,
            "둘": 2, "두": 2,
            "셋": 3, "세": 3,
            "넷": 4, "네": 4,
            "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9
        ]
        if s == "스물" || s == "스무" { return 20 }
        if s == "열" { return 10 }
        if s.hasPrefix("스물"), let u = nativeUnits[String(s.dropFirst(2))] { return 20 + u }
        if s.hasPrefix("열"), let u = nativeUnits[String(s.dropFirst(1))] { return 10 + u }
        if let u = nativeUnits[s] { return u }

        let sinoDigits: [Character: Int] = [
            "일": 1, "이": 2, "삼": 3, "사": 4, "오": 5,
            "육": 6, "칠": 7, "팔": 8, "구": 9
        ]
        var rem = s
        var total = 0
        var matched = false

        if let idx = rem.firstIndex(of: "백") {
            let prefix = String(rem[..<idx])
            let digit: Int?
            if prefix.isEmpty {
                digit = 1
            } else if prefix.count == 1, let first = prefix.first {
                digit = sinoDigits[first]
            } else {
                digit = nil
            }
            guard let d = digit else { return nil }
            total += d * 100
            rem = String(rem[rem.index(after: idx)...])
            matched = true
        }

        if let idx = rem.firstIndex(of: "십") {
            let prefix = String(rem[..<idx])
            let digit: Int?
            if prefix.isEmpty {
                digit = 1
            } else if prefix.count == 1, let first = prefix.first {
                digit = sinoDigits[first]
            } else {
                digit = nil
            }
            guard let d = digit else { return nil }
            total += d * 10
            rem = String(rem[rem.index(after: idx)...])
            matched = true
        }

        if !rem.isEmpty {
            guard rem.count == 1, let first = rem.first, let u = sinoDigits[first] else {
                return nil
            }
            total += u
            matched = true
        }

        return matched ? total : nil
    }

    public static func formatDuration(_ seconds: Int) -> String {
        let h = seconds / 3600
        let m = (seconds % 3600) / 60
        let s = seconds % 60
        var parts: [String] = []
        if h > 0 { parts.append("\(h)시간") }
        if m > 0 { parts.append("\(m)분") }
        if s > 0 || parts.isEmpty { parts.append("\(s)초") }
        return parts.joined(separator: " ")
    }
}
