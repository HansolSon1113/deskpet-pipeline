import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Provides lightweight live context (local time/date and cached weather) for OutputModel only.
public actor OutputContextProvider {
    private var weatherSummary: String?
    private var refreshTask: Task<Void, Never>?
    private let latitude: Double
    private let longitude: Double
    private let timeZone: TimeZone

    public init(latitude: Double = 37.5665, longitude: Double = 126.9780, timeZoneIdentifier: String = "Asia/Seoul") {
        self.latitude = latitude
        self.longitude = longitude
        self.timeZone = TimeZone(identifier: timeZoneIdentifier) ?? .current
    }

    public func startWeatherRefresh() {
        guard refreshTask == nil else { return }
        refreshTask = Task { [weak self] in
            while !Task.isCancelled {
                await self?.fetchWeatherOnce()
                try? await Task.sleep(for: .seconds(1800))
            }
        }
    }

    public func stop() {
        refreshTask?.cancel()
        refreshTask = nil
    }

    public func currentContextLine(now: Date = Date()) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "ko_KR")
        formatter.timeZone = timeZone
        formatter.dateFormat = "yyyy년 M월 d일 EEEE HH시 mm분"
        var line = "현재 시각: \(formatter.string(from: now))"
        if let weatherSummary, !weatherSummary.isEmpty {
            line += ", 현재 날씨: \(weatherSummary)"
        }
        return line
    }

    private func fetchWeatherOnce() async {
        let urlString = "https://api.open-meteo.com/v1/forecast?latitude=\(latitude)&longitude=\(longitude)&current=temperature_2m,weather_code&timezone=Asia%2FSeoul"
        guard let url = URL(string: urlString) else { return }
        var request = URLRequest(url: url)
        request.timeoutInterval = 4
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard (response as? HTTPURLResponse)?.statusCode == 200,
                  let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let current = json["current"] as? [String: Any],
                  let temp = current["temperature_2m"] as? Double else { return }
            let code = current["weather_code"] as? Int ?? -1
            let desc = Self.weatherDescription(code)
            weatherSummary = "\(desc) \(Int(temp.rounded()))도"
        } catch {
            // Offline or transient failure: keep previous cached value or omit weather.
        }
    }

    private static func weatherDescription(_ code: Int) -> String {
        switch code {
        case 0: return "맑음"
        case 1, 2: return "구름 조금"
        case 3: return "흐림"
        case 45, 48: return "안개"
        case 51, 53, 55, 56, 57: return "이슬비"
        case 61, 63, 65, 66, 67, 80, 81, 82: return "비"
        case 71, 73, 75, 77, 85, 86: return "눈"
        case 95, 96, 99: return "천둥번개"
        default: return "보통"
        }
    }
}
