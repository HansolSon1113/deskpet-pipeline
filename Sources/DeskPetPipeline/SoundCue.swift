import Foundation
#if os(macOS)
@preconcurrency import AVFoundation
#endif

public enum SoundCue: Sendable {
    case listenStart
    case listenEnd
}

public protocol SoundCuePlayer: Sendable {
    func play(_ cue: SoundCue) async
}

#if os(macOS)
/// Plays short, low-volume non-blocking audio cues independently of MacSpeakerAdapter
/// so microphone playbackGate is never triggered at speech start.
public final class MacSoundCuePlayer: SoundCuePlayer, @unchecked Sendable {
    private let lock = NSLock()
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let format: AVAudioFormat
    private var started = false

    public init() {
        format = AVAudioFormat(standardFormatWithSampleRate: 24_000, channels: 1)!
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)
        engine.mainMixerNode.outputVolume = 0.2
    }

    public func play(_ cue: SoundCue) async {
        let buffer = makeTone(cue)
        lock.withLock {
            if !started {
                try? engine.start()
                started = engine.isRunning
            }
            guard started else { return }
            player.scheduleBuffer(buffer, at: nil, options: .interrupts, completionHandler: nil)
            if !player.isPlaying { player.play() }
        }
    }

    private func makeTone(_ cue: SoundCue) -> AVAudioPCMBuffer {
        let sampleRate = 24_000.0
        let duration = 0.06
        let frames = Int(sampleRate * duration)
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames))!
        buffer.frameLength = AVAudioFrameCount(frames)
        let samples = buffer.floatChannelData![0]
        let (f0, f1): (Double, Double) = cue == .listenStart ? (660.0, 880.0) : (880.0, 660.0)
        var phase = 0.0
        for i in 0..<frames {
            let t = Double(i) / Double(frames)
            let freq = f0 + (f1 - f0) * t
            phase += 2.0 * .pi * freq / sampleRate
            let env = sin(.pi * t)
            samples[i] = Float(sin(phase) * env * 0.35)
        }
        return buffer
    }

    deinit {
        player.stop()
        engine.stop()
    }
}
#endif
