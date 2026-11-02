import Foundation
import CryptoKit

// Local build evidence. Checkpoints, this manifest, and hashes stay under the
// ignored Models directory; secrets and machine-specific absolute paths do not.
let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("Models")
func artifact(_ relative: String) throws -> [String: Any] {
    let url = root.appendingPathComponent(relative)
    let file = try FileHandle(forReadingFrom: url); defer { try? file.close() }
    var hash = SHA256(); var count = 0
    while let bytes = try file.read(upToCount: 8*1024*1024), !bytes.isEmpty { hash.update(data: bytes); count += bytes.count }
    return ["path": relative, "sha256": hash.finalize().map { String(format: "%02x", $0) }.joined(), "bytes": count]
}
func config(_ path: String) throws -> Any { try JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent(path))) }
let router = try config("router/router_config.json") as! [String: Any]
let tts = "tts/sherpa-onnx-supertonic-3-tts-int8-2026-05-11/"
let inputs = ["router/heads.pt", "router/backbone/adapter_model.safetensors", "router/backbone/adapter_config.json", "router/base_config.json", "router/base/model.safetensors",
              "router/tokenizer/tokenizer.json", "router/tokenizer/tokenizer_config.json", "router/tokenizer/chat_template.jinja", "router/router_config.json", "router/split_indices.json"]
let manifest: [String: Any] = [
    "schemaVersion": 1,
    "backend": "llama.cpp-metal",
    "sources": [
        "routerBase": ["model": "Qwen/Qwen3.5-0.8B", "revision": "2fc06364715b967f1860aea9cf38778875588b17"],
        "routerGGUFBase": ["model": "ggml-org/Qwen3.5-0.8B-GGUF", "revision": "807dfd9a6fadbf703207aac3bfb0b2b3040d077c"],
        "output": ["model": "unsloth/Qwen3.5-2B-GGUF", "revision": "f6d5376be1edb4d416d56da11e5397a961aca8ae"],
        "stt": ["model": "ggerganov/whisper.cpp", "revision": "5359861c739e955e79d9a303bcbc70fb988958b1"],
        "emotion": ["model": "daekeun-ml/koelectra-small-v3-nsmc", "revision": "d1c0233fc013ae8bd879d7b61f2bec69aa3af4c9"],
        "tts": ["model": "sherpa-onnx-supertonic-3-tts-int8-2026-05-11", "release": "tts-models"],
        "vad": ["model": "silero_vad.onnx", "release": "sherpa-onnx/asr-models"]
    ],
    "runtimeRevisions": ["llama.cpp": "caa8191df159f764fb0160e3d2fc6be008948aee", "whisper.cpp": "231638a0717b7822784297e5980965ffcec05810", "sherpa-onnx": "v2.0.1", "onnxruntime": "1.28.2"],
    "precision": ["routerBackbone": "Q4_0 (mixed tensor types recorded by inspect)", "routerHeads": "FP16", "output": "Q4_0", "stt": "Q8_0", "emotion": "FP32", "tts": "INT8"],
    "limits": ["routerInputTokens": router["max_input_tokens"]!, "routerGenerationTokens": router["max_generation_tokens"]!, "routerTokenCharacters": router["max_token_chars"]!, "sttPCM16MonoHz": 16000, "sttMaximumSeconds": 30, "emotionTokens": 128],
    "labels": router["labels"]!,
    "headDefinitions": router["head_definitions"]!,
    "emotionLabels": ["0": "negative", "1": "positive"],
    "inputArtifacts": try inputs.map(artifact),
    "referenceArtifacts": try ["router/reference/model.safetensors", "router/reference/config.json"].map(artifact),
    "backendArtifacts": try (["router/backbone-merged-bf16.gguf", "router/backbone-merged-q4_0.gguf", "router/heads-f16.gguf", "output/qwen-2b-q4_0.gguf", "stt/ggml-small-q8_0.bin", "vad/silero_vad.onnx", "emotion/model.onnx", "emotion/tokenizer.json"] + ["duration_predictor.int8.onnx", "text_encoder.int8.onnx", "vector_estimator.int8.onnx", "vocoder.int8.onnx", "tts.json", "unicode_indexer.bin", "voice.bin"].map { tts+$0 }).map(artifact)
]
let data = try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
try data.write(to: root.appendingPathComponent("prepared-manifest.json"), options: .atomic)
print("Model provenance and checksums: Models/prepared-manifest.json")
