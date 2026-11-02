#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
fetch() {
  mkdir -p "$(dirname "$2")"
  [ -f "$2" ] && return
  curl --fail --location --retry 3 "$1" --output "$2.part"
  mv "$2.part" "$2"
}
fetch https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF/resolve/807dfd9a6fadbf703207aac3bfb0b2b3040d077c/Qwen3.5-0.8B-BF16.gguf Models/router/backbone-bf16.gguf
fetch https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/2fc06364715b967f1860aea9cf38778875588b17/config.json Models/router/base_config.json
fetch https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/2fc06364715b967f1860aea9cf38778875588b17/model.safetensors-00001-of-00001.safetensors Models/router/base/model.safetensors
fetch https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/f6d5376be1edb4d416d56da11e5397a961aca8ae/Qwen3.5-2B-Q4_0.gguf Models/output/qwen-2b-q4_0.gguf
fetch https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-small-q8_0.bin Models/stt/ggml-small-q8_0.bin
fetch https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx Models/vad/silero_vad.onnx
for file in model.safetensors config.json tokenizer.json; do
  fetch "https://huggingface.co/daekeun-ml/koelectra-small-v3-nsmc/resolve/d1c0233fc013ae8bd879d7b61f2bec69aa3af4c9/$file" "Models/emotion/$file"
done
tts=sherpa-onnx-supertonic-3-tts-int8-2026-05-11
if [ ! -f "Models/tts/$tts/voice.bin" ]; then
  fetch "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/$tts.tar.bz2" .build/tts.tar.bz2
  mkdir -p Models/tts
  tar -xjf .build/tts.tar.bz2 -C Models/tts
fi
echo 'Public model assets ready. Existing router checkpoint/tokenizer/config are required separately.'
