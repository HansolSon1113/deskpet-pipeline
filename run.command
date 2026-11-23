#!/bin/sh
set -eu
cd -P "$(dirname "$0")"
export PATH="$PATH:/opt/homebrew/bin:/usr/local/bin:$HOME/Library/pnpm"

low_memory=0
has_help=0
has_config=0
for arg in "$@"; do
  case "$arg" in
    --low-memory) low_memory=1 ;;
    --help) has_help=1 ;;
    --config) has_config=1 ;;
  esac
done

export CLANG_MODULE_CACHE_PATH="$PWD/.build/ModuleCache"
export SWIFTPM_MODULECACHE_OVERRIDE="$PWD/.build/ModuleCache"
sh tools/prepare-workspace.sh

fetch_file() {
  url=$1
  dest=$2
  if [ -f "$dest" ] && [ -s "$dest" ]; then
    return 0
  fi
  mkdir -p "$(dirname "$dest")"
  echo "[run.command] Downloading $(basename "$dest") ..."
  curl --fail --location --retry 3 "$url" --output "$dest.part"
  mv "$dest.part" "$dest"
}

ensure_models() {
  if [ "$has_help" = "1" ] || [ "$has_config" = "1" ] || [ ! -d "Models" ]; then
    return 0
  fi

  fetch_file "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/f6d5376be1edb4d416d56da11e5397a961aca8ae/Qwen3.5-2B-Q4_0.gguf" "Models/output/qwen-2b-q4_0.gguf"
  fetch_file "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx" "Models/vad/silero_vad.onnx"
  fetch_file "https://huggingface.co/daekeun-ml/koelectra-small-v3-nsmc/resolve/d1c0233fc013ae8bd879d7b61f2bec69aa3af4c9/tokenizer.json" "Models/emotion/tokenizer.json"

  if [ "$low_memory" = "1" ]; then
    stt_dir="Models/stt/icefall-asr-ko-streaming-zipformer-174m"
    if [ ! -f "$stt_dir/tokens.txt" ]; then
      base_url="https://huggingface.co/kangkyu/icefall-asr-ko-streaming-zipformer-174m/resolve/main"
      fetch_file "$base_url/tokens.txt" "$stt_dir/tokens.txt"
      fetch_file "$base_url/encoder-epoch-99-avg-1-chunk-64-left-128.int8.onnx" "$stt_dir/encoder-epoch-99-avg-1-chunk-64-left-128.int8.onnx"
      fetch_file "$base_url/decoder-epoch-99-avg-1-chunk-64-left-128.int8.onnx" "$stt_dir/decoder-epoch-99-avg-1-chunk-64-left-128.int8.onnx"
      fetch_file "$base_url/joiner-epoch-99-avg-1-chunk-64-left-128.int8.onnx" "$stt_dir/joiner-epoch-99-avg-1-chunk-64-left-128.int8.onnx"
    fi
    tts_dir="Models/tts/melotts-kr-onnx"
    if [ ! -f "$tts_dir/melotts_kr_int8.onnx" ] || [ ! -f "$tts_dir/symbols.json" ]; then
      base_url="https://huggingface.co/gnyong/melotts-kr-onnx/resolve/main"
      fetch_file "$base_url/symbols.json" "$tts_dir/symbols.json"
      fetch_file "$base_url/melotts_kr_int8.onnx" "$tts_dir/melotts_kr_int8.onnx"
    fi
  else
    stt_dir="Models/stt/sherpa-onnx-zipformer-korean-2024-06-24"
    if [ ! -f "$stt_dir/tokens.txt" ]; then
      archive=".build/sherpa-onnx-zipformer-korean-2024-06-24.tar.bz2"
      mkdir -p ".build" "Models/stt"
      fetch_file "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-korean-2024-06-24.tar.bz2" "$archive"
      tar -xjf "$archive" -C "Models/stt"
      rm -f "$archive"
    fi
    tts_dir="Models/tts/qwen3-tts-12hz-0.6b-customvoice"
    if [ ! -f "$tts_dir/config.json" ] && [ -f "Qwen3-Models.zip" ]; then
      echo "[run.command] Extracting $tts_dir from Qwen3-Models.zip ..."
      unzip -oq "Qwen3-Models.zip" "Models/tts/qwen3-tts-12hz-0.6b-customvoice/*" -d "."
    fi
  fi
}

ensure_models

if [ "$has_help" = "0" ] && [ "$has_config" = "0" ] && [ -d "Models" ]; then
  if [ "$low_memory" = "1" ]; then
    DESKPET_LOW_MEMORY=1 swift tools/configure-mac-json.swift --low-memory
  else
    swift tools/configure-mac-json.swift
  fi
fi

swift build --product deskpet-demo
exec .build/debug/deskpet-demo --project "$@"
