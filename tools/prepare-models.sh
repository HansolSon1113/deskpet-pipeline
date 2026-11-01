#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
prep=.build/native/deskpet-model-prep
[ -x "$prep" ] || { echo 'Build with DESKPET_PREP=ON and the LibTorch C++ SDK first.' >&2; exit 1; }
# The official arm64 LibTorch SDK references OpenMP outside its own rpath.
if [ "$(uname -s)" = Darwin ]; then
  omp=${DESKPET_OPENMP_LIB:-/opt/homebrew/opt/libomp/lib}
  export DYLD_LIBRARY_PATH="$omp${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi
prepare() {
  destination=$1; shift
  [ -f "$destination" ] && return
  "$prep" "$@" "$destination.part"
  mv "$destination.part" "$destination"
}
prepare Models/router/heads-f16.gguf heads Models/router/heads.pt
mkdir -p Models/router/reference
prepare Models/router/reference/model.safetensors merge-reference Models/router/base/model.safetensors \
  Models/router/backbone/adapter_model.safetensors Models/router/backbone/adapter_config.json
cp Models/router/base_config.json Models/router/reference/config.json
prepare Models/router/backbone-merged-bf16.gguf merge Models/router/backbone-bf16.gguf Models/router/backbone/adapter_model.safetensors \
  Models/router/backbone/adapter_config.json Models/router/base_config.json
prepare Models/router/backbone-merged-q4_0.gguf quantize Models/router/backbone-merged-bf16.gguf
prepare Models/emotion/model.onnx emotion Models/emotion/model.safetensors Models/emotion/config.json
"$prep" audit-heads Models/router/heads.pt Models/router/heads-f16.gguf
"$prep" audit-tokenizer Models/router/tokenizer/tokenizer.json Models/router/backbone-merged-q4_0.gguf
"$prep" audit-emotion Models/emotion/model.safetensors Models/emotion/config.json Models/emotion/model.onnx
"$prep" inspect Models/router/backbone-merged-q4_0.gguf
"$prep" inspect Models/router/backbone-merged-bf16.gguf
find Models -type f \( -name '*.gguf' -o -name '*.onnx' -o -name '*config.json' -o -name 'tokenizer.json' \) -exec shasum -a 256 {} \; > Models/prepared.sha256
env CLANG_MODULE_CACHE_PATH="$PWD/.build/ModuleCache" swift tools/model-manifest.swift
echo 'Native model files prepared. Original checkpoints are unchanged.'
