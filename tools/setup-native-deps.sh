#!/bin/sh
# Native SDKs only; no Python environment or model server is created.
set -eu
cd "$(dirname "$0")/.."
[ "$(uname -s)-$(uname -m)" = Darwin-arm64 ] || { echo 'Supply SHERPA_ROOT and the platform LibTorch SDK on this platform.' >&2; exit 1; }
mkdir -p .build/native-deps
download() {
  [ -f "$2" ] && return
  curl --fail --location --retry 3 "$1" --output "$2.part"
  mv "$2.part" "$2"
}
sherpa=sherpa-onnx-v2.0.1-osx-arm64-shared-lib
if [ ! -f .build/native-deps/sherpa/lib/libsherpa-onnx-c-api.dylib ]; then
  download "https://github.com/k2-fsa/sherpa-onnx/releases/download/v2.0.1/$sherpa.tar.bz2" .build/sherpa.tar.bz2
  tar -xjf .build/sherpa.tar.bz2 -C .build
  mkdir -p .build/native-deps/sherpa/lib
  cp ".build/$sherpa/lib/"*.dylib .build/native-deps/sherpa/lib/
fi
include=.build/native-deps/sherpa/include
mkdir -p "$include/sherpa-onnx/c-api" "$include/onnxruntime/core/session"
download https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/v2.0.1/sherpa-onnx/c-api/c-api.h "$include/sherpa-onnx/c-api/c-api.h"
for header in onnxruntime_c_api.h onnxruntime_cxx_api.h onnxruntime_cxx_inline.h onnxruntime_float16.h onnxruntime_error_code.h onnxruntime_ep_c_api.h; do
  download "https://raw.githubusercontent.com/microsoft/onnxruntime/v1.28.2/include/onnxruntime/core/session/$header" "$include/onnxruntime/core/session/$header"
done
if [ ! -d .build/native-deps/libtorch ]; then
  download https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.11.0.zip .build/libtorch.zip
  unzip -q .build/libtorch.zip -d .build/native-deps
fi
echo 'Native SDKs ready. LibTorch is used only by deskpet-model-prep.'
