#!/bin/sh
# Linux (Jetson Orin Nano aarch64 / x86_64) native dependency setup.
# Existing macOS setup-native-deps.sh is left untouched.
set -eu
cd "$(dirname "$0")/.."

if [ "${1:-}" = "--install-apt" ]; then
  sudo apt-get update
  sudo apt-get install -y \
    build-essential cmake pkg-config git curl unzip bzip2 \
    libsdl2-dev libsdl2-ttf-dev libasound2-dev libsoxr-dev \
    libcurl4-openssl-dev libxml2-dev
fi

os_arch="$(uname -s)-$(uname -m)"
case "$os_arch" in
  Linux-aarch64|Linux-arm64)
    sherpa_candidates="sherpa-onnx-v2.0.1-linux-aarch64-shared-cpu sherpa-onnx-v2.0.1-linux-aarch64-shared-lib sherpa-onnx-v2.0.1-linux-aarch64-shared"
    ;;
  Linux-x86_64)
    sherpa_candidates="sherpa-onnx-v2.0.1-linux-x64-shared-cpu sherpa-onnx-v2.0.1-linux-x64-shared-lib sherpa-onnx-v2.0.1-linux-x64-shared"
    ;;
  *)
    echo "This script targets Linux (aarch64/x86_64). Current platform: $os_arch" >&2
    exit 1
    ;;
esac

mkdir -p .build/native-deps
download() {
  [ -f "$2" ] && return
  curl --fail --location --retry 3 "$1" --output "$2.part"
  mv "$2.part" "$2"
}

if [ ! -f .build/native-deps/sherpa/lib/libsherpa-onnx-c-api.so ]; then
  downloaded_dir=""
  for candidate in $sherpa_candidates; do
    url="https://github.com/k2-fsa/sherpa-onnx/releases/download/v2.0.1/$candidate.tar.bz2"
    if curl --fail --location --retry 2 "$url" --output .build/sherpa-linux.tar.bz2.part 2>/dev/null; then
      mv .build/sherpa-linux.tar.bz2.part .build/sherpa-linux.tar.bz2
      downloaded_dir="$candidate"
      break
    fi
  done
  if [ -z "$downloaded_dir" ]; then
    echo "Failed to download sherpa-onnx v2.0.1 archive for $os_arch. Set SHERPA_ROOT manually." >&2
    exit 1
  fi
  tar -xjf .build/sherpa-linux.tar.bz2 -C .build
  mkdir -p .build/native-deps/sherpa/lib
  cp -a ".build/$downloaded_dir/lib/"*.so* .build/native-deps/sherpa/lib/
fi

include=.build/native-deps/sherpa/include
mkdir -p "$include/sherpa-onnx/c-api" "$include/onnxruntime/core/session"
download https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/v2.0.1/sherpa-onnx/c-api/c-api.h "$include/sherpa-onnx/c-api/c-api.h"
for header in onnxruntime_c_api.h onnxruntime_cxx_api.h onnxruntime_cxx_inline.h onnxruntime_float16.h onnxruntime_error_code.h onnxruntime_ep_c_api.h; do
  download "https://raw.githubusercontent.com/microsoft/onnxruntime/v1.28.2/include/onnxruntime/core/session/$header" "$include/onnxruntime/core/session/$header"
done

echo "Linux native SDKs ready at $PWD/.build/native-deps/sherpa"
