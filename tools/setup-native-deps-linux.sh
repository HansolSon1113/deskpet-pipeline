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

desired_mode="gpu"
if [ "${DESKPET_USE_AUDIO_GPU:-1}" = "0" ] && [ "${SHERPA_ONNX_USE_GPU:-0}" != "1" ]; then
  echo "DESKPET_USE_AUDIO_GPU=0 is disabled: GPU-only mode requires CUDA." >&2
  exit 1
fi

os_arch="$(uname -s)-$(uname -m)"
case "$os_arch" in
  Linux-aarch64|Linux-arm64)
    sherpa_candidates="sherpa-onnx-v2.0.1-linux-aarch64-shared-gpu-onnxruntime-1.18.1 sherpa-onnx-v2.0.1-linux-aarch64-shared-gpu-onnxruntime-1.18.0"
    ;;
  Linux-x86_64)
    sherpa_candidates="sherpa-onnx-v2.0.1-linux-x64-shared-gpu-onnxruntime-1.18.1"
    ;;
  *)
    echo "This script targets Linux (aarch64/x86_64). Current platform: $os_arch" >&2
    exit 1
    ;;
esac

mkdir -p .build/native-deps/sherpa
download() {
  [ -f "$2" ] && return
  curl --fail --location --retry 3 "$1" --output "$2.part"
  mv "$2.part" "$2"
}

installed_mode=""
if [ -f .build/native-deps/sherpa/.mode ]; then
  installed_mode="$(cat .build/native-deps/sherpa/.mode)"
fi

if [ ! -f .build/native-deps/sherpa/lib/libsherpa-onnx-c-api.so ] || [ "$installed_mode" != "$desired_mode" ]; then
  rm -rf .build/native-deps/sherpa/lib .build/native-deps/sherpa/include/onnxruntime .build/sherpa-linux.tar.bz2
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
  printf '%s\n' "$downloaded_dir" > .build/native-deps/sherpa/.archive
  printf '%s' "$desired_mode" > .build/native-deps/sherpa/.mode
fi

include=.build/native-deps/sherpa/include
mkdir -p "$include/sherpa-onnx/c-api" "$include/onnxruntime/core/session"
download https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/v2.0.1/sherpa-onnx/c-api/c-api.h "$include/sherpa-onnx/c-api/c-api.h"

ort_ver="1.28.0"
ort_headers="onnxruntime_c_api.h onnxruntime_cxx_api.h onnxruntime_cxx_inline.h onnxruntime_float16.h onnxruntime_error_code.h onnxruntime_ep_c_api.h"
archive_name=""
if [ -f .build/native-deps/sherpa/.archive ]; then
  archive_name="$(cat .build/native-deps/sherpa/.archive)"
fi
case "$archive_name" in
  *onnxruntime-1.18.1*)
    ort_ver="1.18.1"
    ort_headers="onnxruntime_c_api.h onnxruntime_cxx_api.h onnxruntime_cxx_inline.h onnxruntime_float16.h"
    ;;
  *onnxruntime-1.18.0*)
    ort_ver="1.18.0"
    ort_headers="onnxruntime_c_api.h onnxruntime_cxx_api.h onnxruntime_cxx_inline.h onnxruntime_float16.h"
    ;;
esac

missing_ort_header=0
for header in $ort_headers; do
  if [ ! -f "$include/onnxruntime/core/session/$header" ]; then
    missing_ort_header=1
    break
  fi
done

if [ "$missing_ort_header" = "1" ]; then
  nupkg=".build/ort-$ort_ver.nupkg"
  download "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/$ort_ver/microsoft.ml.onnxruntime.$ort_ver.nupkg" "$nupkg"
  if command -v unzip >/dev/null 2>&1; then
    for header in $ort_headers; do
      unzip -oqj "$nupkg" "build/native/include/$header" -d "$include/onnxruntime/core/session"
    done
  elif command -v python3 >/dev/null 2>&1; then
    python3 - "$nupkg" "$include/onnxruntime/core/session" $ort_headers <<'PY'
import os, sys, zipfile
nupkg, dest_dir, *headers = sys.argv[1:]
with zipfile.ZipFile(nupkg) as zf:
    for h in headers:
        with open(os.path.join(dest_dir, h), "wb") as out:
            out.write(zf.read(f"build/native/include/{h}"))
PY
  else
    echo "Neither unzip nor python3 is available to extract onnxruntime headers from $nupkg" >&2
    exit 1
  fi
  rm -f "$nupkg"
fi

echo "Linux native SDKs ready at $PWD/.build/native-deps/sherpa"
