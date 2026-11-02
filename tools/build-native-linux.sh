#!/bin/sh
# Linux / Jetson Orin Nano native backend build script.
# Leaves existing tools/build-native.sh and native/CMakeLists.txt untouched.
set -eu
cd "$(dirname "$0")/.."
sh tools/prepare-cmake-build.sh "$PWD/native" "$PWD/.build/native"
root="$PWD"
mkdir -p .build/native-src

fetch() {
  name="$1"; url="$2"; revision="$3"
  if [ ! -d ".build/native-src/$name/.git" ]; then
    git clone --no-checkout "$url" ".build/native-src/$name"
    git -C ".build/native-src/$name" checkout "$revision"
  fi
  actual=$(git -C ".build/native-src/$name" rev-parse HEAD)
  [ "$actual" = "$revision" ] || { echo "Dependency revision mismatch: $name" >&2; exit 1; }
}

fetch llama https://github.com/ggml-org/llama.cpp.git caa8191df159f764fb0160e3d2fc6be008948aee
fetch whisper https://github.com/ggml-org/whisper.cpp.git 231638a0717b7822784297e5980965ffcec05810

if [ -x /usr/local/cuda/bin/nvcc ] && [ -z "${CUDACXX:-}" ]; then
  export CUDACXX=/usr/local/cuda/bin/nvcc
  export PATH="/usr/local/cuda/bin:$PATH"
fi

# Default to SM 87 (Jetson Orin series) on aarch64 to avoid compiling kernels for all desktop architectures.
cuda_arch="${DESKPET_CUDA_ARCH:-}"
if [ -z "$cuda_arch" ] && [ "$(uname -m)" = "aarch64" ]; then
  cuda_arch="87"
fi

sherpa_root="${SHERPA_ROOT:-$PWD/.build/native-deps/sherpa}"
rpath_flags="-Wl,-rpath,\$ORIGIN -Wl,-rpath,\$ORIGIN/llama/src -Wl,-rpath,\$ORIGIN/llama/ggml/src -Wl,-rpath,\$ORIGIN/llama/ggml/src/ggml-cuda -Wl,-rpath,\$ORIGIN/whisper/src -Wl,-rpath,$sherpa_root/lib"

cmake_args="-DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF -DGGML_CUDA=ON -DSHERPA_ROOT=$sherpa_root"
if [ -n "$cuda_arch" ]; then
  cmake_args="$cmake_args -DCMAKE_CUDA_ARCHITECTURES=$cuda_arch"
fi

cmake -S native -B .build/native $cmake_args \
  "-DCMAKE_SHARED_LINKER_FLAGS=$rpath_flags" "$@"
cmake --build .build/native --target deskpet_inference -j "${DESKPET_BUILD_JOBS:-4}"
echo "Linux native backend: $root/.build/native/libdeskpet_inference.so"
