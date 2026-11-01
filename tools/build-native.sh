#!/bin/sh
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
case "$(uname -s)" in
  Darwin) backend='-DGGML_METAL=ON -DGGML_CUDA=OFF' ;;
  Linux) backend='-DGGML_METAL=OFF -DGGML_CUDA=ON' ;;
  *) echo 'Unsupported native platform' >&2; exit 1 ;;
esac
cmake -S native -B .build/native -DCMAKE_BUILD_TYPE=Release $backend \
  -DSHERPA_ROOT="${SHERPA_ROOT:-$PWD/.build/native-deps/sherpa}" "$@"
cmake --build .build/native --target deskpet_inference -j "${DESKPET_BUILD_JOBS:-4}"
echo "Native backend: $root/.build/native/libdeskpet_inference"
