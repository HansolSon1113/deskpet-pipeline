#!/bin/sh
# Linux / Jetson Orin Nano entry point (leaves run.command untouched).
set -eu
cd -P "$(dirname "$0")"
export PATH="$PATH:/usr/local/cuda/bin:/usr/local/bin:$HOME/.local/share/pnpm:$HOME/Library/pnpm"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$PWD/.build/native:$PWD/.build/native/llama/src:$PWD/.build/native/llama/ggml/src:$PWD/.build/native/llama/ggml/src/ggml-cuda:$PWD/.build/native/whisper/src:$PWD/.build/native-deps/sherpa/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu"
sh tools/prepare-workspace.sh
export CLANG_MODULE_CACHE_PATH="$PWD/.build/ModuleCache"
export SWIFTPM_MODULECACHE_OVERRIDE="$PWD/.build/ModuleCache"
swift build --product deskpet-demo
exec .build/debug/deskpet-demo --jetson "$@"
