#!/bin/sh
set -eu
cd -P "$(dirname "$0")"
export PATH="$PATH:/opt/homebrew/bin:/usr/local/bin:$HOME/Library/pnpm"
sh tools/prepare-workspace.sh
export CLANG_MODULE_CACHE_PATH="$PWD/.build/ModuleCache"
export SWIFTPM_MODULECACHE_OVERRIDE="$PWD/.build/ModuleCache"
swift build --product deskpet-demo
exec .build/debug/deskpet-demo --project "$@"
