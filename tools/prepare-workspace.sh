#!/bin/sh
# Build metadata contains absolute paths. Keep models, SDKs and logs on relocation.
set -eu
cd -P "$(dirname "$0")/.."
workspace_root=$PWD
mkdir -p .build
stamp=.build/workspace-root
previous_root=
if [ -f "$stamp" ]; then previous_root=$(cat "$stamp"); fi

if [ "$previous_root" != "$workspace_root" ]; then
  archive_dir=".build/relocated/$(date +%Y%m%d-%H%M%S)-$$/swift"
  archived=false
  for entry in .build/ModuleCache .build/*-apple-* .build/*-unknown-* \
    .build/debug .build/release .build/build.db .build/build.db-journal \
    .build/debug.yaml .build/release.yaml .build/plugin-tools.yaml \
    .build/workspace-state.json; do
    if [ -e "$entry" ] || [ -L "$entry" ]; then
      mkdir -p "$archive_dir"
      mv "$entry" "$archive_dir/"
      archived=true
    fi
  done
  printf '%s\n' "$workspace_root" > "$stamp"
fi

sh tools/prepare-cmake-build.sh "$workspace_root/native" "$workspace_root/.build/native"
