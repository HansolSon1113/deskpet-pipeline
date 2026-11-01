#!/bin/sh
# A relocated CMake build must be configured again; never rewrite its old paths.
set -eu
source_dir=$(cd -P "$1" && pwd)
build_dir=$2
cache="$build_dir/CMakeCache.txt"
if [ ! -f "$cache" ]; then exit 0; fi
cached_source=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$cache")
cached_build=$(sed -n 's/^CMAKE_CACHEFILE_DIR:INTERNAL=//p' "$cache")
physical_build=$(cd -P "$build_dir" && pwd)
if [ "$cached_source" != "$source_dir" ] || [ "$cached_build" != "$physical_build" ]; then
  archive_dir="$physical_build.relocated-$(date +%Y%m%d-%H%M%S)-$$"
  mv "$physical_build" "$archive_dir"
fi
