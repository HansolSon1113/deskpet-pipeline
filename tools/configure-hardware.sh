#!/bin/sh
# Prepare the existing hardware checkout. No inference or device access occurs.
set -eu
hardware_dir=$1
build_dir=$2
sh "$(dirname "$0")/prepare-cmake-build.sh" "$hardware_dir" "$build_dir"
case "${3:-}" in
  --headless)
    exec cmake -S "$hardware_dir" -B "$build_dir" \
      -DDESKPET_LINUX=OFF -DBUILD_TESTING=OFF -DDESKPET_SDL=OFF
    ;;
  '') ;;
  *) echo 'Unknown display option' >&2; exit 1 ;;
esac

if [ "$(uname -s)" = Darwin ]; then
  if ! command -v brew >/dev/null 2>&1; then
    echo '모의 화면에 SDL2와 SDL2_ttf가 필요합니다. Homebrew 설치 후 다시 실행하세요.' >&2
    exit 1
  fi
  sdl_prefix=$(brew --prefix sdl2)
  ttf_prefix=$(brew --prefix sdl2_ttf)
  if [ ! -d "$sdl_prefix/lib/cmake/SDL2" ] || [ ! -d "$ttf_prefix/lib/cmake/SDL2_ttf" ]; then
    echo '모의 화면 개발 패키지 준비: SDL2 / SDL2_ttf'
    HOMEBREW_NO_AUTO_UPDATE=1 brew install sdl2 sdl2_ttf
  fi
  # Homebrew's prefix is not always in CMake's search paths, particularly when
  # the entry point was opened from Finder. Do not rely on PATH for discovery.
  exec cmake -S "$hardware_dir" -B "$build_dir" \
    -DDESKPET_LINUX=OFF -DBUILD_TESTING=OFF -DDESKPET_SDL=ON \
    "-DSDL2_DIR=$sdl_prefix/lib/cmake/SDL2" \
    "-DSDL2_ttf_DIR=$ttf_prefix/lib/cmake/SDL2_ttf"
fi

exec cmake -S "$hardware_dir" -B "$build_dir" \
  -DDESKPET_LINUX=OFF -DBUILD_TESTING=OFF -DDESKPET_SDL=ON
