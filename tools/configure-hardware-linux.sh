#!/bin/sh
# Prepare the hardware checkout on Linux / Jetson without modifying configure-hardware.sh.
set -eu
hardware_dir=$1
build_dir=$2
mode="${3:-}"
sh "$(dirname "$0")/prepare-cmake-build.sh" "$hardware_dir" "$build_dir"

case "$mode" in
  --headless)
    exec cmake -S "$hardware_dir" -B "$build_dir" \
      -DDESKPET_LINUX=OFF -DBUILD_TESTING=OFF -DDESKPET_SDL=OFF
    ;;
  --real-headless)
    exec cmake -S "$hardware_dir" -B "$build_dir" \
      -DDESKPET_LINUX=ON -DBUILD_TESTING=OFF -DDESKPET_SDL=OFF
    ;;
  --real)
    exec cmake -S "$hardware_dir" -B "$build_dir" \
      -DDESKPET_LINUX=ON -DBUILD_TESTING=OFF -DDESKPET_SDL=ON
    ;;
  '')
    exec cmake -S "$hardware_dir" -B "$build_dir" \
      -DDESKPET_LINUX=OFF -DBUILD_TESTING=OFF -DDESKPET_SDL=ON
    ;;
  *)
    echo "Unknown hardware mode: $mode" >&2
    exit 1
    ;;
esac
