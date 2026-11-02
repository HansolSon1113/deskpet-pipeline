#!/bin/sh
set -eu
exec "$(dirname "$0")/run.command" --keyboard "$@"
