#!/usr/bin/env bash
# Builds pckern.dll for Windows, copies it and the gateway sources to the
# stage folder on C:, and builds pckern.mexw64 there with MATLAB's mex.
#
# Usage:
#   spikes/stage.sh              cargo build, copy, mex
#   spikes/stage.sh --mex-only   copy the C sources and run mex only
#
# A desktop MATLAB that has loaded pckern.dll keeps the file locked, so the
# copy fails until that MATLAB exits.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(dirname "$here")"
stage="${PHASECONT_STAGE:-/mnt/c/Users/$(cmd.exe /c "echo %USERNAME%" 2>/dev/null | tr -d "\r")/phasecont-stage/spikes}"
target=x86_64-pc-windows-gnu

mex_only=0
case "${1:-}" in
  "") ;;
  --mex-only) mex_only=1 ;;
  *) echo "usage: spikes/stage.sh [--mex-only]" >&2; exit 2 ;;
esac

mkdir -p "$stage"

if (( ! mex_only )); then
  # cargo reads .cargo/config.toml (the MinGW linker) from the current
  # folder, not from the manifest's folder.
  (cd "$here" && cargo build --release --target "$target")
  dll="$here/target/$target/release/pckern.dll"
  deps="$(x86_64-w64-mingw32-objdump -p "$dll" | sed -n 's/^[[:space:]]*DLL Name: //p')"
  echo "pckern.dll imports:"
  sed 's/^/  /' <<<"$deps"
  if grep -Eiq 'libgcc|libwinpthread|libstdc' <<<"$deps"; then
    echo "stage.sh: pckern.dll needs MinGW runtime DLLs; link them statically" >&2
    exit 1
  fi
  cp "$dll" "$stage/"
fi

cp "$here/mex/pckern_gateway.c" "$here/mex/fhn_aot.c" "$stage/"
stage_win="$(wslpath -w "$stage")"
cd "$repo"
scripts/matlab-win "cd('${stage_win//\'/\'\'}'); mex('-R2018a', 'COMPFLAGS=\$COMPFLAGS /W4', '-output', 'pckern', 'pckern_gateway.c', 'fhn_aot.c')"
ls -l "$stage"
