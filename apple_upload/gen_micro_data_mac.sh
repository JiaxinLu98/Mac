#!/bin/bash
# Generates the sorted u32 arrays that the micro-benchmark reads, into public/data.
# The repo's scripts/create_data.sh still refers to an older .cu generator, so this script
# compiles scripts/create_data.cpp directly with the Xcode command line tools.
#
# Usage: bash apple_upload/gen_micro_data_mac.sh [ranges] [sizes]
#   ranges  e6 (micro-benchmark, default) or "e2 e6" (e2 is also needed by the fusion runs)
#   sizes   sizes in millions (default: 1 2 4 8 16 32 64 128)
# Disk: one range of the default sizes needs about 2 GB, both ranges about 4 GB.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT/public/data"
SRC="$ROOT/scripts/create_data.cpp"
BIN="$ROOT/scripts/create_data"
RANGES="${1:-e6}"
SIZES="${2:-1 2 4 8 16 32 64 128}"

command -v c++ > /dev/null || { echo "c++ not found. Install the Xcode command line tools: xcode-select --install"; exit 1; }
mkdir -p "$OUT_DIR"

if [ ! -x "$BIN" ] || [ "$SRC" -nt "$BIN" ]; then
    echo "compiling $SRC"
    c++ -O3 -std=c++17 "$SRC" -o "$BIN"
fi

for range in $RANGES; do
    exp="${range#e}"
    r=1
    for _ in $(seq "$exp"); do r=$((r * 10)); done
    for size in $SIZES; do
        a="$OUT_DIR/A_${size}${range}.bin"
        b="$OUT_DIR/B_${size}${range}.bin"
        if [ -s "$a" ] && [ -s "$b" ]; then
            echo "skip ${size}M $range (already present)"
            continue
        fi
        echo "generating ${size}M $range (values 1..$r)"
        "$BIN" $((size * 1000000)) "$r" "$a" "$b" > /dev/null
    done
done

echo "done. $(ls -1 "$OUT_DIR" | wc -l | tr -d ' ') files in $OUT_DIR"
du -sh "$OUT_DIR"
