#!/bin/bash
# Micro-benchmark on the M5 Pro through wgpu-native on Metal: intersection at 1M-128M elements.
# Same protocol as the A100 and the RTX 3060: 1000 ms of unrelated GPU work before every size,
# then 10 warmup and 100 timed runs, repeated over LAUNCHES processes, with every timed run printed
# in a [micro-result] JSON line. GPU time comes from timestamp queries, so no CPU overhead is included.
#
# The 128M size is also the forward-progress check: its Lookback dispatch needs more than 65,535
# workgroups, so it runs on a two-dimensional grid, the case that used to deadlock on the M4 Pro.
#
# Usage: bash apple_upload/run_micro_mac.sh
#   DATA       directory with A_<n>e6.bin and B_<n>e6.bin (default: public/data)
#   RANGES     value ranges to run (default: e6)
#   LAUNCHES   processes per range (default: 3)
#   PREHEAT_MS unrelated GPU work before each size (default: 1000)
#   OUT        log directory (default: apple_logs)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${DATA:-$ROOT/public/data}"
RANGES="${RANGES:-e6}"
LAUNCHES="${LAUNCHES:-3}"
PREHEAT_MS="${PREHEAT_MS:-1000}"
OUT="${OUT:-$ROOT/apple_logs}"
BUILD="$ROOT/set_intersection_cpp/build_mac"
DATE="$(date +%F)"
mkdir -p "$OUT"

command -v cmake > /dev/null || { echo "cmake not found: brew install cmake"; exit 1; }
for r in $RANGES; do
    [ -s "$DATA/A_1$r.bin" ] || { echo "missing $DATA/A_1$r.bin. Run: bash apple_upload/gen_micro_data_mac.sh \"$RANGES\""; exit 1; }
done

cmake -S "$ROOT/set_intersection_cpp" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DFETCHCONTENT_BASE_DIR="$BUILD/_deps"
cmake --build "$BUILD" -j

system_profiler SPDisplaysDataType > "$OUT/mac_gpu_$DATE.txt" 2>&1 || true
sw_vers > "$OUT/mac_os_$DATE.txt" 2>&1 || true

for r in $RANGES; do
    for launch in $(seq 1 "$LAUNCHES"); do
        log="$OUT/m5pro_wgpu_micro_${r}_r${launch}_$DATE.log"
        echo "=== range $r, launch $launch of $LAUNCHES -> $(basename "$log")"
        (cd "$BUILD" && MICRO_PREHEAT_MS="$PREHEAT_MS" ./App "$DATA" "$r") 2>&1 | tee "$log"
        grep -c '^\[micro-result\]' "$log" | awk -v l="$launch" '{print "  launch " l ": " $1 " sizes recorded"}'
    done
done

echo "done. logs in $OUT"
