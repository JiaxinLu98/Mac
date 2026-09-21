#!/bin/bash
# Follow-up runs for the Lookback split-dispatch workaround (no 128M here, so nothing in this script can hang the GPU):
#   1. cost of the split: Lookback forced into k = 1, 2, 4, 8 one-dimensional dispatches at 16M, 32M and 64M,
#      native (wgpu) and Chrome, logs in apple_logs/split/
#   2. fusion (e2: intersection + union, e6: intersection) at 1M-64M with the micro-benchmark protocol:
#      1000 ms of unrelated GPU work before each pipeline, 3 launches
#   3. two more Chrome micro-benchmark launches, so Chrome has 3 like the native runs
# Usage: bash apple_upload/run_split_mac.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$ROOT/apple_logs}"
DATE="$(date +%F)"
NV=3072
SAFE="1,2,4,8,16,32,64"
command -v cmake > /dev/null || export PATH="$HOME/Library/Python/3.9/bin:$PATH"
if [ -z "${CHROME:-}" ] && [ ! -x "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" ]; then
    export CHROME="$HOME/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
fi
mkdir -p "$OUT/split"

gpu_util() { ioreg -r -d 1 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*' | grep -o '[0-9]*$' | head -1; }
require_idle() {
    local busy=0
    for _ in 1 2 3 4 5; do [ "$(gpu_util)" -ge 50 ] && busy=$((busy + 1)); sleep 1; done
    if [ "$busy" -ge 3 ]; then echo "[stop] GPU busy ($(gpu_util)%), not measuring on it"; exit 2; fi
}

cmake --build "$ROOT/set_intersection_cpp/build_mac" -j > /dev/null

for size in 16 32 64; do
    numWg=$(( (2 * size * 1000000 + NV - 1) / NV ))
    one="$ROOT/public/data/only_$size"; mkdir -p "$one"
    for ab in A B; do ln -sf "$ROOT/public/data/${ab}_${size}e6.bin" "$one/${ab}_${size}e6.bin"; done
    for k in 1 2 4 8; do
        lbmax=$(( (numWg + k - 1) / k ))
        require_idle
        echo "=== split cost: ${size}M, k=$k (lbmax=$lbmax), native"
        (cd "$ROOT/set_intersection_cpp/build_mac" && MICRO_PREHEAT_MS=1000 MICRO_LB_MAX=$lbmax ./App "$one" e6) \
            > "$OUT/split/m5pro_wgpu_micro_e6_${size}M_k${k}_$DATE.log" 2>&1
        require_idle
        echo "=== split cost: ${size}M, k=$k (lbmax=$lbmax), Chrome"
        OUT="$OUT/split" SIZES=$size EXTRA="&lbmax=$lbmax" TAG="${size}M_k${k}" TIMEOUT=600 \
            bash "$ROOT/apple_upload/run_browser_mac.sh" micro | tail -1
    done
done

for launch in 1 2 3; do
    require_idle; echo "=== fusion e2 with preheat, launch $launch"
    OUT="$OUT" SIZES=$SAFE EXTRA="&ph=1000" TAG="ph1000_r$launch" TIMEOUT=900 bash "$ROOT/apple_upload/run_browser_mac.sh" fusion | tail -1
    require_idle; echo "=== fusion e6 with preheat, launch $launch"
    OUT="$OUT" SIZES=$SAFE EXTRA="&ph=1000" TAG="ph1000_r$launch" TIMEOUT=900 bash "$ROOT/apple_upload/run_browser_mac.sh" fusion-e6 | tail -1
done

for launch in 2 3; do
    require_idle; echo "=== Chrome micro, launch $launch"
    OUT="$OUT" SIZES=$SAFE TAG="r$launch" TIMEOUT=900 bash "$ROOT/apple_upload/run_browser_mac.sh" micro | tail -1
done
echo "done. logs in $OUT and $OUT/split"
