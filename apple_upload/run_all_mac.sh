#!/bin/bash
# Runs every M5 Pro experiment in sequence and can be re-run after an interruption: finished steps are skipped.
#
# Two guards protect the numbers:
#   1. Before every step the GPU must be idle. A hung Lookback dispatch keeps spinning on the GPU after its
#      process is killed (seen on the M5 Pro at 128M: 100% device utilization, 80x slower kernels for every later
#      run, no driver recovery within 8 minutes). Only a reboot clears it, so the suite refuses to start on a busy GPU.
#   2. A step whose logs stop growing for STALL seconds is treated as hung: it is interrupted, marked, and the
#      suite stops, because everything after it would be measured on an occupied GPU.
#
# The sizes that need a two-dimensional Lookback dispatch (128M) run last, one step each, under size128/.
#
# Usage: bash apple_upload/run_all_mac.sh [step ...]      (no arguments: every step)
#   STALL  seconds without log output before a step counts as hung (default: 300)
#   OUT    log directory (default: apple_logs)
set -uo pipefail
set -m  # every background step gets its own process group, so a hung step can be interrupted as a whole

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OUT="${OUT:-$ROOT/apple_logs}"
STALL="${STALL:-300}"
SAFE_SIZES="1,2,4,8,16,32,64"
NO128="$ROOT/public/data/no128"
mkdir -p "$OUT" "$OUT/size128"

# cmake from `pip3 install --user cmake` and Chrome in ~/Applications, for accounts without admin rights.
command -v cmake > /dev/null || export PATH="$HOME/Library/Python/3.9/bin:$PATH"
if [ -z "${CHROME:-}" ] && [ ! -x "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" ]; then
    export CHROME="$HOME/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
fi

gpu_util() { ioreg -r -d 1 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*' | grep -o '[0-9]*$' | head -1; }

gpu_idle() {
    local busy=0
    for _ in 1 2 3 4 5; do
        [ "$(gpu_util)" -ge 50 ] && busy=$((busy + 1))
        sleep 1
    done
    [ "$busy" -lt 3 ]
}

newest_log_age() {
    local newest now
    newest="$(find "$OUT" -type f -exec stat -f %m {} + 2>/dev/null | sort -n | tail -1)"
    now="$(date +%s)"
    echo $((now - ${newest:-$now}))
}

# The native micro-benchmark runs every size it finds, so the 1M-64M runs read a directory without the 128M files.
mkdir -p "$NO128"
for n in 1 2 4 8 16 32 64; do
    for ab in A B; do
        [ -e "$NO128/${ab}_${n}e6.bin" ] || ln -s "$ROOT/public/data/${ab}_${n}e6.bin" "$NO128/${ab}_${n}e6.bin"
    done
done

run_step() {
    local name="$1"; shift
    if [ -e "$OUT/.done_$name" ]; then echo "[skip] $name (done)"; return 0; fi
    if [ -e "$OUT/.hang_$name" ]; then echo "[skip] $name (hung before, see $OUT/.hang_$name)"; return 0; fi
    if ! gpu_idle; then
        echo "[stop] GPU is busy before $name (device utilization $(gpu_util)%). A hung dispatch is probably still"
        echo "       spinning on it. Reboot, then run this script again."
        exit 2
    fi
    echo "[run ] $name  $(date +%T)"
    touch "$OUT/.started_$name"
    env "$@" > "$OUT/driver_$name.out" 2>&1 &
    local pid=$!
    trap 'kill -INT -- "-'"$pid"'" 2> /dev/null; exit 130' INT TERM  # Ctrl-C on the suite also stops the step
    while kill -0 "$pid" 2> /dev/null; do
        sleep 10
        if [ "$(newest_log_age)" -ge "$STALL" ]; then
            echo "[hang] $name: no log output for ${STALL}s, GPU utilization $(gpu_util)%. Interrupting."
            { date; echo "no log output for ${STALL}s, GPU utilization $(gpu_util)%"; } > "$OUT/.hang_$name"
            kill -INT -- "-$pid" 2> /dev/null; sleep 5; kill -KILL -- "-$pid" 2> /dev/null
            echo "[stop] Reboot before running anything else, then run this script again to continue."
            exit 3
        fi
    done
    wait "$pid"; local rc=$?
    if [ "$rc" -ne 0 ]; then echo "[fail] $name exited with $rc, see $OUT/driver_$name.out"; exit 1; fi
    touch "$OUT/.done_$name"
    echo "[done] $name  $(date +%T)"
}

step() {
    case "$1" in
        micro)          run_step micro          DATA="$NO128" bash "$ROOT/apple_upload/run_micro_mac.sh" ;;
        eclat)          run_step eclat          bash "$ROOT/apple_upload/run_eclat_mac.sh" ;;
        fusion)         run_step fusion         SIZES="$SAFE_SIZES" bash "$ROOT/apple_upload/run_browser_mac.sh" fusion ;;
        fusion-e6)      run_step fusion-e6      SIZES="$SAFE_SIZES" bash "$ROOT/apple_upload/run_browser_mac.sh" fusion-e6 ;;
        chrome-micro)   run_step chrome-micro   SIZES="$SAFE_SIZES" bash "$ROOT/apple_upload/run_browser_mac.sh" micro ;;
        chrome-eclat)   run_step chrome-eclat   bash "$ROOT/apple_upload/run_browser_mac.sh" eclat ;;
        # 128M: numWg = 83,334 > 65,535, so Lookback runs on a 65535 x 2 grid. Native wgpu hung here on 2026-09-20.
        chrome-micro-128) run_step chrome-micro-128 OUT="$OUT/size128" SIZES=128 bash "$ROOT/apple_upload/run_browser_mac.sh" micro ;;
        fusion-128)       run_step fusion-128       OUT="$OUT/size128" SIZES=128 bash "$ROOT/apple_upload/run_browser_mac.sh" fusion ;;
        fusion-e6-128)    run_step fusion-e6-128    OUT="$OUT/size128" SIZES=128 bash "$ROOT/apple_upload/run_browser_mac.sh" fusion-e6 ;;
        *) echo "unknown step $1"; exit 1 ;;
    esac
}

STEPS=("$@")
[ ${#STEPS[@]} -gt 0 ] || STEPS=(micro eclat fusion fusion-e6 chrome-micro chrome-eclat chrome-micro-128 fusion-128 fusion-e6-128)
for s in "${STEPS[@]}"; do step "$s"; done
echo "all requested steps finished. logs in $OUT"
