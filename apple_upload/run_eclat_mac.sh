#!/bin/bash
# ECLAT on the M5 Pro through wgpu-native on Metal, with the same batched pipeline, chunk size and
# run counts as the A100 runs: 3 warmup and 10 timed runs per dataset, chunks of at most 512 MB of input.
#
# Usage: bash apple_upload/run_eclat_mac.sh
#   DATA   directory with chess.dat, kosarak.dat and webdocs.dat (default: eclat_cpp/data)
#   SETS   datasets to run (default: "chess kosarak webdocs"; webdocs is 1.4 GB and is skipped if absent)
#   OUT    log directory (default: apple_logs)
# webdocs is not in the bundle. Download it with: bash scripts/download_fimi.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${DATA:-$ROOT/eclat_cpp/data}"
# The datasets are not in the repository: scripts/download_fimi.sh puts them in public/fimi.
[ -s "$DATA/chess.dat" ] || DATA="$ROOT/public/fimi"
SETS="${SETS:-chess kosarak webdocs}"
OUT="${OUT:-$ROOT/apple_logs}"
BUILD="$ROOT/eclat_cpp/build_mac"
DATE="$(date +%F)"
WARMUP=3
ITERS=10
CHUNK_MB=512
mkdir -p "$OUT"

command -v cmake > /dev/null || { echo "cmake not found: brew install cmake"; exit 1; }
cmake -S "$ROOT/eclat_cpp" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DFETCHCONTENT_BASE_DIR="$BUILD/_deps"
cmake --build "$BUILD" --target AppOpt -j

support_of() {
    case "$1" in
        chess)   echo 2000 ;;
        kosarak) echo 5000 ;;
        webdocs) echo 500000 ;;
        *)       echo "" ;;
    esac
}

for d in $SETS; do
    support="$(support_of "$d")"
    if [ -z "$support" ]; then echo "unknown dataset $d, skipped"; continue; fi
    if [ ! -s "$DATA/$d.dat" ]; then
        echo "missing $DATA/$d.dat, skipped (webdocs: bash scripts/download_fimi.sh)"
        continue
    fi
    log="$OUT/m5pro_wgpu_eclat_${d}_$DATE.log"
    echo "=== $d, min support $support -> $(basename "$log")"
    # AppOpt looks for shaders/ relative to the working directory.
    (cd "$BUILD" && ./AppOpt "$DATA/$d.dat" "$support" "$WARMUP" "$ITERS" "$CHUNK_MB") 2>&1 | tee "$log"
done

echo "done. logs in $OUT"
