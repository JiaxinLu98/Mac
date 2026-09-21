#!/bin/bash
# Download the five FIMI benchmark transaction datasets used by the ECLAT
# application (Section 5.1 / Figures for ECLAT).
#
# Source: http://fimi.uantwerpen.be/data/ (public, stable URLs)
# Output: $REPO_ROOT/public/fimi/{accidents,chess,kosarak,mushroom,webdocs}.dat
#
# Usage: bash scripts/download_fimi.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$REPO_ROOT/public/fimi"
BASE_URL="https://fimi.uantwerpen.be/data"

mkdir -p "$OUT_DIR"

fetch() {
    local url=$1
    local out=$2
    if [ -s "$OUT_DIR/$out" ]; then
        echo "[skip] $out (already present)"
        return
    fi
    echo "[get ] $out"
    curl -fL --progress-bar -o "$OUT_DIR/$out" "$url"
}

echo "Downloading FIMI datasets into $OUT_DIR"
echo "-----------------------------------------------------------"

fetch "$BASE_URL/accidents.dat" "accidents.dat"
fetch "$BASE_URL/chess.dat"     "chess.dat"
fetch "$BASE_URL/kosarak.dat"   "kosarak.dat"
fetch "$BASE_URL/mushroom.dat"  "mushroom.dat"

# webdocs is distributed gzip-compressed on the FIMI server
if [ -s "$OUT_DIR/webdocs.dat" ]; then
    echo "[skip] webdocs.dat (already present)"
else
    fetch "$BASE_URL/webdocs.dat.gz" "webdocs.dat.gz"
    echo "[gunzip] webdocs.dat.gz -> webdocs.dat"
    gunzip -f "$OUT_DIR/webdocs.dat.gz"
fi

echo "-----------------------------------------------------------"
echo "Done. FIMI datasets:"
ls -lh "$OUT_DIR"/*.dat 2>/dev/null || echo "(nothing downloaded)"
