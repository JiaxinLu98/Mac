#!/bin/bash
# Browser experiments on the M5 Pro: Chrome drives Metal through WebGPU.
# Only the 4-step versus 2-step fusion comparison needs the browser, because the native harness
# implements the 2-step pipeline alone. The micro-benchmark and ECLAT can also run here as a
# cross-check of the native runs.
#
# The script starts the webpack dev server, opens Chrome with the requested benchmark, and waits
# for the page to print [bench-done]. Chrome writes its console output to the log.
#
# Usage: bash apple_upload/run_browser_mac.sh [fusion|micro|eclat]
#   SIZES   sizes in millions for fusion and micro (default: 1,2,4,8,16,32,64,128)
#   RANGE   value range for fusion (default: e2, the range of the fusion table)
#   OUT     log directory (default: apple_logs)
#   PORT    dev server port (default: 8080)
#   TIMEOUT seconds to wait for the end marker (default: 3600)
#   EXTRA   extra URL parameters, e.g. "&ph=1000&lbmax=4096" (Lookback dispatch scheme: lb=split|2d, lbmax=<n>)
#   TAG     suffix for the log name, so variants do not overwrite each other
# The dev server serves public/ at the site root, so the data files must be in public/data
# (bash apple_upload/gen_micro_data_mac.sh "e2 e6").
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="${1:-fusion}"
SIZES="${SIZES:-1,2,4,8,16,32,64,128}"
RANGE="${RANGE:-e2}"
OUT="${OUT:-$ROOT/apple_logs}"
PORT="${PORT:-8080}"
DATE="$(date +%F)"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
mkdir -p "$OUT"

[ -x "$CHROME" ] || { echo "Chrome not found at $CHROME (set CHROME=...)"; exit 1; }
command -v npm > /dev/null || { echo "npm not found: brew install node"; exit 1; }

case "$APP" in
    fusion) URL="http://localhost:$PORT/?app=fusion&ops=0,2&range=$RANGE&sizes=$SIZES&w=10&n=100"; MARK='fusion-result' ;;
    fusion-e6) URL="http://localhost:$PORT/?app=fusion&ops=0&range=e6&sizes=$SIZES&w=10&n=100"; MARK='fusion-result'; APP="fusion_e6" ;;
    micro)  URL="http://localhost:$PORT/?app=micro&sizes=$SIZES&ph=1000"; MARK='micro-result' ;;
    eclat)  URL="http://localhost:$PORT/?app=eclat&ds=${DS:-chess,kosarak}&impl=inplace&w=3&n=10"; MARK='eclat' ;;
    *) echo "usage: bash apple_upload/run_browser_mac.sh [fusion|fusion-e6|micro|eclat]"; exit 1 ;;
esac

[ -d "$ROOT/node_modules" ] || (cd "$ROOT" && npm install)

# The page fetches the FIMI datasets from the site root, so they must sit in public/fimi.
if [ "$APP" = "eclat" ]; then
    mkdir -p "$ROOT/public/fimi"
    for d in chess kosarak webdocs; do
        [ -s "$ROOT/eclat_cpp/data/$d.dat" ] && [ ! -s "$ROOT/public/fimi/$d.dat" ] && cp "$ROOT/eclat_cpp/data/$d.dat" "$ROOT/public/fimi/$d.dat"
    done
fi

URL="$URL${EXTRA:-}"
log="$OUT/m5pro_chrome_${APP}${TAG:+_$TAG}_$DATE.log"
serverLog="$OUT/m5pro_devserver_$DATE.log"
(cd "$ROOT" && npm run serve -- --port "$PORT" > "$serverLog" 2>&1) &
serverPid=$!
trap 'kill $serverPid 2>/dev/null || true' EXIT

echo "waiting for the dev server on port $PORT"
for _ in $(seq 60); do
    curl -sf "http://localhost:$PORT" > /dev/null && break
    sleep 1
done

profile="$(mktemp -d)"
echo "=== $APP -> $(basename "$log")"
# No --virtual-time-budget: under virtual time requestDevice() never resolves (Chrome 153, M5 Pro). Headless Chrome
# does not exit on its own either, so it runs in the background until the page prints its end marker
# ([bench-done], or [eclat-done] for ECLAT) or TIMEOUT seconds pass.
"$CHROME" --headless=new --disable-gpu-sandbox --enable-unsafe-webgpu \
    --user-data-dir="$profile" \
    --enable-logging=stderr --v=0 "$URL" > "$log" 2>&1 &
chromePid=$!
# npm leaves the webpack node process behind when its shell is killed, so the listener on the port is stopped too.
trap 'kill $chromePid $serverPid $(lsof -ti tcp:$PORT -sTCP:LISTEN) 2>/dev/null || true' EXIT
waited=0
until grep -q -E 'bench-done|eclat-done' "$log" 2>/dev/null; do
    kill -0 "$chromePid" 2>/dev/null || break
    [ "$waited" -ge "${TIMEOUT:-3600}" ] && { echo "  timed out after ${TIMEOUT:-3600}s"; break; }
    sleep 2; waited=$((waited + 2))
done
kill "$chromePid" 2>/dev/null || true
wait "$chromePid" 2>/dev/null || true
rm -rf "$profile"

echo "  $(grep -c "$MARK" "$log" || true) result lines in $(basename "$log")"
if ! grep -q -E 'bench-done|eclat-done' "$log"; then
    echo "  [bench-done] not found. Headless Chrome may not expose WebGPU on this machine."
    echo "  Run it by hand instead: npm run serve, then open $URL in Chrome and copy the console output."
    exit 1
fi
