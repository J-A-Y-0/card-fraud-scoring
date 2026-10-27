#!/usr/bin/env bash
# Render the report and deck to PDF and the two README figures to PNG from site/.
# Run after `uv run fraudscore export`. Needs Google Chrome (or set CHROME).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
PORT="${PORT:-8799}"
URL="http://localhost:$PORT"

uv run --project "$ROOT" python -m http.server "$PORT" --directory "$ROOT/site" >/dev/null 2>&1 &
SERVER=$!
trap 'kill $SERVER' EXIT
for _ in $(seq 1 50); do curl -s "$URL/" >/dev/null && break; sleep 0.1; done

chrome() { "$CHROME" --headless=new --disable-gpu --hide-scrollbars --virtual-time-budget=10000 "$@" 2>/dev/null; }

mkdir -p "$ROOT/reports" "$ROOT/docs/img"
chrome --no-pdf-header-footer --print-to-pdf="$ROOT/reports/fraud-scoring-report.pdf" "$URL/report.html"
chrome --no-pdf-header-footer --print-to-pdf="$ROOT/reports/fraud-scoring-deck.pdf" "$URL/deck.html"

# name:height pairs (bash 3 on macOS has no associative arrays)
for spec in validation:470 speed:560; do
  f="${spec%%:*}"; h="${spec##*:}"
  chrome --force-device-scale-factor=2 --window-size="1200,$h" --screenshot="$ROOT/docs/img/$f.png" "$URL/figure.html?f=$f"
done
echo "built: reports/fraud-scoring-report.pdf, reports/fraud-scoring-deck.pdf, docs/img/*.png"
