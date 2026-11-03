#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# tunnel.sh  — Cloudflare Quick Tunnels for teammate demo
#
# Exposes:
#   API  http://localhost:8000  →  *.trycloudflare.com
#   Web  http://localhost:4321  →  *.trycloudflare.com
#
# No Cloudflare account needed (Quick Tunnels).
# URLs are printed once they appear in cloudflared output.
# ─────────────────────────────────────────────────────────────

set -euo pipefail

CLOUDFLARED="${HOME}/.local/bin/cloudflared"

if ! command -v "$CLOUDFLARED" &>/dev/null; then
  echo "❌  cloudflared not found at $CLOUDFLARED"
  echo "    Install: curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o ~/.local/bin/cloudflared && chmod +x ~/.local/bin/cloudflared"
  exit 1
fi

API_LOG=$(mktemp)
WEB_LOG=$(mktemp)

cleanup() {
  echo ""
  echo "🛑  Shutting down tunnels..."
  kill "$API_PID" "$WEB_PID" 2>/dev/null || true
  rm -f "$API_LOG" "$WEB_LOG"
}
trap cleanup EXIT INT TERM

echo "🚀  Starting Quick Tunnels…"
echo ""

# Launch both tunnels in background, redirect stderr (where the URL appears) to log files
"$CLOUDFLARED" tunnel --url http://localhost:8000 --no-autoupdate 2>"$API_LOG" &
API_PID=$!

"$CLOUDFLARED" tunnel --url http://localhost:4321 --no-autoupdate 2>"$WEB_LOG" &
WEB_PID=$!

# Wait until each tunnel prints its public URL (timeout 30s each)
wait_for_url() {
  local log="$1"
  local label="$2"
  local deadline=$(( $(date +%s) + 30 ))
  while [[ $(date +%s) -lt $deadline ]]; do
    local url
    url=$(grep -oP 'https://[a-zA-Z0-9\-]+\.trycloudflare\.com' "$log" 2>/dev/null | head -1)
    if [[ -n "$url" ]]; then
      echo "$url"
      return 0
    fi
    sleep 0.5
  done
  echo "(timeout — check $log)"
}

echo "⏳  Waiting for tunnel URLs…"
API_URL=$(wait_for_url "$API_LOG" "API")
WEB_URL=$(wait_for_url "$WEB_LOG" "Web")

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║              🌐  Cloudflare Quick Tunnels                ║"
echo "╠══════════════════════════════════════════════════════════╣"
printf  "║  🔌  API (FastAPI docs)  %-34s║\n" "${API_URL}/docs"
printf  "║  💻  Web interface       %-34s║\n" "${WEB_URL}"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "📋  Share these URLs with your teammate."
echo "    Press Ctrl+C to stop both tunnels."
echo ""

# Keep the script alive until Ctrl+C
wait "$API_PID" "$WEB_PID"
