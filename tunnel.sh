#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# tunnel.sh  — Cloudflare Quick Tunnels for teammate demo
#
# Exposes:
#   API  http://localhost:8000  →  *.trycloudflare.com
#   Web  http://localhost:4321  →  *.trycloudflare.com
#
# No Cloudflare account needed (Quick Tunnels).
# ─────────────────────────────────────────────────────────────

set -euo pipefail

CLOUDFLARED="${HOME}/.local/bin/cloudflared"

if [[ ! -x "$CLOUDFLARED" ]]; then
  echo "❌  cloudflared not found at $CLOUDFLARED"
  echo "    Install: curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o ~/.local/bin/cloudflared && chmod +x ~/.local/bin/cloudflared"
  exit 1
fi

API_LOG=$(mktemp /tmp/cf-api-XXXX.log)
WEB_LOG=$(mktemp /tmp/cf-web-XXXX.log)

cleanup() {
  echo ""
  echo "🛑  Shutting down tunnels..."
  kill "$API_PID" "$WEB_PID" 2>/dev/null || true
  rm -f "$API_LOG" "$WEB_LOG"
}
trap cleanup EXIT INT TERM

echo "🚀  Starting Quick Tunnels…"
echo "    (Make sure services are running:)"
echo "    API : uv run uvicorn api.main:app --host 0.0.0.0 --port 8000"
echo "    Web : cd web && pnpm dev -- --host"
echo ""

# Capture both stdout AND stderr (cloudflared changed where it prints the URL across versions)
"$CLOUDFLARED" tunnel --url http://localhost:8000 --no-autoupdate > "$API_LOG" 2>&1 &
API_PID=$!

"$CLOUDFLARED" tunnel --url http://localhost:4321 --no-autoupdate > "$WEB_LOG" 2>&1 &
WEB_PID=$!

# Poll log file until a trycloudflare.com URL appears (timeout 40s)
wait_for_url() {
  local log="$1"
  local deadline=$(( $(date +%s) + 40 ))
  while [[ $(date +%s) -lt $deadline ]]; do
    # Try multiple patterns — cloudflared prints it in different lines/formats
    local url
    url=$(grep -oE 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' "$log" 2>/dev/null | head -1)
    if [[ -n "$url" ]]; then
      printf '%s' "$url"
      return 0
    fi
    sleep 0.5
  done
  # Fallback: dump the log so the user can see what happened
  echo "(timeout — cloudflared output:)"
  cat "$log" >&2
  printf '%s' "TUNNEL_FAILED"
}

echo "⏳  Waiting for tunnel URLs (up to 40s)…"
API_URL=$(wait_for_url "$API_LOG")
WEB_URL=$(wait_for_url "$WEB_LOG")

echo ""
echo "=================================================="
echo "  Cloudflare Quick Tunnels — share these URLs"
echo "=================================================="
echo ""
echo "  API  (FastAPI + /docs):"
echo "  --> ${API_URL}/docs"
echo ""
echo "  Web  (Astro interface):"
echo "  --> ${WEB_URL}"
echo ""
echo "=================================================="
echo "  Copy each URL on a SINGLE line (no line breaks)."
echo "  Press Ctrl+C to stop both tunnels."
echo "=================================================="
echo ""

# Keep alive until Ctrl+C
wait "$API_PID" "$WEB_PID"
