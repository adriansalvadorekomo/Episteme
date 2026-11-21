#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# tunnel.sh  — Cloudflare Quick Tunnels for teammate demo
#
# Exposes:
#   API  http://localhost:8000  →  *.trycloudflare.com
#   Web  http://localhost:4321  →  *.trycloudflare.com   (port auto-detected)
#
# No Cloudflare account needed (Quick Tunnels).
#
# A "502 Bad gateway" from trycloudflare.com means the tunnel works but nothing answers
# on the LOCAL port it points to. This script therefore checks both services first,
# finds the port the web dev server really uses (Astro moves to 4322, 4323… when 4321 is
# taken), and keeps watching them while the tunnels are up.
#
# Overrides: API_PORT=8000 WEB_PORT=4321 ./tunnel.sh
# ─────────────────────────────────────────────────────────────

set -euo pipefail

CLOUDFLARED="${CLOUDFLARED:-${HOME}/.local/bin/cloudflared}"
API_PORT="${API_PORT:-8000}"

if [[ ! -x "$CLOUDFLARED" ]]; then
  echo "❌  cloudflared not found at $CLOUDFLARED"
  echo "    Install: curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o ~/.local/bin/cloudflared && chmod +x ~/.local/bin/cloudflared"
  exit 1
fi

# HTTP status of a local URL ("000" = nothing listening)
http_code() { curl -s -o /dev/null -w '%{http_code}' --max-time 3 "$1" 2>/dev/null || true; }

# ── 1. Preflight: are the services really answering? ────────
api_ok() { [[ "$(http_code "http://localhost:${API_PORT}/health")" == "200" ]]; }

find_web_port() {
  # explicit WEB_PORT wins; otherwise the first Astro dev server on 4321…4330
  local candidates=("${WEB_PORT:-}")
  [[ -z "${WEB_PORT:-}" ]] && candidates=(4321 4322 4323 4324 4325 4326 4327 4328 4329 4330)
  local p
  for p in "${candidates[@]}"; do
    if curl -s --max-time 3 "http://localhost:${p}/" 2>/dev/null | grep -qiE 'astro|episteme'; then
      printf '%s' "$p"
      return 0
    fi
  done
  return 1
}

echo "🔎  Checking local services…"
if api_ok; then
  echo "    ✓ API  answers on :${API_PORT}"
else
  echo "    ✗ API  is not answering on :${API_PORT}"
  echo "      start it:  uv run uvicorn api.main:app --host 0.0.0.0 --port ${API_PORT}"
  echo "      (the model takes ~20 s to load; /health answers before that)"
  exit 1
fi

if WEB_PORT="$(find_web_port)"; then
  echo "    ✓ Web  answers on :${WEB_PORT}"
  if [[ "$WEB_PORT" != "4321" ]]; then
    echo "      note: the web server is on :${WEB_PORT}, not :4321 (4321 was busy when it started)."
    echo "      The tunnel follows it; restart 'pnpm dev' to get back to :4321."
  fi
else
  echo "    ✗ Web  is not answering on :${WEB_PORT:-4321…4330}"
  echo "      start it:  cd web && pnpm dev --host"
  exit 1
fi
echo ""

# ── 2. Tunnels ──────────────────────────────────────────────
API_LOG=$(mktemp /tmp/cf-api-XXXX.log)
WEB_LOG=$(mktemp /tmp/cf-web-XXXX.log)
API_PID="" WEB_PID="" WATCH_PID=""

cleanup() {
  echo ""
  echo "🛑  Shutting down tunnels..."
  kill $API_PID $WEB_PID $WATCH_PID 2>/dev/null || true
  rm -f "$API_LOG" "$WEB_LOG"
}
trap cleanup EXIT INT TERM

echo "🚀  Starting Quick Tunnels…"
# Capture both stdout AND stderr (cloudflared changed where it prints the URL across versions)
"$CLOUDFLARED" tunnel --url "http://localhost:${API_PORT}" --no-autoupdate > "$API_LOG" 2>&1 &
API_PID=$!
"$CLOUDFLARED" tunnel --url "http://localhost:${WEB_PORT}" --no-autoupdate > "$WEB_LOG" 2>&1 &
WEB_PID=$!

# Poll log file until a trycloudflare.com URL appears (timeout 40s)
wait_for_url() {
  local log="$1"
  local deadline=$(( $(date +%s) + 40 ))
  while [[ $(date +%s) -lt $deadline ]]; do
    local url
    url=$(grep -oE 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' "$log" 2>/dev/null | head -1)
    if [[ -n "$url" ]]; then
      printf '%s' "$url"
      return 0
    fi
    sleep 0.5
  done
  # Fallback: dump the log (on stderr, so it doesn't end up in the URL variable)
  echo "(timeout — cloudflared output:)" >&2
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
echo "  API  (FastAPI + /docs)   → localhost:${API_PORT}"
echo "  --> ${API_URL}/docs"
echo ""
echo "  Web  (Astro interface)   → localhost:${WEB_PORT}"
echo "  --> ${WEB_URL}"
echo ""
echo "=================================================="
echo "  Copy each URL on a SINGLE line (no line breaks)."
echo "  A new trycloudflare.com DNS name can take ~30 s to resolve."
echo "  Press Ctrl+C to stop both tunnels."
echo "=================================================="
echo ""

# ── 3. Watchdog: warn as soon as a local service stops answering (= 502 for visitors) ──
(
  api_down=0 web_down=0
  while sleep 15; do
    if api_ok; then
      (( api_down )) && echo "✓ $(date +%T) API is answering again on :${API_PORT}"
      api_down=0
    elif (( ! api_down )); then
      echo "⚠️  $(date +%T) API stopped answering on :${API_PORT}: visitors get 502. Restart uvicorn (same port) and the URL keeps working."
      api_down=1
    fi
    if [[ "$(http_code "http://localhost:${WEB_PORT}/")" == "200" ]]; then
      (( web_down )) && echo "✓ $(date +%T) Web is answering again on :${WEB_PORT}"
      web_down=0
    elif (( ! web_down )); then
      echo "⚠️  $(date +%T) Web stopped answering on :${WEB_PORT}: visitors get 502. Restart 'pnpm dev --host' on port ${WEB_PORT}."
      web_down=1
    fi
  done
) &
WATCH_PID=$!

# Keep alive until Ctrl+C
wait "$API_PID" "$WEB_PID"
