#!/usr/bin/env bash
# Probe the Claude Code subscription's current usage by spawning a short
# `claude` session in tmux, sending /usage, and capturing the output.
#
# Prints (one line each):
#   SESSION_PCT=<n>      Percent used in the current 5h session window
#   WEEK_PCT=<n>         Percent used in the current weekly window
#   SESSION_RESET=<...>  Human-readable reset time string
#
# Exits 0 even if it fails to parse — callers should check whether the
# expected lines are present.
set -u
TMUX_SESSION="usage_probe_$$"
tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true
tmux new-session -d -s "${TMUX_SESSION}" -x 200 -y 60 'claude --no-chrome --no-session-persistence' >/dev/null 2>&1
# Boot time for the interactive shell + initial render.
sleep 8
tmux send-keys -t "${TMUX_SESSION}" '/usage' Enter 2>/dev/null || true
# Render time for the /usage panel.
sleep 4
PANE=$(tmux capture-pane -t "${TMUX_SESSION}" -p 2>/dev/null || echo "")
tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true

# Parse the rendered text. The /usage panel prints lines like:
#   Current session
#   ████████████████  60% used
#   Resets 4:29am (Europe/Warsaw)
#
# We extract the first two "NN% used" lines (session, then week).
SESSION_LINE=$(echo "${PANE}" | grep -E "% used" | head -1 || true)
WEEK_LINE=$(echo "${PANE}" | grep -E "% used" | sed -n '2p' || true)
RESET_LINE=$(echo "${PANE}" | grep -E "^[[:space:]]*Resets " | head -1 || true)

SESSION_PCT=$(echo "${SESSION_LINE}" | grep -oE "[0-9]+%" | head -1 | tr -d '%')
WEEK_PCT=$(echo "${WEEK_LINE}" | grep -oE "[0-9]+%" | head -1 | tr -d '%')
SESSION_RESET=$(echo "${RESET_LINE}" | sed -E 's/^[[:space:]]*Resets //; s/[[:space:]]+$//')

echo "SESSION_PCT=${SESSION_PCT:-unknown}"
echo "WEEK_PCT=${WEEK_PCT:-unknown}"
echo "SESSION_RESET=${SESSION_RESET:-unknown}"
