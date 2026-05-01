#!/usr/bin/env bash
# Launch EchoZero in a dedicated macOS Terminal shell.
# Exists so running from an existing terminal does not occupy that shell.
# Connects developer CLI workflows to a detached app launch behavior.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
ENTRYPOINT="${REPO_ROOT}/run_echozero.py"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing venv interpreter: ${PYTHON_BIN}" >&2
    echo "Run: python3 scripts/dev_bootstrap.py" >&2
    exit 1
fi

if [[ ! -f "${ENTRYPOINT}" ]]; then
    echo "Missing launcher entrypoint: ${ENTRYPOINT}" >&2
    exit 1
fi

args=()
for arg in "$@"; do
    args+=("$(printf "%q" "${arg}")")
done

args_segment=""
if ((${#args[@]} > 0)); then
    args_segment=" ${args[*]}"
fi

launch_command="cd $(printf "%q" "${REPO_ROOT}"); $(printf "%q" "${PYTHON_BIN}") $(printf "%q" "${ENTRYPOINT}")${args_segment}; exit"

/usr/bin/osascript <<APPLESCRIPT
tell application "Terminal"
    activate
    set launchedTab to do script "$(printf '%s' "${launch_command}" | sed 's/\\/\\\\/g; s/\"/\\"/g')"
    repeat while busy of launchedTab
        delay 0.2
    end repeat
    try
        close launchedTab
    on error
        try
            close (first window whose selected tab is launchedTab)
        end try
    end try
end tell
APPLESCRIPT
