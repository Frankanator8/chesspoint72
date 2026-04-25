#!/usr/bin/env bash
# Wrapper that lets host UIs (Pygame, CuteChess, Arena, lichess-bot, …)
# launch Calix by an executable path even though the engine itself is a
# Python module. Forwards all stdin/stdout/stderr verbatim and passes any
# extra args through to the engine.
#
# Default agent mode is "aware". Override at launch time with:
#   --engine /path/to/calix.sh --agent-mode autonomous
exec python -m chesspoint72.aiengines.jonathan.v2.main --agent-mode aware "$@"
