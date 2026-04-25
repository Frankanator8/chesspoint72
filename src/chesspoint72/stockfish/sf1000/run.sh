#!/bin/bash

# Define the command for the Stockfish executable
ENGINE="stockfish"

# Check if stockfish is accessible
if ! command -v $ENGINE &> /dev/null; then
    echo "Error: $ENGINE could not be found. Please ensure it is installed and in your PATH."
    exit 1
fi

# The script uses a subshell to pipe configuration commands followed by
# an open 'cat' command. This ensures the options are set immediately
# upon launch while keeping the input stream open for your moves.

# 1. UCI_LimitStrength: Enables the internal logic that throttles the engine.
# 2. UCI_Elo: Sets the target strength. Stockfish 11 and later support
#    calibrated Elo settings.

(
  echo "setoption name UCI_LimitStrength value true"
  echo "setoption name UCI_Elo value 1000"
  echo "isready"
  cat
) | $ENGINE