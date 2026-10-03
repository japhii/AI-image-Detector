#!/usr/bin/env bash
# =============================================================================
# start.sh
# Starts the Flask backend and the React frontend with a single command.
#
# Usage:
#   chmod +x start.sh   (only needed once, makes the file executable)
#   ./start.sh
# =============================================================================

set -e  # Exit immediately if any command fails

# Kill all background jobs (backend) when the script exits or is interrupted
trap "kill 0" EXIT

echo ""
echo "========================================"
echo "  AI vs Real Image Detector — Launcher  "
echo "========================================"
echo ""

# ── 1. Check that the trained model has been copied to the backend ─────────────
MODEL_PATH="ai-detector-app/backend/cnn_model.keras"
if [ ! -f "$MODEL_PATH" ]; then
    echo "⚠️  Model file not found at: $MODEL_PATH"
    echo ""
    echo "   Please copy your trained model first:"
    echo "   cp cifake_model.keras ai-detector-app/backend/cnn_model.keras"
    echo ""
    exit 1
fi

# ── 2. Find the correct Python 3 interpreter ──────────────────────────────────
PYTHON_CMD="python3"

if ! command -v "$PYTHON_CMD" &>/dev/null; then
    # Fallback to Anaconda if system python3 is not available
    if [ -f "/opt/anaconda3/bin/python3" ]; then
        PYTHON_CMD="/opt/anaconda3/bin/python3"
    else
        echo "❌ Python 3 not found. Please install Python 3.9 or newer."
        exit 1
    fi
fi

echo "🐍 Using Python: $($PYTHON_CMD --version)"

# ── 3. Start Flask backend on port 5000 ───────────────────────────────────────
echo ""
echo "⚡ Starting Flask backend on http://localhost:5000 ..."
(
    cd ai-detector-app/backend
    "$PYTHON_CMD" app.py
) &

# Wait briefly for the backend to finish loading the model before starting the UI
sleep 3

# ── 4. Start React frontend on port 3000 ──────────────────────────────────────
echo ""
echo "🎨 Starting React frontend on http://localhost:3000 ..."
echo ""
(
    cd ai-detector-app/frontend
    npm start
)
