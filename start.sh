set -e

trap "kill 0" EXIT

echo ""
echo "========================================"
echo "  AI vs Real Image Detector — Launcher  "
echo "========================================"
echo ""

MODEL_PATH="ai-detector-app/backend/cnn_model.keras"
if [ ! -f "$MODEL_PATH" ]; then
    echo "⚠️  Model file not found at: $MODEL_PATH"
    echo ""
    echo "   Please copy your trained model first:"
    echo "   cp cifake_model.keras ai-detector-app/backend/cnn_model.keras"
    echo ""
    exit 1
fi

PYTHON_CMD="python3"

if ! command -v "$PYTHON_CMD" &>/dev/null; then
    if [ -f "/opt/anaconda3/bin/python3" ]; then
        PYTHON_CMD="/opt/anaconda3/bin/python3"
    else
        echo "❌ Python 3 not found. Please install Python 3.9 or newer."
        exit 1
    fi
fi

echo "🐍 Using Python: $($PYTHON_CMD --version)"

echo ""
echo "⚡ Starting Flask backend on http://localhost:5000 ..."
(
    cd ai-detector-app/backend
    "$PYTHON_CMD" app.py
) &

sleep 3

echo ""
echo "🎨 Starting React frontend on http://localhost:3000 ..."
echo ""
(
    cd ai-detector-app/frontend
    npm start
)
