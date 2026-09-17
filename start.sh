#!/bin/bash

# AI Image Detector - Start Script
# Starts FastAPI backend on port 8000 and Next.js frontend on port 3000

echo "🚀 Starting AI Image Detector Backend & Frontend..."

# Kill background jobs on exit
trap "kill 0" EXIT

# Determine python command with uvicorn available
PYTHON_CMD="python3"
if ! $PYTHON_CMD -c "import uvicorn" &>/dev/null; then
    if [ -f "/opt/anaconda3/bin/python3" ] && /opt/anaconda3/bin/python3 -c "import uvicorn" &>/dev/null; then
        PYTHON_CMD="/opt/anaconda3/bin/python3"
    else
        echo "📦 Installing backend requirements for $PYTHON_CMD..."
        $PYTHON_CMD -m pip install -r backend/requirements.txt
    fi
fi

# Start FastAPI server
echo "⚡ Starting FastAPI backend on http://localhost:8000 using $PYTHON_CMD..."
$PYTHON_CMD -m uvicorn backend.main:app --port 8000 --reload &

# Wait briefly for backend to boot
sleep 2

# Start Next.js frontend
echo "🎨 Starting Next.js frontend on http://localhost:3000..."
cd frontend && npm run dev
