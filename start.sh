#!/bin/bash

# AI Image Detector - Start Script
# Starts FastAPI backend on port 8000 and Next.js frontend on port 3000

echo "🚀 Starting AI Image Detector Backend & Frontend..."

# Kill background jobs on exit
trap "kill 0" EXIT

# Start FastAPI server
echo "⚡ Starting FastAPI backend on http://localhost:8000..."
python3 -m uvicorn backend.main:app --port 8000 --reload &

# Wait briefly for backend to boot
sleep 2

# Start Next.js frontend
echo "🎨 Starting Next.js frontend on http://localhost:3000..."
cd frontend && npm run dev
