#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

# Free ports 8000 and 3000 if occupied
fuser -k 8000/tcp 2>/dev/null || true
fuser -k 3000/tcp 2>/dev/null || true

echo "Starting AARC Backend on http://localhost:8000..."
PYTHONPATH="$DIR" "$DIR/backend/.venv/bin/uvicorn" backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "Starting AARC Frontend on http://localhost:3000..."
cd "$DIR/frontend"
npm run start -- -H 0.0.0.0 &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true; exit" SIGINT SIGTERM EXIT

echo "=========================================================="
echo " AARC QUANTITATIVE TERMINAL IS RUNNING"
echo " Backend:  http://localhost:8000"
echo " Frontend: http://localhost:3000"
echo "=========================================================="

wait
