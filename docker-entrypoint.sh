#!/bin/bash
# Startup script: runs proxy + Python simulation together

set -e

echo "Starting VoxelSim Backend..."

# Start proxy in background
echo "[1/2] Starting WebSocket proxy..."
cd /app/proxy
node proxy.js &
PROXY_PID=$!

# Wait for proxy to be ready
sleep 2

# Start Python simulation (runs forever)
echo "[2/2] Starting Python simulation..."
cd /app/python
python3 povtest.py

# If Python exits, cleanup
kill $PROXY_PID 2>/dev/null || true
