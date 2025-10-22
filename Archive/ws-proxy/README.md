# WebSocket Proxy for VoxelSim Renderer

This directory contains the WebSocket proxy server that bridges TCP (Python simulation) with WebSocket (browser WASM renderer).

**⚠️ PORT CHANGE:** WebSocket ports changed from 90xx to 180xx to avoid conflicts with Logitech software.
- New mapping: TCP 8080 → WS **18080**, TCP 8081 → WS **18081**

## Architecture

```
Python Simulation (test_terrain.py)
    ↓ TCP port 8080 (VoxelGrid)
    ↓ TCP port 8081 (Agents)
WebSocket Proxy (proxy.py)
    ↓ WebSocket port 9080 (VoxelGrid)
    ↓ WebSocket port 9081 (Agents)
Browser WASM Renderer
    ↓ Renders in <canvas>
User sees 3D visualization
```

## Files

- `proxy.py` - WebSocket proxy server (TCP ↔ WebSocket bridge)
- `test_terrain.py` - Test script that generates simple terrain
- `README.md` - This file

## Dependencies

Install required Python packages:

```bash
pip install websockets
```

The voxelsim package should already be installed if you're working with this repo.

## Testing the Complete Pipeline

Follow these steps to test the full WebSocket integration:

### Step 1: Build the WASM Renderer

```bash
cd ../voxelsim-renderer
cargo build --target wasm32-unknown-unknown --release
```

### Step 2: Start the WebSocket Proxy

In a **new terminal**:

```bash
cd ws-proxy
python proxy.py
```

You should see:
```
Starting WebSocket Proxy Server...
============================================================
Port Mappings:
  TCP :8080 ↔ WS :9080
  TCP :8081 ↔ WS :9081
============================================================
WebSocket server listening on ws://localhost:9080
WebSocket server listening on ws://localhost:9081
Connecting to TCP localhost:8080...
Connecting to TCP localhost:8081...
```

The proxy will keep trying to connect to TCP ports (this is normal - it waits for the Python simulation).

### Step 3: Run the Terrain Test Script

In a **new terminal**:

```bash
cd ws-proxy
python test_terrain.py
```

You should see:
```
============================================================
VoxelSim Terrain Test
============================================================

Creating RendererClient (TCP)...
✓ Connected to TCP ports 8080 (world)

Generating terrain...
Creating ground floor...
Creating walls...
Adding target blocks...
Adding tower...
Adding obstacles...
✓ Created world with 144 cells

Sending terrain to renderer...
(Press Ctrl+C to stop)
------------------------------------------------------------
Sent 10 frames...
Sent 20 frames...
```

In the **proxy terminal**, you should now see:
```
Connected to TCP :8080
Broadcasting 1234 bytes from TCP :8080 to 0 WS clients
```

(The "0 WS clients" is normal - the browser isn't connected yet)

### Step 4: Start the WASM Renderer in Browser

In a **new terminal**:

```bash
cd ../voxelsim-renderer
wasm-server-runner target/wasm32-unknown-unknown/release/voxelsim-renderer.wasm
```

Or if you have a web server set up, open:
```
http://localhost:8000/index.html
```

You should see the browser open and the 3D renderer start. Check the browser console (F12) for logs:
```
Starting VoxelSim Renderer (WASM WebSocket Mode)...
Connecting to WebSocket: ws://localhost:9080
Connecting to WebSocket: ws://localhost:9081
WebSocket connected successfully
```

In the **proxy terminal**, you should see:
```
WS client connected on :9080 from ('127.0.0.1', 54321)
Broadcasting 1234 bytes from TCP :8080 to 1 WS clients
```

### Step 5: Verify the Rendering

In the browser window, you should see:
- A 10×10 ground floor (gray)
- Walls around the perimeter
- Red target blocks at various positions
- A tower in one corner
- Some obstacle blocks

Use the mouse to:
- **Left drag**: Rotate camera
- **Right drag**: Pan camera
- **Scroll**: Zoom in/out

## Troubleshooting

### "Cannot connect to TCP :8080"

The proxy can't find the Python simulation. Make sure `test_terrain.py` is running.

### "WebSocket connection failed" (in browser console)

The proxy isn't running or is on a different port. Make sure `proxy.py` is running and check the port numbers.

### "Failed to create RendererClient"

The proxy isn't accepting TCP connections. Make sure:
1. The proxy is running
2. Ports 8080 and 8081 aren't blocked by firewall
3. No other program is using these ports

### I see the browser but no terrain

Check the browser console (F12) for errors. Common issues:
- WebSocket not connecting (see above)
- Data format mismatch (should auto-fix with our code)
- WASM build issue (rebuild the WASM target)

### Proxy shows "Broadcasting to 0 WS clients"

The browser hasn't connected yet. Make sure the WASM renderer is running in the browser.

## Port Reference

| Service | TCP Port | WebSocket Port | Data Type |
|---------|----------|----------------|-----------|
| World   | 8080     | 9080           | VoxelGrid |
| Agents  | 8081     | 9081           | HashMap<usize, Agent> |
| POV World | 8090   | 9090           | PovData (disabled by default) |
| POV Agents | 9090  | 10090          | PovData (disabled by default) |

## Next Steps

Once this basic test works:

1. **Add agent support** - Modify test_terrain.py to send agents on port 8081
2. **Add POV support** - Uncomment POV bridges in proxy.py and update pov.rs
3. **Create interactive controls** - Add keyboard input forwarding (browser → proxy → Python)
4. **Integrate with real simulation** - Replace test_terrain.py with your actual simulation

## Architecture Notes

### Why use a proxy?

Browsers cannot directly connect to TCP sockets due to security restrictions. WebSocket is the browser-native bidirectional protocol. The proxy bridges the gap without requiring changes to the Python simulation code.

### Binary format

Both TCP and WebSocket use the same binary format:
```
[4 bytes: message length (u32 little-endian)]
[N bytes: bincode-serialized payload]
```

The proxy forwards these frames as-is without parsing, making it very efficient.

### Bidirectional support

The proxy supports bidirectional communication:
- **Downstream**: Python → TCP → Proxy → WebSocket → Browser
- **Upstream**: Browser → WebSocket → Proxy → TCP → Python

This enables browser-based UI controls to send commands back to the simulation.
