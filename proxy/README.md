# VoxelSim WebSocket Proxy

TCP-to-WebSocket proxy server for connecting Python simulation to browser-based WASM renderer.

## Purpose

Browsers cannot directly open TCP connections due to security restrictions. This proxy bridges the gap:

```
Python Simulation → TCP → Proxy → WebSocket → Browser (WASM)
```

## Port Mapping

| Data Type | Python Connects (TCP) | Proxy Serves (WebSocket) |
|-----------|----------------------|--------------------------|
| VoxelGrid | localhost:8080 | ws://localhost:9080 |
| Agents | localhost:8081 | ws://localhost:9081 |
| POV World | localhost:8090 | ws://localhost:9090 |
| POV Agents | localhost:9090 | ws://localhost:10090 |

## Installation

```bash
cd proxy
npm install
```

## Usage

### Start the proxy server:
```bash
npm start
```

The proxy will:
1. Listen for TCP connections from Python on ports 8080, 8081, 8090, 9090
2. Serve WebSocket connections to browsers on ports 9080, 9081, 9090, 10090
3. Forward all binary data unchanged (no parsing/modification)

### Start Python simulation (in another terminal):
```bash
cd python
python povtest.py
# or
python sim_from_world.py
```

### Open browser renderer:
```bash
# After WASM implementation is ready:
# Open http://localhost:8000 (or wherever WASM is served)
```

## Data Format

All data is forwarded as-is in binary format:
- 4-byte little-endian length prefix
- Bincode-serialized payload

The proxy does NOT parse or modify the data stream.

## Troubleshooting

**"TCP connection error"**: Normal on startup. Proxy will keep retrying until Python connects.

**"Browser disconnected"**: Browser closed or refreshed. Proxy continues running.

**Multiple browsers**: Only the last connected browser receives data. For multiple viewers, see architecture notes.
