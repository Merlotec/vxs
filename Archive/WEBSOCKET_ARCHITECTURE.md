# WebSocket Bidirectional Architecture

## Overview

This document describes the bidirectional WebSocket architecture for running voxelsim-renderer in the browser while keeping the Python simulation server running locally via TCP.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         BROWSER (WASM)                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Bevy Renderer (WASM)                        │  │
│  │  ┌────────────┐         ┌─────────────────┐             │  │
│  │  │ UI Input   │────────>│ network_ws.rs   │             │  │
│  │  │ (Keyboard) │         │  .send()        │             │  │
│  │  └────────────┘         └────────┬────────┘             │  │
│  │                                  │                       │  │
│  │                                  ↓ (GuiCommand)         │  │
│  │         ┌────────────────────────┴───────────┐          │  │
│  │         │     WebSocket Connection           │          │  │
│  │         │   (ws://localhost:9080, 9081...)   │          │  │
│  │         └────────────────────────┬───────────┘          │  │
│  │                                  ↑                       │  │
│  │                                  │ (VoxelGrid, Agents)  │  │
│  │         ┌────────────────────────┴───────────┐          │  │
│  │         │  network_ws.rs NetworkSubscriber    │          │  │
│  │         │    (receives data via onmessage)    │          │  │
│  │         └──────────────────────────────────────┘         │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 │ WebSocket
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                    WEBSOCKET PROXY SERVER                       │
│                  (Node.js or Python - TBD)                      │
│                                                                 │
│  Downstream (Sim → Browser):                                   │
│    - Listens on TCP ports 8080, 8081, 8090+, 9090+            │
│    - Receives [4-byte len][bincode payload]                    │
│    - Forwards as-is to WebSocket clients                       │
│                                                                 │
│  Upstream (Browser → Sim):                                     │
│    - Receives from WebSocket: [4-byte len][bincode payload]   │
│    - Forwards to TCP connection back to simulation             │
│                                                                 │
│  Port Mapping:                                                 │
│    TCP 8080 (VoxelGrid)  ↔ WS 9080                            │
│    TCP 8081 (Agents)     ↔ WS 9081                            │
│    TCP 8090 (POV World)  ↔ WS 9090                            │
│    TCP 9090 (POV Agents) ↔ WS 10090                           │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 │ TCP
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                  PYTHON SIMULATION SERVER                       │
│                                                                 │
│  RendererClient (currently send-only):                         │
│    - Sends VoxelGrid on port 8080                              │
│    - Sends Agents on port 8081                                 │
│    - Sends POV data on ports 8090+, 9090+                      │
│                                                                 │
│  NEW: Command Receiver (needs implementation):                 │
│    - Listen for incoming GuiCommand from proxy                 │
│    - Process keyboard inputs (A, B, C, D, R)                   │
│    - Control simulation based on browser inputs                │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Downstream (Simulation → Browser)

1. **Python Simulation** serializes data using bincode
   - Format: `[4-byte length (u32 LE)][bincode payload]`
   - Types: VoxelGrid, HashMap<usize, Agent>, PovData

2. **Sends via TCP** to localhost ports:
   - 8080: VoxelGrid (world state)
   - 8081: HashMap<usize, Agent> (agent positions)
   - 8090+: PovData (first-person view world)
   - 9090+: PovData (first-person view agents)

3. **WebSocket Proxy** receives TCP data:
   - Accepts TCP connections on 8080, 8081, etc.
   - Forwards frames as-is to WebSocket clients
   - WebSocket ports = TCP port + 1000 (8080 → 9080)

4. **Browser (WASM)** receives via WebSocket:
   - `network_ws.rs::NetworkSubscriber::onmessage` callback triggered
   - Parses [4-byte len][bincode payload]
   - Deserializes using bincode
   - Sends to crossbeam_channel for Bevy to consume

### Upstream (Browser → Simulation) - NEW!

1. **Browser captures keyboard input**:
   - Bevy's `ButtonInput<KeyCode>` system detects key presses
   - Keys: A, B, C, D (move agents), R (regenerate world)
   - Creates `GuiCommand` enum variant

2. **Sends via WebSocket**:
   - Calls `network_ws.rs::NetworkSubscriber::send(command)`
   - Serializes GuiCommand to bincode
   - Adds [4-byte length] prefix
   - Sends binary message via WebSocket.send_with_u8_array()

3. **WebSocket Proxy** receives and forwards:
   - Receives binary WebSocket message
   - Forwards [4-byte len][bincode payload] to TCP connection
   - Sends to Python simulation server

4. **Python Simulation** processes command:
   - **TODO**: Needs new TCP listener for receiving commands
   - Deserializes GuiCommand
   - Executes corresponding simulation action

## File Changes Made

### 1. voxelsim-renderer/src/network_ws.rs (NEW FILE)

Complete WebSocket implementation for WASM builds:

```rust
pub struct NetworkSubscriber<T> {
    sender: Sender<T>,
    port: u16,
    addr: String,
    websocket: Arc<Mutex<Option<WebSocket>>>,  // Store for bidirectional use
    _phantom: std::marker::PhantomData<T>,
}

impl<T> NetworkSubscriber<T> {
    pub fn new(addr: String, port: u16) -> (Self, Receiver<T>)

    // NEW: Send data back to proxy/simulation
    pub fn send<S: Serialize>(&self, data: &S) -> Result<(), String>
}

impl<T: DeserializeOwned + Send + Sync> NetworkSubscriber<T> {
    pub fn start(&mut self)  // Connects WebSocket, sets up callbacks
}
```

**Key features:**
- Same interface as TCP-based `network.rs`
- Stores WebSocket reference for bidirectional use
- `send()` method for upstream communication
- Uses browser event loop (no async/Tokio)

### 2. voxelsim-renderer/src/world.rs

**Made GuiCommand serializable:**

```rust
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum GuiCommand {
    RegenerateWorld,
    MoveAgentA,
    MoveAgentB,
    MoveAgentC,
    MoveAgentD,
}
```

This allows GuiCommand to be sent through WebSocket back to the simulation.

### 3. voxelsim-renderer/Cargo.toml

**Added WebSocket features to web-sys:**

```toml
web-sys = { version = "0.3.104", features = [
    "Window",
    "Document",
    "console",
    "WebSocket",      # NEW
    "BinaryType",     # NEW
    "MessageEvent",   # NEW
    "ErrorEvent",     # NEW
    "CloseEvent"      # NEW
] }
```

## Remaining Work

### 1. WebSocket Proxy Server

**Need to create:** `websocket-proxy/` directory with Node.js or Python proxy

**Requirements:**
- Listen on TCP ports 8080, 8081, 8090+, 9090+ (from simulation)
- Listen on WebSocket ports 9080, 9081, 9090+, 10090+ (from browser)
- **Bidirectional forwarding:**
  - Downstream: TCP → WebSocket (already designed)
  - Upstream: WebSocket → TCP (NEW requirement)
- Handle multiple simultaneous connections
- Preserve binary [4-byte len][payload] format

**Suggested implementation:** Node.js with `ws` library or Python with `websockets`

### 2. Integrate network_ws into Renderer

**Files to modify:**
- `src/main.rs`: Import network_ws module conditionally
- `src/world.rs`: Use network_ws::NetworkSubscriber on WASM
- `src/pov.rs`: Use network_ws::NetworkSubscriber on WASM

**Example pattern:**
```rust
#[cfg(not(target_arch = "wasm32"))]
use crate::network::NetworkSubscriber;

#[cfg(target_arch = "wasm32")]
use crate::network_ws::NetworkSubscriber;
```

### 3. Python Simulation Command Receiver

**Need to add to voxelsim/src/network.rs:**

```rust
pub struct SimulationServer {
    command_listener: TcpListener,  // NEW: Listen for commands
    // ... existing fields ...
}

impl SimulationServer {
    pub fn receive_command<T: DeserializeOwned>() -> Result<T>  // NEW
}
```

**Python side needs:**
- New TCP socket to receive commands from proxy
- Deserialize GuiCommand using bincode
- Execute simulation actions (move agents, regenerate world)

### 4. Browser UI Overlay (Future)

- HTML/CSS overlay on canvas for controls
- JavaScript event handlers
- Send custom commands beyond just keyboard (buttons, sliders, etc.)

## Testing Plan

### Phase 1: Static Demo (DONE)
✅ Verify Bevy renders in browser without networking
✅ Confirmed WASM compilation works

### Phase 2: Unidirectional (Sim → Browser)
- [ ] Create WebSocket proxy server
- [ ] Integrate network_ws into world.rs
- [ ] Run Python simulation
- [ ] Verify VoxelGrid and Agents render in browser

### Phase 3: Bidirectional (Browser ↔ Sim)
- [ ] Add command receiver to Python simulation
- [ ] Test keyboard inputs in browser
- [ ] Verify commands reach simulation
- [ ] Confirm simulation responds to browser inputs

### Phase 4: POV Streams
- [ ] Add POV data streams (ports 8090+, 9090+)
- [ ] Test first-person camera views
- [ ] Verify camera switching works

## Technical Notes

### Why WebSocket + Proxy Instead of Direct TCP?

**Browsers cannot use TCP directly** - they're sandboxed for security. WebSocket is the browser-native bidirectional protocol.

### Why Not Just Use WebSocket Everywhere?

**Python simulation uses existing TCP code** that works well. The proxy allows incremental migration without rewriting the simulation server.

### Binary Format Compatibility

Both TCP and WebSocket use the **exact same binary format**:
```
[4 bytes: message length as u32 little-endian]
[N bytes: bincode-serialized payload]
```

This allows the proxy to forward frames **without parsing or re-encoding** - just raw byte forwarding.

### crossbeam_channel Works in WASM

Unlike Tokio, `crossbeam_channel` is pure Rust with no OS dependencies, so it works perfectly in WASM. This is why the interface remains identical between native and WASM builds.

## Port Reference

| Service | Native (TCP) | WASM (WebSocket) |
|---------|-------------|------------------|
| VoxelGrid | 8080 | 9080 |
| Agents | 8081 | 9081 |
| POV World 0 | 8090 | 9090 |
| POV Agents 0 | 9090 | 10090 |
| POV World 1 | 8091 | 9091 |
| POV Agents 1 | 9091 | 10091 |
| ... | ... | ... |

**Pattern:** WebSocket port = TCP port + 1000

## Next Immediate Steps

1. **Create WebSocket proxy server** (websocket-proxy/)
2. **Test proxy with simple echo** (verify bidirectional forwarding)
3. **Integrate network_ws** into world.rs
4. **Test end-to-end**: Python → Proxy → Browser (downstream)
5. **Add Python command receiver**
6. **Test end-to-end**: Browser → Proxy → Python (upstream)
