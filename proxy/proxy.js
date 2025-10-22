#!/usr/bin/env node

/**
 * VoxelSim WebSocket Proxy Server
 *
 * Bridges TCP connections from Python simulation to WebSocket connections for browser WASM renderer.
 *
 * Data Flow:
 *   Python (TCP Client) → This Proxy (TCP Server + WebSocket Server) → Browser (WebSocket Client)
 *
 * Protocol:
 *   - Each message: [4-byte length (little-endian)][bincode payload]
 *   - This proxy buffers TCP stream and forwards complete messages only
 *
 * Port Mapping:
 *   - TCP :8080 → WebSocket :9080 (VoxelGrid world data)
 *   - TCP :8081 → WebSocket :9081 (Agent data)
 *   - TCP :8090 → WebSocket :9090 (POV world data)
 *   - TCP :9090 → WebSocket :10090 (POV agent data)
 */

const net = require('net');
const { WebSocketServer } = require('ws');

/**
 * Creates a bidirectional proxy between TCP and WebSocket
 * @param {number} tcpPort - Port to listen for TCP connections from Python
 * @param {number} wsPort - Port to serve WebSocket connections to browser
 * @param {string} label - Description for logging
 */
function createProxy(tcpPort, wsPort, label) {
    let pythonSocket = null;
    const websockets = new Set();

    // Buffer for incomplete TCP messages
    let buffer = Buffer.alloc(0);

    // 1. TCP Server - Accept connections from Python
    const tcpServer = net.createServer((socket) => {
        console.log(`[${label}] Python connected via TCP :${tcpPort}`);
        pythonSocket = socket;
        buffer = Buffer.alloc(0); // Reset buffer on new connection

        // Forward data: Python (TCP) → All browsers (WebSocket)
        socket.on('data', (chunk) => {
            // Append new data to buffer
            buffer = Buffer.concat([buffer, chunk]);

            // Process all complete messages in buffer
            while (buffer.length >= 4) {
                // Read length prefix (4 bytes, little-endian)
                const msgLen = buffer.readUInt32LE(0);
                const totalLen = 4 + msgLen;

                // Check if we have the complete message
                if (buffer.length < totalLen) {
                    // Not enough data yet, wait for more
                    break;
                }

                // Extract complete message (including 4-byte prefix)
                const completeMessage = buffer.slice(0, totalLen);

                // Remove processed message from buffer
                buffer = buffer.slice(totalLen);

                // Forward to all connected browsers
                websockets.forEach((ws) => {
                    if (ws.readyState === ws.OPEN) {
                        ws.send(completeMessage, { binary: true });
                    }
                });
            }
        });

        socket.on('error', (err) => {
            console.error(`[${label}] Python TCP error:`, err.message);
        });

        socket.on('close', () => {
            console.log(`[${label}] Python disconnected from TCP :${tcpPort}`);
            pythonSocket = null;
            buffer = Buffer.alloc(0);
        });
    });

    tcpServer.listen(tcpPort, '127.0.0.1', () => {
        console.log(`[${label}] TCP server listening on :${tcpPort}`);
    });

    tcpServer.on('error', (err) => {
        console.error(`[${label}] TCP server error:`, err.message);
        process.exit(1);
    });

    // 2. WebSocket Server - Accept connections from browsers
    const wss = new WebSocketServer({ port: wsPort });

    wss.on('connection', (ws) => {
        console.log(`[${label}] Browser connected via WebSocket :${wsPort}`);
        websockets.add(ws);

        // Forward data: Browser (WebSocket) → Python (TCP)
        ws.on('message', (data) => {
            if (pythonSocket && !pythonSocket.destroyed) {
                pythonSocket.write(data);
            }
        });

        ws.on('error', (err) => {
            console.error(`[${label}] WebSocket error:`, err.message);
        });

        ws.on('close', () => {
            console.log(`[${label}] Browser disconnected from WebSocket :${wsPort}`);
            websockets.delete(ws);
        });
    });

    wss.on('error', (err) => {
        console.error(`[${label}] WebSocket server error:`, err.message);
        process.exit(1);
    });

    console.log(`[${label}] Proxy ready: TCP :${tcpPort} ←→ WebSocket :${wsPort}`);
}

// Start all proxy bridges
console.log('VoxelSim WebSocket Proxy Server Starting...\n');

createProxy(8080, 9080, 'VoxelGrid');
createProxy(8081, 9081, 'Agents');
createProxy(8090, 9090, 'POV World');
createProxy(9090, 10090, 'POV Agents');

console.log('\n✓ All proxies running!');
console.log('  Python can connect to TCP ports: 8080, 8081, 8090, 9090');
console.log('  Browser can connect to WebSocket ports: ws://localhost:9080, etc.\n');

// Keep the process alive
process.on('SIGINT', () => {
    console.log('\nShutting down proxies...');
    process.exit(0);
});
