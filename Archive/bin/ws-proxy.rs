//! WebSocket Proxy for VoxelSim Renderer
//!
//! Bridges TCP (Python simulation) ↔ WebSocket (Browser WASM renderer)
//!
//! Port mapping:
//! - TCP 8080 (VoxelGrid)  ↔ WS 18080
//! - TCP 8081 (Agents)     ↔ WS 18081
//! - TCP 8090 (POV World)  ↔ WS 18090
//! - TCP 9090 (POV Agents) ↔ WS 19090
//!
//! Usage:
//!     cargo run --bin ws-proxy
//!
//! Then:
//! 1. Run your Python simulation (sends to TCP 8080, 8081, etc.)
//! 2. Open browser with WASM renderer (connects to WS 18080, 18081, etc.)

use std::net::SocketAddr;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::broadcast;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Configuration for a single TCP ↔ WebSocket bridge
#[derive(Debug, Clone)]
struct BridgeConfig {
    tcp_host: String,
    tcp_port: u16,
    ws_port: u16,
}

impl BridgeConfig {
    fn new(tcp_port: u16, ws_port: u16) -> Self {
        Self {
            tcp_host: "127.0.0.1".to_string(),
            tcp_port,
            ws_port,
        }
    }
}

/// Handles a single TCP ↔ WebSocket bridge
struct Bridge {
    config: BridgeConfig,
    // Broadcast channel to send TCP data to all WebSocket clients
    tx: broadcast::Sender<Vec<u8>>,
}

impl Bridge {
    fn new(config: BridgeConfig) -> Self {
        let (tx, _rx) = broadcast::channel(100);
        Self { config, tx }
    }

    /// Start the bridge: WebSocket server + TCP client
    async fn start(self: Arc<Self>) -> Result<()> {
        println!(
            "Starting bridge: TCP :{}  ↔  WS :{}",
            self.config.tcp_port, self.config.ws_port
        );

        // Start WebSocket server
        let ws_task = {
            let bridge = Arc::clone(&self);
            tokio::spawn(async move {
                if let Err(e) = bridge.run_websocket_server().await {
                    eprintln!("WebSocket server error on :{}: {}", bridge.config.ws_port, e);
                }
            })
        };

        // Start TCP client (connects to Python simulation)
        let tcp_task = {
            let bridge = Arc::clone(&self);
            tokio::spawn(async move {
                bridge.run_tcp_client().await;
            })
        };

        // Wait for both tasks
        tokio::try_join!(ws_task, tcp_task)?;
        Ok(())
    }

    /// Run WebSocket server that accepts browser connections
    async fn run_websocket_server(&self) -> Result<()> {
        let addr: SocketAddr = format!("127.0.0.1:{}", self.config.ws_port).parse()?;
        let listener = TcpListener::bind(&addr).await?;
        println!("WebSocket server listening on ws://localhost:{}", self.config.ws_port);

        loop {
            let (stream, client_addr) = listener.accept().await?;
            println!("WS client connected on :{} from {}", self.config.ws_port, client_addr);

            let mut rx = self.tx.subscribe();

            // Handle WebSocket connection
            tokio::spawn(async move {
                // Perform WebSocket handshake
                let ws_stream = match tokio_tungstenite::accept_async(stream).await {
                    Ok(ws) => ws,
                    Err(e) => {
                        eprintln!("WebSocket handshake failed: {}", e);
                        return;
                    }
                };

                let (mut ws_tx, mut ws_rx) = ws_stream.split();

                // Task 1: Forward TCP data to WebSocket
                let forward_task = tokio::spawn(async move {
                    while let Ok(data) = rx.recv().await {
                        use tokio_tungstenite::tungstenite::Message;
                        if let Err(e) = futures_util::SinkExt::send(&mut ws_tx, Message::Binary(data)).await {
                            eprintln!("Failed to send to WebSocket client: {}", e);
                            break;
                        }
                    }
                });

                // Task 2: Receive data from WebSocket (for bidirectional support)
                // TODO: Forward WebSocket → TCP for browser input commands
                let receive_task = tokio::spawn(async move {
                    use futures_util::StreamExt;
                    while let Some(msg) = ws_rx.next().await {
                        match msg {
                            Ok(msg) if msg.is_binary() => {
                                // Future: forward to TCP connection back to Python
                                println!("Received {} bytes from WebSocket (bidirectional not yet implemented)", msg.len());
                            }
                            Err(e) => {
                                eprintln!("WebSocket receive error: {}", e);
                                break;
                            }
                            _ => {}
                        }
                    }
                });

                // Wait for either task to complete
                tokio::select! {
                    _ = forward_task => {},
                    _ = receive_task => {},
                }

                println!("WS client disconnected");
            });
        }
    }

    /// Run TCP client that connects to Python simulation
    async fn run_tcp_client(&self) {
        let tcp_addr = format!("{}:{}", self.config.tcp_host, self.config.tcp_port);

        loop {
            println!("Connecting to TCP {}...", tcp_addr);

            match TcpStream::connect(&tcp_addr).await {
                Ok(mut stream) => {
                    println!("Connected to TCP :{}", self.config.tcp_port);

                    // Read frames from TCP and broadcast to WebSocket clients
                    loop {
                        // Read 4-byte length prefix
                        let mut len_buf = [0u8; 4];
                        match stream.read_exact(&mut len_buf).await {
                            Ok(_) => {}
                            Err(e) => {
                                eprintln!("TCP read error on :{}: {}", self.config.tcp_port, e);
                                break;
                            }
                        }

                        let frame_len = u32::from_le_bytes(len_buf) as usize;

                        // Read the payload
                        let mut payload = vec![0u8; frame_len];
                        match stream.read_exact(&mut payload).await {
                            Ok(_) => {}
                            Err(e) => {
                                eprintln!("TCP read payload error on :{}: {}", self.config.tcp_port, e);
                                break;
                            }
                        }

                        // Reconstruct full frame: [4-byte length][payload]
                        let mut full_frame = Vec::with_capacity(4 + frame_len);
                        full_frame.extend_from_slice(&len_buf);
                        full_frame.extend_from_slice(&payload);

                        // Broadcast to all WebSocket clients
                        let receiver_count = self.tx.receiver_count();
                        if receiver_count > 0 {
                            if let Err(e) = self.tx.send(full_frame) {
                                eprintln!("Failed to broadcast: {}", e);
                            }
                        }
                    }

                    println!("TCP connection closed on :{}", self.config.tcp_port);
                }
                Err(e) => {
                    eprintln!("Cannot connect to TCP :{}: {}", self.config.tcp_port, e);
                }
            }

            // Wait before reconnecting
            println!("Retrying TCP connection in 2 seconds...");
            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("VoxelSim WebSocket Proxy");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("Port Mappings:");
    println!("  TCP :8080  ↔  WS :18080  (VoxelGrid)");
    println!("  TCP :8081  ↔  WS :18081  (Agents)");
    println!("  TCP :8090  ↔  WS :18090  (POV World - optional)");
    println!("  TCP :9090  ↔  WS :19090  (POV Agents - optional)");
    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!();

    // Create bridge configurations
    let configs = vec![
        BridgeConfig::new(8080, 18080), // VoxelGrid
        BridgeConfig::new(8081, 18081), // Agents
        // Uncomment for POV support:
        // BridgeConfig::new(8090, 18090), // POV World
        // BridgeConfig::new(9090, 19090), // POV Agents
    ];

    // Create bridges
    let bridges: Vec<Arc<Bridge>> = configs
        .into_iter()
        .map(|config| Arc::new(Bridge::new(config)))
        .collect();

    // Start all bridges concurrently
    let tasks: Vec<_> = bridges
        .into_iter()
        .map(|bridge| {
            let bridge = Arc::clone(&bridge);
            tokio::spawn(async move {
                if let Err(e) = bridge.start().await {
                    eprintln!("Bridge error: {}", e);
                }
            })
        })
        .collect();

    // Wait for all tasks
    for task in tasks {
        task.await?;
    }

    Ok(())
}
