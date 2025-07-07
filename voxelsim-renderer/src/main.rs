use crossbeam_channel::{Receiver, Sender};
use std::sync::Arc;
use tokio::io::AsyncReadExt;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use voxelsim::{Agent, VoxelGrid};

pub mod render;
use render::GuiCommand;

/// Network configuration for receiving data from simulator
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    pub world_port: u16,
    pub agent_port: u16,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            world_port: 8080,
            agent_port: 8081,
        }
    }
}

/// Main simulation state for the renderer
pub struct RendererSimulation {
    pub world_sender: Sender<VoxelGrid>,
    pub agent_sender: Sender<Vec<Agent>>,
    pub quit_sender: Sender<()>,
    pub gui_sender: Sender<GuiCommand>,
}

impl RendererSimulation {
    pub fn new() -> (Self, RendererReceivers) {
        let (world_sender, world_receiver) = crossbeam_channel::unbounded::<VoxelGrid>();
        let (agent_sender, agent_receiver) = crossbeam_channel::unbounded::<Vec<Agent>>();
        let (gui_sender, gui_receiver) = crossbeam_channel::unbounded::<GuiCommand>();
        let (quit_sender, quit_receiver) = crossbeam_channel::unbounded::<()>();

        let simulation = Self {
            world_sender,
            agent_sender,
            quit_sender,
            gui_sender: gui_sender.clone(),
        };

        let receivers = RendererReceivers {
            world_receiver,
            agent_receiver,
            quit_receiver,
            gui_sender,
        };

        (simulation, receivers)
    }

    /// Send world data to the render thread
    pub fn send_world(&self, world: VoxelGrid) {
        let _ = self.world_sender.send(world);
    }

    /// Send agent data to the render thread
    pub fn send_agents(&self, agents: Vec<Agent>) {
        let _ = self.agent_sender.send(agents);
    }

    /// Send quit signal
    pub fn quit(&self) {
        let _ = self.quit_sender.send(());
    }
}

pub struct RendererReceivers {
    pub world_receiver: Receiver<VoxelGrid>,
    pub agent_receiver: Receiver<Vec<Agent>>,
    pub gui_sender: Sender<GuiCommand>,
    pub quit_receiver: Receiver<()>,
}

impl RendererReceivers {
    pub fn start_render(self) {
        render::begin_render(
            self.world_receiver,
            self.agent_receiver,
            self.gui_sender,
            self.quit_receiver,
        );
    }
}

/// Network listener for receiving data from simulator
pub struct NetworkListener {
    world_port: u16,
    agent_port: u16,
    simulation: Arc<RendererSimulation>,
}

impl NetworkListener {
    pub fn new(world_port: u16, agent_port: u16, simulation: Arc<RendererSimulation>) -> Self {
        Self {
            world_port,
            agent_port,
            simulation,
        }
    }

    /// Start the network listener
    pub async fn start(&self) {
        println!("Starting network listener...");
        println!("World data port: {}", self.world_port);
        println!("Agent data port: {}", self.agent_port);

        // Start world data listener
        let world_sim = Arc::clone(&self.simulation);
        let world_port = self.world_port;
        tokio::spawn(async move {
            if let Err(e) = Self::listen_world_data(world_port, world_sim).await {
                eprintln!("World data listener error: {}", e);
            }
        });

        // Start agent data listener
        let agent_sim = Arc::clone(&self.simulation);
        let agent_port = self.agent_port;
        tokio::spawn(async move {
            if let Err(e) = Self::listen_agent_data(agent_port, agent_sim).await {
                eprintln!("Agent data listener error: {}", e);
            }
        });

        // Keep the async context alive
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
    }

    /// Listen for world data on the specified port
    async fn listen_world_data(
        port: u16,
        simulation: Arc<RendererSimulation>,
    ) -> tokio::io::Result<()> {
        let listener = TcpListener::bind(format!("127.0.0.1:{}", port)).await?;
        println!("World data listener bound to port {}", port);

        loop {
            let (socket, addr) = listener.accept().await?;
            println!("World data connection from: {}", addr);

            let sim = Arc::clone(&simulation);
            tokio::spawn(async move {
                if let Err(e) = Self::handle_world_connection(socket, sim).await {
                    eprintln!("World connection error: {}", e);
                }
            });
        }
    }

    /// Listen for agent data on the specified port
    async fn listen_agent_data(
        port: u16,
        simulation: Arc<RendererSimulation>,
    ) -> tokio::io::Result<()> {
        let listener = TcpListener::bind(format!("127.0.0.1:{}", port)).await?;
        println!("Agent data listener bound to port {}", port);

        loop {
            let (socket, addr) = listener.accept().await?;
            println!("Agent data connection from: {}", addr);

            let sim = Arc::clone(&simulation);
            tokio::spawn(async move {
                if let Err(e) = Self::handle_agent_connection(socket, sim).await {
                    eprintln!("Agent connection error: {}", e);
                }
            });
        }
    }

    /// Handle a world data connection
    async fn handle_world_connection(
        mut socket: TcpStream,
        simulation: Arc<RendererSimulation>,
    ) -> tokio::io::Result<()> {
        loop {
            // 1) Read the 4-byte length prefix
            let mut len_buf = [0u8; 4];
            if let Err(e) = socket.read_exact(&mut len_buf).await {
                if e.kind() == tokio::io::ErrorKind::UnexpectedEof {
                    break; // connection closed cleanly
                } else {
                    return Err(e);
                }
            }
            let msg_len = u32::from_le_bytes(len_buf) as usize;

            // 2) Read the frame payload
            let mut buf = vec![0u8; msg_len];
            socket.read_exact(&mut buf).await?;

            // 3) Deserialize with bincode
            match bincode::serde::decode_from_slice::<VoxelGrid, _>(
                &buf,
                bincode::config::standard(),
            ) {
                Ok((world_data, _)) => {
                    simulation.send_world(world_data);
                    println!("Received bincode-encoded voxel grid ({} bytes)", msg_len);
                }
                Err(e) => {
                    eprintln!("Failed to deserialize VoxelGrid: {}", e);
                }
            }
        }

        Ok(())
    }

    /// Handle an agent data connection
    async fn handle_agent_connection(
        mut socket: TcpStream,
        simulation: Arc<RendererSimulation>,
    ) -> tokio::io::Result<()> {
        loop {
            // 1) Read the 4-byte length prefix
            let mut len_buf = [0u8; 4];
            if let Err(e) = socket.read_exact(&mut len_buf).await {
                if e.kind() == tokio::io::ErrorKind::UnexpectedEof {
                    break; // connection closed cleanly
                } else {
                    return Err(e);
                }
            }
            let msg_len = u32::from_le_bytes(len_buf) as usize;

            // 2) Read the frame payload
            let mut buf = vec![0u8; msg_len];
            socket.read_exact(&mut buf).await?;

            // 3) Deserialize with bincode
            match bincode::serde::decode_from_slice::<Vec<Agent>, _>(
                &buf,
                bincode::config::standard(),
            ) {
                Ok((agents_data, _)) => {
                    simulation.send_agents(agents_data);
                    println!("Received bincode-encoded voxel grid ({} bytes)", msg_len);
                }
                Err(e) => {
                    eprintln!("Failed to deserialize VoxelGrid: {}", e);
                }
            }
        }

        Ok(())
    }
}

fn main() {
    println!("Starting VoxelSim Renderer...");

    // Create simulation channels
    let (simulation, receivers) = RendererSimulation::new();
    let simulation = Arc::new(simulation);

    // Network configuration
    let network_config = NetworkConfig::default();
    println!("Listening for connections:");
    println!("  World data: 127.0.0.1:{}", network_config.world_port);
    println!("  Agent data: 127.0.0.1:{}", network_config.agent_port);

    // Start network listeners in background thread
    println!("Starting network listeners...");
    let network_listener = NetworkListener::new(
        network_config.world_port,
        network_config.agent_port,
        Arc::clone(&simulation),
    );

    // Start network listener in background thread with its own tokio runtime
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            network_listener.start().await;
        });
    });

    println!("VoxelSim Renderer is now running!");
    println!("Waiting for data from simulator...");
    println!("Press Ctrl+C or close window to exit.");

    // Run the UI on the main thread (blocking)
    receivers.start_render();
}
