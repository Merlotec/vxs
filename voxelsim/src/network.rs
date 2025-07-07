use crate::{Agent, TerrainConfig, VoxelGrid};
use serde::{Deserialize, Serialize};

use std::io::Write;
use std::net::TcpStream;

/// Network client for sending data to the renderer
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct RendererClient {
    world_stream: Option<TcpStream>,
    agent_stream: Option<TcpStream>,
    host: String,
    world_port: u16,
    agent_port: u16,
}

impl RendererClient {
    pub fn new(host: &str, world_port: u16, agent_port: u16) -> Self {
        Self {
            world_stream: None,
            agent_stream: None,
            host: host.to_string(),
            world_port,
            agent_port,
        }
    }

    pub fn connect(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Connect to world port
        self.world_stream = Some(TcpStream::connect(format!(
            "{}:{}",
            self.host, self.world_port
        ))?);

        // Connect to agent port
        self.agent_stream = Some(TcpStream::connect(format!(
            "{}:{}",
            self.host, self.agent_port
        ))?);

        Ok(())
    }

    pub fn send_world(&mut self, data: &VoxelGrid) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref mut stream) = self.world_stream {
            let bin = bincode::serde::encode_to_vec(data, bincode::config::standard())?;

            // Send length prefix (4 bytes, little-endian)
            let len = (bin.len() as u32).to_le_bytes();
            stream.write_all(&len)?;

            // Send the actual data
            stream.write_all(&bin)?;
            stream.flush()?;
        } else {
            return Err("Not connected to world port".into());
        }
        Ok(())
    }

    pub fn send_agents(&mut self, data: &[Agent]) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref mut stream) = self.agent_stream {
            let bin = bincode::serde::encode_to_vec(data, bincode::config::standard())?;

            // Send length prefix (4 bytes, little-endian)
            let len = (bin.len() as u32).to_le_bytes();
            stream.write_all(&len)?;

            // Send the actual data
            stream.write_all(&bin)?;
            stream.flush()?;
        } else {
            return Err("Not connected to agent port".into());
        }
        Ok(())
    }

    pub fn disconnect(&mut self) {
        self.world_stream = None;
        self.agent_stream = None;
    }
}

impl Drop for RendererClient {
    fn drop(&mut self) {
        self.disconnect();
    }
}
