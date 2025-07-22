use crate::viewport::{CameraProjection, VirtualGrid};
use crate::{Agent, VoxelGrid};
use serde::{Deserialize, Serialize};

use std::collections::HashMap;
use std::io::Write;
use std::net::TcpStream;

/// Network client for sending data to the renderer
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct RendererClient {
    pov_streams: Vec<TcpStream>,
    world_stream: Option<TcpStream>,
    agent_stream: Option<TcpStream>,
    host: String,
    world_port: u16,
    agent_port: u16,
    pov_start_port: u16,
}

impl RendererClient {
    pub fn new(host: &str, world_port: u16, agent_port: u16, pov_start_port: u16) -> Self {
        Self {
            pov_streams: Vec::new(),
            world_stream: None,
            agent_stream: None,
            host: host.to_string(),
            world_port,
            agent_port,
            pov_start_port,
        }
    }

    pub fn connect(&mut self, pov_count: u16) -> Result<(), Box<dyn std::error::Error>> {
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

        self.pov_streams.clear();
        for i in 0..pov_count {
            let port = self.pov_start_port + i;
            if let Ok(stream) = TcpStream::connect(format!("{}:{}", self.host, port)) {
                self.pov_streams.push(stream);
            }
        }

        Ok(())
    }

    pub fn send_world(&mut self, data: &VoxelGrid) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref mut stream) = self.world_stream {
            Self::send_data(data, stream)
        } else {
            Err("Not connected to world port".into())
        }
    }

    pub fn send_agents(
        &mut self,
        data: &HashMap<usize, Agent>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref mut stream) = self.agent_stream {
            Self::send_data(data, stream)
        } else {
            Err("Not connected to agent port".into())
        }
    }

    pub fn send_pov(
        &mut self,
        stream_idx: usize,
        data: &PovData,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref mut stream) = self.pov_streams.get_mut(stream_idx) {
            Self::send_data(data, stream)
        } else {
            Err("Not connected to pov port".into())
        }
    }

    pub fn send_data<D>(data: D, stream: &mut TcpStream) -> Result<(), Box<dyn std::error::Error>>
    where
        D: serde::Serialize,
    {
        let bin = bincode::serde::encode_to_vec(data, bincode::config::standard())?;

        // Send length prefix (4 bytes, little-endian)
        let len = (bin.len() as u32).to_le_bytes();
        stream.write_all(&len)?;

        // Send the actual data
        stream.write_all(&bin)?;
        stream.flush()?;

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PovData {
    pub virtual_world: VirtualGrid,
    pub agent_id: usize,
    pub proj: CameraProjection,
}
