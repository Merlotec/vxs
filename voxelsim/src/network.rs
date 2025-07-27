use crate::viewport::{CameraOrientation, CameraProjection, VirtualGrid};
use crate::{Agent, VoxelGrid};
use serde::{Deserialize, Serialize};

use std::collections::HashMap;
use std::io::Write;
use std::net::TcpStream;

pub struct PovStream {
    virtual_world_stream: TcpStream,
    agent_stream: TcpStream,
}

/// Network client for sending data to the renderer
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
pub struct RendererClient {
    // First stream is the virtual world data, second is tth
    pov_streams: Vec<PovStream>,
    world_stream: Option<TcpStream>,
    agent_stream: Option<TcpStream>,
    host: String,
    world_port: u16,
    agent_port: u16,
    pov_start_port: u16,
    pov_agent_start_port: u16,
}

impl RendererClient {
    pub fn new(
        host: &str,
        world_port: u16,
        agent_port: u16,
        pov_start_port: u16,
        pov_agent_start_port: u16,
    ) -> Self {
        Self {
            pov_streams: Vec::new(),
            world_stream: None,
            agent_stream: None,
            host: host.to_string(),
            world_port,
            agent_port,
            pov_start_port,
            pov_agent_start_port,
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
            let agent_port = self.pov_agent_start_port + i;
            if let (Ok(virtual_world_stream), Ok(agent_stream)) = (
                TcpStream::connect(format!("{}:{}", self.host, port)),
                TcpStream::connect(format!("{}:{}", self.host, agent_port)),
            ) {
                self.pov_streams.push(PovStream {
                    virtual_world_stream,
                    agent_stream,
                });
            } else {
                return Err(format!("Could not connect pov/agent stream for agent {}", i).into());
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
            Self::send_data(data, stream)?;
        } else {
            return Err("Not connected to agent port".into());
        }
        for stream in self.pov_streams.iter_mut() {
            Self::send_data(data, &mut stream.agent_stream)?;
        }
        Ok(())
    }

    pub fn send_pov(
        &mut self,
        stream_idx: usize,
        data: &PovData,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref mut stream) = self.pov_streams.get_mut(stream_idx) {
            Self::send_data(data, &mut stream.virtual_world_stream)
        } else {
            Err("Not connected to pov port".into())
        }
    }

    pub fn send_pov_ref(
        &mut self,
        stream_idx: usize,
        data: &PovDataRef,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref mut stream) = self.pov_streams.get_mut(stream_idx) {
            Self::send_data(data, &mut stream.virtual_world_stream)
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
    pub orientation: CameraOrientation,
}

#[derive(Debug, Clone, Serialize)]
pub struct PovDataRef<'a> {
    pub virtual_world: &'a VirtualGrid,
    pub agent_id: usize,
    pub proj: CameraProjection,
    pub orientation: CameraOrientation,
}
