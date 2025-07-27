use crossbeam_channel::{Receiver, Sender};
use serde::de::DeserializeOwned;
use std::env;
use std::marker::PhantomData;
use std::sync::Arc;
use tokio::io::AsyncReadExt;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use voxelsim::viewport::VirtualGrid;
use voxelsim::{Agent, VoxelGrid};

pub mod pov;

pub mod convert;
pub mod network;
pub mod render;
pub mod world;

fn main() {
    println!("Starting VoxelSim Renderer...");

    // Collect all the command‐line arguments into a Vec<String>
    let args: Vec<String> = env::args().collect();

    // Look for "--mode" in the args
    if let Some(pos) = args.iter().position(|arg| arg == "--virtual") {
        // Make sure there's something after "--mode"
        if let Some(val) = args.get(pos + 1) {
            if let Ok(num) = val.parse::<u16>() {
                // <-- YOUR special‐mode branch
                println!("Running virtual viewport with offset {}", num);
                pov::run_pov_server(num); // ... do your special work here ...
            } else {
                println!("Please specify a virtual viewport offset.");
            }
        } else {
            eprintln!("Error: '--virtual' was provided but no port offset followed it.");
        }
    } else {
        println!("Running in world mode.");
        world::run_world_server();
    }
}
