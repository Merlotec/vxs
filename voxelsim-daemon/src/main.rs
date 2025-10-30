pub mod backend;
pub mod controller;
pub mod server;

fn main() {
    if let Err(e) = server::run_server("127.0.0.1:7000") {
        eprintln!("Server error: {}", e);
    }
}
