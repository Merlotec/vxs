pub mod pipeline;
pub mod rasterizer;

use voxelsim::{TerrainConfig, VoxelGrid};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

use std::sync::Arc;

use pipeline::State;

#[derive(Default)]
struct App {
    state: Option<State>,
    world: VoxelGrid,
}

impl App {
    pub fn new(world: VoxelGrid) -> Self {
        Self { state: None, world }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Create window object
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("voxelsim-compute"))
                .unwrap(),
        );

        let state = pollster::block_on(State::create(window.clone(), &self.world));

        self.state = Some(state);

        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let state = self.state.as_mut().unwrap();
        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                state.render().unwrap();

                // Emits a new redraw requested event.
                state.window.request_redraw();
            }
            WindowEvent::Resized(size) => {
                // Reconfigures the size of the surface. We do not re-render
                // here as this event is always followed up by redraw request.
                state.resize(size);
            }
            _ => (),
        }
    }
}
// Main entry point for the application
fn main() {
    // Set up logging
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();

    // When the current loop iteration finishes, immediately begin a new
    // iteration regardless of whether or not new events are available to
    // process. Preferred for applications that want to render as fast as
    // possible, like games.
    event_loop.set_control_flow(ControlFlow::Poll);

    // When the current loop iteration finishes, suspend the thread until
    // another event arrives. Helps keeping CPU utilization low if nothing
    // is happening, which is preferred if the application might be idling in
    // the background.
    // event_loop.set_control_flow(ControlFlow::Wait);
    let mut world = VoxelGrid::default();
    world.generate_terrain(&TerrainConfig {
        seed: 100,
        ..Default::default()
    });

    let mut app = App::new(world);
    event_loop.run_app(&mut app).unwrap();
}
