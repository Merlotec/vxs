use voxelsim::{Action, Agent, viewport::VirtualGrid};

pub mod manual;

pub trait ControlBackend {
    fn update_action(&mut self, agent: &Agent, fw: &VirtualGrid) -> Option<Action>;
}
