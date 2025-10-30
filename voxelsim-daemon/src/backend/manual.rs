use voxelsim::Action;

use crate::backend::ControlBackend;

/// Abstracts over how the signal is received.
/// May want to be able to receive signal from both web sockets and mavlink radio signals.
pub trait ActionReceiver {
    fn try_recv_signal(&self) -> Option<Action>;
}

pub struct ManualBackend<R: ActionReceiver> {
    receiver: R,
}

impl<R: ActionReceiver> ManualBackend<R> {
    pub fn new(receiver: R) -> Self {
        Self { receiver }
    }
}

impl<R: ActionReceiver> ControlBackend for ManualBackend<R> {
    fn update_action(
        &mut self,
        agent: &voxelsim::Agent,
        fw: &voxelsim::VoxelGrid,
    ) -> Option<Action> {
        self.receiver.try_recv_signal()
    }
}
