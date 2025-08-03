use std::{error::Error, io};

use mavlink::{
    MavHeader,
    ardupilotmega::{HEARTBEAT_DATA, LED_CONTROL_DATA, MavMessage},
    error::MessageWriteError,
};

mod ctrl {
    pub const LED_CONTROL_PATTERN_OFF: u8 = 0;
    pub const LED_CONTROL_PATTERN_CUSTOM: u8 = 255;
}

pub struct ConnectionInterface {
    conn: Box<dyn mavlink::MavConnection<MavMessage>>,
    target_system: u8,
    target_component: u8,
}

impl ConnectionInterface {
    pub fn connect(address: &str) -> io::Result<Self> {
        let conn = mavlink::connect(address)?;
        Ok(Self {
            conn,
            target_system: 1,
            target_component: 1,
        })
    }

    pub fn send_heartbeat(&self) -> Result<usize, MessageWriteError> {
        self.conn.send(
            &MavHeader::default(),
            &MavMessage::HEARTBEAT(HEARTBEAT_DATA::DEFAULT),
        )
    }

    pub fn send_led(&self, colour: [u8; 3], blink_rate: u8) -> Result<usize, MessageWriteError> {
        // build a custom‐pattern LED_CONTROL message
        let mut bytes = [0u8; 24];
        bytes[0..2].copy_from_slice(&colour);
        bytes[3] = blink_rate; // 2 Hz blink rate
        self.conn.send(
            &MavHeader::default(),
            &MavMessage::LED_CONTROL(LED_CONTROL_DATA {
                target_system: 1,
                target_component: 1,
                instance: 255,
                pattern: ctrl::LED_CONTROL_PATTERN_CUSTOM,
                custom_len: 4,
                custom_bytes: bytes,
            }),
        )
    }

    pub fn send_armed(&self, armed: bool) -> Result<usize, MessageWriteError> {}

    pub fn send_waypoint(&self) -> Result<usize, MessageWriteError> {}
}
