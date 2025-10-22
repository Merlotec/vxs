// WebSocket-based network subscriber for WASM builds
// This provides the same interface as network.rs but uses WebSockets instead of TCP
// Now supports BIDIRECTIONAL communication: receive data AND send commands back

use crossbeam_channel::{Receiver, Sender};
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::sync::{Arc, Mutex};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{BinaryType, ErrorEvent, MessageEvent, WebSocket};
use wasm_bindgen::JsValue;

/// WebSocket subscriber that receives data from a WebSocket server
/// and forwards it to a crossbeam channel for Bevy to consume.
///
/// Can also SEND data back through the WebSocket (bidirectional).
/// This allows browser UI inputs to be sent back to the simulation server.
pub struct NetworkSubscriber<T> {
    sender: Sender<T>,
    port: u16,
    addr: String,
    websocket: Arc<Mutex<Option<WebSocket>>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> NetworkSubscriber<T> {
    /// Create a new WebSocket subscriber
    ///
    /// Returns the subscriber and a receiver that Bevy can poll for data.
    /// The receiver is the same as the TCP version - Bevy code doesn't change!
    pub fn new(addr: String, port: u16) -> (Self, Receiver<T>) {
        let (sender, receiver) = crossbeam_channel::unbounded::<T>();

        (
            Self {
                sender,
                port,
                addr,
                websocket: Arc::new(Mutex::new(None)),
                _phantom: std::marker::PhantomData,
            },
            receiver,
        )
    }

    /// Send data back through the WebSocket to the proxy/server
    ///
    /// This enables bidirectional communication: browser → proxy → simulation
    /// Serializes data using bincode with [4-byte length][payload] format
    pub fn send<S: Serialize>(&self, data: &S) -> Result<(), String> {
        // Serialize the data using bincode
        let payload = bincode::serde::encode_to_vec(data, bincode::config::standard())
            .map_err(|e| format!("Failed to serialize data: {}", e))?;

        // Prepare message with length prefix: [4-byte length][bincode payload]
        let msg_len = payload.len() as u32;
        let mut message = Vec::with_capacity(4 + payload.len());
        message.extend_from_slice(&msg_len.to_le_bytes());
        message.extend_from_slice(&payload);

        // Send through WebSocket
        let ws_guard = self.websocket.lock().unwrap();
        if let Some(ws) = ws_guard.as_ref() {
            ws.send_with_u8_array(&message)
                .map_err(|e| format!("Failed to send WebSocket message: {:?}", e))?;
            Ok(())
        } else {
            Err("WebSocket not connected".to_string())
        }
    }
}

impl<T: 'static + DeserializeOwned + Send + Sync> NetworkSubscriber<T> {
    /// Start listening for WebSocket data
    ///
    /// Note: Unlike the TCP version, this is NOT async because
    /// WASM uses the browser's event loop instead of Tokio.
    pub fn start(&mut self) {
        // Set panic hook for better error messages in browser console
        console_error_panic_hook::set_once();

        // WebSocket URL - the proxy serves on different ports
        // Port mapping: TCP 8080 → WS 18080, TCP 8081 → WS 18081, etc.
        // Changed from +1000 to +10000 to avoid conflicts with other software
        let ws_port = self.port + 10000;  // Offset to avoid collision
        let ws_url = format!("ws://{}:{}", self.addr, ws_port);

        web_sys::console::log_1(&format!("Connecting to WebSocket: {}", ws_url).into());

        // Create WebSocket connection
        let ws = match WebSocket::new(&ws_url) {
            Ok(ws) => ws,
            Err(e) => {
                web_sys::console::error_1(
                    &format!("Failed to create WebSocket: {:?}", e).into(),
                );
                return;
            }
        };

        // Set binary type to arraybuffer (for bincode data)
        ws.set_binary_type(BinaryType::Arraybuffer);

        // Store WebSocket reference so we can send data later
        *self.websocket.lock().unwrap() = Some(ws.clone());

        let sender = self.sender.clone();

        // === onmessage callback: Receives data from WebSocket ===
        let onmessage_callback = Closure::<dyn FnMut(_)>::new(move |e: MessageEvent| {
            // Get the binary data
            if let Ok(arraybuf) = e.data().dyn_into::<js_sys::ArrayBuffer>() {
                let uint8_array = js_sys::Uint8Array::new(&arraybuf);
                let buf = uint8_array.to_vec();

                // The proxy forwards frames as-is: [4-byte length][bincode payload]
                // This is the SAME format as the TCP version!
                if buf.len() < 4 {
                    web_sys::console::error_1(&"Message too short (need 4-byte length prefix)".into());
                    return;
                }

                // Read length prefix (little-endian u32)
                let msg_len = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;

                if buf.len() < 4 + msg_len {
                    web_sys::console::error_1(
                        &format!("Incomplete message: expected {} bytes, got {}", msg_len + 4, buf.len()).into()
                    );
                    return;
                }

                // Extract the bincode payload
                let payload = &buf[4..4 + msg_len];

                // Deserialize using bincode (same as TCP version)
                match bincode::serde::decode_from_slice::<T, _>(
                    payload,
                    bincode::config::standard(),
                ) {
                    Ok((data, _)) => {
                        // Send to crossbeam channel → Bevy will poll this!
                        if let Err(e) = sender.send(data) {
                            web_sys::console::error_1(
                                &format!("Failed to send data to channel: {}", e).into(),
                            );
                        }
                    }
                    Err(e) => {
                        web_sys::console::error_1(
                            &format!("Failed to deserialize data: {}", e).into(),
                        );
                    }
                }
            } else {
                web_sys::console::warn_1(&"Received non-binary WebSocket message".into());
            }
        });

        ws.set_onmessage(Some(onmessage_callback.as_ref().unchecked_ref()));
        onmessage_callback.forget(); // Keep callback alive

        // === onerror callback ===
        let onerror_callback = Closure::<dyn FnMut(_)>::new(move |e: ErrorEvent| {
            web_sys::console::error_1(&format!("WebSocket error: {:?}", e.message()).into());
        });
        ws.set_onerror(Some(onerror_callback.as_ref().unchecked_ref()));
        onerror_callback.forget();

        // === onopen callback ===
        let onopen_callback = Closure::<dyn FnMut(JsValue)>::new(move |_event: JsValue| {
            web_sys::console::log_1(&"WebSocket connected successfully".into());
        });
        ws.set_onopen(Some(onopen_callback.as_ref().unchecked_ref()));
        onopen_callback.forget();

        // === onclose callback ===
        let onclose_callback = Closure::<dyn FnMut(JsValue)>::new(move |_event: JsValue| {
            web_sys::console::warn_1(&"WebSocket connection closed".into());
        });
        ws.set_onclose(Some(onclose_callback.as_ref().unchecked_ref()));
        onclose_callback.forget();

        // Note: The WebSocket is kept alive by the forget() calls above.
        // In production, you'd want to store the WebSocket and manage reconnection logic.
    }
}
