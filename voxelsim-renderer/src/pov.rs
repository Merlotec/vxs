use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use crossbeam_channel::{Receiver, Sender};
use nalgebra::Vector3;
use voxelsim::viewport::VirtualGrid;

use crate::network::NetworkSubscriber;
use crate::render::{
    self, ActionCell, AgentComponent, AgentReceiver, CellAssets, CellComponent, FocusedAgent,
    OriginCell, PovReceiver, QuitReceiver, VirtualCellComponent, WorldReceiver,
};
use voxelsim::{Agent, PovData, VoxelGrid};

use bevy::app::AppExit;
use bevy::prelude::*;

#[derive(Resource)]
pub struct CameraMode {
    pub is_pov: bool,
}

#[derive(Component)]
pub struct PovCamera;

pub fn run_pov_server(port_offset: u16) {
    let (mut pov_sub, pov_receiver) = NetworkSubscriber::<PovData>::new(
        std::env::var("VXS_POV_ADDR").unwrap_or("172.0.0.1".to_string()),
        std::env::var("VXS_POV_PORT")
            .ok()
            .and_then(|x| x.parse::<u16>().ok())
            .unwrap_or(8090)
            + port_offset,
    );

    let (mut agent_sub, agent_receiver) = NetworkSubscriber::<Vec<Agent>>::new(
        std::env::var("VXS_AGENT_ADDR").unwrap_or("172.0.0.1".to_string()),
        std::env::var("VXS_AGENT_PORT")
            .ok()
            .and_then(|x| x.parse::<u16>().ok())
            .unwrap_or(8081),
    );

    // Start network listener in background thread with its own tokio runtime
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            pov_sub.start().await;
            agent_sub.start().await;

            // Keep the async context alive
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            }
        });
    });

    let (gui_sender, gui_receiver) = crossbeam_channel::unbounded::<GuiCommand>();
    let (quit_sender, quit_receiver) = crossbeam_channel::unbounded::<()>();

    begin_render(pov_receiver, agent_receiver, gui_sender, quit_receiver);
}

pub enum GuiCommand {
    RegenerateWorld,
    MoveAgentA,
    MoveAgentB,
    MoveAgentC,
    MoveAgentD,
}

pub fn begin_render(
    pov_receiver: Receiver<PovData>,
    agent_receiver: Receiver<Vec<Agent>>,
    gui_sender: Sender<GuiCommand>,
    quit_receiver: Receiver<()>,
) {
    App::new()
        .add_plugins((
            DefaultPlugins.set(ImagePlugin::default_nearest()),
            PanOrbitCameraPlugin,
        ))
        .add_systems(Startup, render::setup)
        .add_systems(
            Update,
            (
                synchronise_world,
                input_system,
                centre_camera_system,
                quit_system,
            ),
        )
        .insert_resource(PovReceiver(pov_receiver))
        .insert_resource(AgentReceiver(agent_receiver))
        .insert_resource(GuiChannel(gui_sender))
        .insert_resource(FocusedAgent(0))
        .insert_resource(CameraMode { is_pov: false })
        .insert_resource(QuitReceiver(quit_receiver))
        .run();
}

#[derive(Resource)]
struct GuiChannel(Sender<GuiCommand>);

fn input_system(gui_channel: Res<GuiChannel>, keyboard: Res<ButtonInput<KeyCode>>) {
    for k in keyboard.get_just_pressed() {
        // we assign to `_` to drop the Result<(), SendError> so it's not unused
        let _ = match k {
            KeyCode::KeyA => gui_channel.0.send(GuiCommand::MoveAgentA),
            KeyCode::KeyB => gui_channel.0.send(GuiCommand::MoveAgentB),
            KeyCode::KeyC => gui_channel.0.send(GuiCommand::MoveAgentC),
            KeyCode::KeyD => gui_channel.0.send(GuiCommand::MoveAgentD),
            KeyCode::KeyR => gui_channel.0.send(GuiCommand::RegenerateWorld),

            _ => Ok(()),
        };
    }
}

fn quit_system(quit_rx: Res<QuitReceiver>, mut exit_ev: ResMut<Events<AppExit>>) {
    // try_recv() returns Ok(()) if someone sent a quit message
    if quit_rx.0.try_recv().is_ok() {
        exit_ev.send(AppExit::Success);
    }
}

fn centre_camera_system(
    agent_query: Query<(&Transform, &AgentComponent), Without<Camera3d>>,
    mut pan_orbit_camera_query: Query<
        (&mut PanOrbitCamera, &mut Transform, &mut Camera),
        (With<Camera3d>, Without<PovCamera>, Without<AgentComponent>),
    >,
    mut pov_camera_query: Query<
        (&mut Transform, &mut Camera),
        (With<Camera3d>, With<PovCamera>, Without<AgentComponent>),
    >,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut focused: ResMut<FocusedAgent>,
    mut camera_mode: ResMut<CameraMode>,
) {
    // Toggle camera mode with Tab key
    if keyboard_input.just_pressed(KeyCode::Tab) {
        camera_mode.is_pov = !camera_mode.is_pov;
        println!(
            "Camera mode: {}",
            if camera_mode.is_pov {
                "POV"
            } else {
                "Pan Orbit"
            }
        );

        // Enable/disable appropriate cameras
        for (_, _, mut camera) in pan_orbit_camera_query.iter_mut() {
            camera.is_active = !camera_mode.is_pov;
        }
        for (_, mut camera) in pov_camera_query.iter_mut() {
            camera.is_active = camera_mode.is_pov;
        }
    }

    // Handle agent switching with Z key
    if keyboard_input.just_pressed(KeyCode::KeyZ) {
        let mut positions = Vec::new();
        for (t, a) in agent_query.iter() {
            positions.push((a.agent.id, t.translation, a.agent.clone()));
        }
        positions.sort_by_key(|(k, _, _)| *k);

        let next_i = if let Some(i) = positions.iter().position(|(k, _, _)| *k == focused.0) {
            if (i + 1) < positions.len() {
                i + 1
            } else {
                0
            }
        } else if !positions.is_empty() {
            0
        } else {
            return;
        };

        let (agent_id, pos, agent) = &positions[next_i];
        focused.0 = *agent_id;

        if camera_mode.is_pov {
            // POV mode: position camera at agent center with velocity-based rotation and thrust tilt
            if let Ok((mut pov_transform, _)) = pov_camera_query.single_mut() {
                update_pov_camera(&mut pov_transform, pos, agent);
            }
        } else {
            // Pan orbit mode: set target focus
            for (mut pan_orbit, _, _) in pan_orbit_camera_query.iter_mut() {
                pan_orbit.target_focus = *pos;
            }
        }
    }

    // Continuous updates for POV mode
    if camera_mode.is_pov {
        // Find the focused agent
        for (t, a) in agent_query.iter() {
            if a.agent.id == focused.0 {
                if let Ok((mut pov_transform, _)) = pov_camera_query.single_mut() {
                    update_pov_camera(&mut pov_transform, &t.translation, &a.agent);
                }
                break;
            }
        }
    }
}

fn update_pov_camera(camera_transform: &mut Transform, agent_pos: &Vec3, agent: &Agent) {
    // Position camera at agent center
    camera_transform.translation = *agent_pos;

    // Calculate velocity-based horizontal rotation
    let velocity = Vec3::new(agent.vel.x, 0.0, agent.vel.z); // Only horizontal component
    let horizontal_direction = if velocity.length() > 0.001 {
        velocity.normalize()
    } else {
        Vec3::NEG_Z // Default forward direction
    };

    // Calculate thrust magnitude for tilt
    let thrust_magnitude = Vec3::new(agent.thrust.x, agent.thrust.y, agent.thrust.z).length();

    // Tilt factor: higher thrust = more downward tilt
    // Scale thrust to reasonable tilt range (0 to 45 degrees)
    let max_tilt_degrees: f32 = 45.0;
    let tilt_radians = (thrust_magnitude * 0.1).min(1.0) * max_tilt_degrees.to_radians();

    // Create rotation: face horizontal velocity direction, then tilt down based on thrust
    let yaw = (-horizontal_direction.x).atan2(-horizontal_direction.z);
    let pitch = -tilt_radians; // Negative for downward tilt

    camera_transform.rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);
}

#[allow(clippy::too_many_arguments)]
fn synchronise_world(
    mut commands: Commands,
    pov: Res<PovReceiver>,
    agents: Res<AgentReceiver>,
    assets: Res<CellAssets>,
    mut cell_query: Query<(
        Entity,
        &mut VirtualCellComponent,
        &mut MeshMaterial3d<StandardMaterial>,
    )>,
    action_cell_query: Query<(Entity, &ActionCell)>,
    action_origin_cell_query: Query<(Entity, &OriginCell)>,
    mut agent_query: Query<(Entity, &mut AgentComponent, &mut Transform)>,
) {
    if let Some(mut pov) = pov.0.try_iter().last() {
        for (entity, mut cell, mut material) in cell_query.iter_mut() {
            if let Some(v) = pov.virtual_world.cells().get(&cell.coord).copied() {
                cell.value = v;
                // Change visual type.
                **material = assets.material_for_cell(&v.cell);
                pov.virtual_world.remove(&cell.coord);
            } else {
                commands.entity(entity).despawn();
            }
        }

        // Add remaining cells.render.rs

        // Add remaining cells.
        for (coord, cell) in pov.virtual_world.cells().iter() {
            commands.spawn((
                VirtualCellComponent {
                    // <-- NEW
                    coord: *coord,
                    value: *cell,
                },
                Mesh3d(assets.cube_mesh.clone()),
                MeshMaterial3d(assets.material_for_cell(&cell.cell)),
                Transform::from_xyz(coord[0] as f32, coord[1] as f32, coord[2] as f32),
            ));
        }
    }

    if let Some(mut agents) = agents.0.try_iter().last() {
        // Update the action space of the drones.
        let mut action_cells: Vec<Vector3<i32>> = Vec::new();
        let mut origin_cells: Vec<Vector3<i32>> = Vec::new();
        for agent in agents.iter() {
            if let Some(action) = &agent.action {
                let mut buf = action.origin;
                origin_cells.push(buf);

                for cmd in action.cmd_sequence.iter() {
                    if let Some(dir_vec) = cmd.dir.dir_vec() {
                        buf += dir_vec;
                    }
                    action_cells.push(buf);
                }
            }
        }

        for (entity, cell) in action_cell_query.iter() {
            let mut contains = false;
            action_cells.retain(|x| {
                if x == &Vector3::from(cell.coord) {
                    contains = true;
                    false
                } else {
                    true
                }
            });
            if !contains {
                commands.entity(entity).despawn();
            }
        }

        for a in action_cells {
            commands.spawn((
                ActionCell { coord: a.into() },
                Mesh3d(assets.action_mesh.clone()),
                MeshMaterial3d(assets.action_mat.clone()),
                Transform::from_translation(Vec3::from_array(a.cast::<f32>().into())),
            ));
        }

        for (entity, cell) in action_origin_cell_query.iter() {
            let mut contains = false;
            origin_cells.retain(|x| {
                if x == &Vector3::from(cell.coord) {
                    contains = true;
                    false
                } else {
                    true
                }
            });
            if !contains {
                commands.entity(entity).despawn();
            }
        }

        for a in origin_cells {
            commands.spawn((
                OriginCell { coord: a.into() },
                Mesh3d(assets.action_mesh.clone()),
                MeshMaterial3d(assets.action_origin_mat.clone()),
                Transform::from_translation(Vec3::from_array(a.cast::<f32>().into())),
            ));
        }

        for (entity, mut agent, mut transform) in agent_query.iter_mut() {
            let mut keep = false;
            agents.retain(|a| {
                if agent.agent.id == a.id {
                    transform.translation = Vec3::from_array(a.pos.into());
                    agent.agent = a.clone();
                    keep = true;
                    false
                } else {
                    true
                }
            });

            if !keep {
                commands.entity(entity).despawn();
            }
        }

        for a in agents.iter() {
            commands.spawn((
                AgentComponent { agent: a.clone() },
                Mesh3d(assets.drone_mesh.clone()),
                MeshMaterial3d(assets.drone_body_mat.clone()),
                Transform::from_translation(Vec3::from_array(a.pos.into())),
            ));
        }
    }
}
