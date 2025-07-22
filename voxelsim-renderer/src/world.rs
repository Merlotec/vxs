use std::ops::Deref;

use bevy::platform::collections::HashMap;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use crossbeam_channel::{Receiver, Sender};
use nalgebra::Vector3;

use crate::network::NetworkSubscriber;
use crate::render::{
    self, ActionCell, AgentComponent, AgentReceiver, CellAssets, CellComponent, FocusedAgent,
    OriginCell, QuitReceiver, WorldReceiver,
};
use voxelsim::{Agent, Cell, VoxelGrid};

use bevy::app::AppExit;
use bevy::prelude::*;

pub fn run_world_server() {
    let (mut world_sub, world_receiver) = NetworkSubscriber::<VoxelGrid>::new(
        std::env::var("VXS_WORLD_PORT").unwrap_or("172.0.0.1".to_string()),
        std::env::var("VXS_WORLD_PORT")
            .ok()
            .and_then(|x| x.parse::<u16>().ok())
            .unwrap_or(8080),
    );

    let (mut agent_sub, agent_receiver) = NetworkSubscriber::<HashMap<usize, Agent>>::new(
        std::env::var("VXS_AGENT_PORT").unwrap_or("172.0.0.1".to_string()),
        std::env::var("VXS_AGENT_PORT")
            .ok()
            .and_then(|x| x.parse::<u16>().ok())
            .unwrap_or(8081),
    );

    // Start network listener in background thread with its own tokio runtime
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            world_sub.start().await;
            agent_sub.start().await;

            // Keep the async context alive
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            }
        });
    });

    let (gui_sender, gui_receiver) = crossbeam_channel::unbounded::<GuiCommand>();
    let (quit_sender, quit_receiver) = crossbeam_channel::unbounded::<()>();

    begin_render(world_receiver, agent_receiver, gui_sender, quit_receiver);
}

pub enum GuiCommand {
    RegenerateWorld,
    MoveAgentA,
    MoveAgentB,
    MoveAgentC,
    MoveAgentD,
}

pub fn begin_render(
    world_receiver: Receiver<VoxelGrid>,
    agent_receiver: Receiver<HashMap<usize, Agent>>,
    gui_sender: Sender<GuiCommand>,
    quit_receiver: Receiver<()>,
) {
    App::new()
        .add_plugins((
            DefaultPlugins
                .set(ImagePlugin::default_nearest())
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "World View".into(),
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
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
        .insert_resource(WorldReceiver(world_receiver))
        .insert_resource(AgentReceiver(agent_receiver))
        .insert_resource(GuiChannel(gui_sender))
        .insert_resource(FocusedAgent(0))
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
    mut camera_query: Query<&mut PanOrbitCamera, Without<AgentComponent>>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut focused: ResMut<FocusedAgent>,
) {
    if keyboard_input.just_pressed(KeyCode::KeyZ) {
        let mut positions = Vec::new();
        for (t, a) in agent_query.iter() {
            positions.push((a.agent.id, t.translation));
        }
        positions.sort_by_key(|(k, _)| *k);
        let next_i = if let Some(i) = positions.iter().position(|(k, _)| *k == focused.0) {
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

        let pos = positions[next_i].1;
        for mut t in camera_query.iter_mut() {
            t.target_focus = pos;
        }
        focused.0 = positions[next_i].0;
    }
}

#[allow(clippy::too_many_arguments)]
fn synchronise_world(
    mut commands: Commands,
    world: Res<WorldReceiver>,
    agents: Res<AgentReceiver>,
    assets: Res<CellAssets>,
    mut cell_query: Query<(
        Entity,
        &mut CellComponent,
        &mut MeshMaterial3d<StandardMaterial>,
    )>,
    action_cell_query: Query<(Entity, &ActionCell)>,
    action_origin_cell_query: Query<(Entity, &OriginCell)>,
    mut agent_query: Query<(Entity, &mut AgentComponent, &mut Transform)>,
) {
    if let Some(mut world) = world.0.try_iter().last() {
        for (entity, mut cell, mut material) in cell_query.iter_mut() {
            if let Some(v) = world.cells().get(&cell.coord).map(|x| *x) {
                cell.value = v;
                // Change visual type.
                **material = assets.material_for_cell(&v);
                world.remove(&cell.coord);
            } else {
                commands.entity(entity).despawn();
            }
        }

        // Add remaining cells.render.rs

        // Add remaining cells.
        for rm in world.cells().iter() {
            let (coord, cell) = rm.pair();
            commands.spawn((
                CellComponent {
                    // <-- NEW
                    coord: *coord,
                    value: *cell,
                },
                Mesh3d(assets.cube_mesh.clone()),
                MeshMaterial3d(assets.material_for_cell(cell)),
                Transform::from_xyz(coord[0] as f32, coord[1] as f32, coord[2] as f32),
            ));
        }
    }

    if let Some(mut agents) = agents.0.try_iter().last() {
        // Update the action space of the drones.
        let mut action_cells: Vec<Vector3<i32>> = Vec::new();
        let mut origin_cells: Vec<Vector3<i32>> = Vec::new();
        for (_id, agent) in agents.iter() {
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
            agents.retain(|_id, a| {
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

        for (_id, a) in agents.iter() {
            commands.spawn((
                AgentComponent { agent: a.clone() },
                Mesh3d(assets.drone_mesh.clone()),
                MeshMaterial3d(assets.drone_body_mat.clone()),
                Transform::from_translation(Vec3::from_array(a.pos.into())),
            ));
        }
    }
}
