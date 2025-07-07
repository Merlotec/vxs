use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use crossbeam_channel::{Receiver, Sender};
use nalgebra::Vector3;

use voxelsim::{Agent, Cell, VoxelGrid};

pub type Coord = [i32; 3];
use bevy::app::AppExit;
use bevy::prelude::*;

#[derive(Resource)]
pub struct WorldReceiver(Receiver<VoxelGrid>);

#[derive(Resource)]
pub struct AgentReceiver(Receiver<Vec<Agent>>);

pub fn begin_render(
    world_receiver: Receiver<VoxelGrid>,
    agent_receiver: Receiver<Vec<Agent>>,
    gui_sender: Sender<GuiCommand>,
    quit_receiver: Receiver<()>,
) {
    App::new()
        .add_plugins((
            DefaultPlugins.set(ImagePlugin::default_nearest()),
            PanOrbitCameraPlugin,
        ))
        .add_systems(Startup, setup)
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

#[derive(Component)]
struct CellComponent {
    coord: Coord,
    value: Cell,
}

#[derive(Component)]
struct AgentComponent {
    agent: Agent,
}

#[derive(Component)]
struct ActionCell {
    coord: Coord,
}

#[derive(Component)]
struct OriginCell {
    coord: Coord,
}

pub enum GuiCommand {
    RegenerateWorld,
    MoveAgentA,
    MoveAgentB,
    MoveAgentC,
    MoveAgentD,
}

#[derive(Resource)]
pub struct QuitReceiver(pub Receiver<()>);

#[derive(Resource)]
struct GuiChannel(Sender<GuiCommand>);

#[derive(Resource)]
struct CellAssets {
    cube_mesh: Handle<Mesh>,
    drone_mesh: Handle<Mesh>,
    action_mesh: Handle<Mesh>,
    drone_body_mat: Handle<StandardMaterial>,
    sparse_mat: Handle<StandardMaterial>,
    ground_mat: Handle<StandardMaterial>,
    solid_mat: Handle<StandardMaterial>,
    target_mat: Handle<StandardMaterial>,
    drone_mat: Handle<StandardMaterial>,
    action_mat: Handle<StandardMaterial>,
    action_origin_mat: Handle<StandardMaterial>,
}

impl CellAssets {
    fn material_for_cell(&self, cell: &Cell) -> Handle<StandardMaterial> {
        if cell.contains(Cell::TARGET) {
            return self.target_mat.clone();
        }
        if cell.contains(Cell::FILLED) {
            if cell.contains(Cell::GROUND) {
                self.ground_mat.clone()
            } else {
                self.solid_mat.clone()
            }
        } else if cell.contains(Cell::SPARSE) {
            self.sparse_mat.clone()
        } else {
            self.drone_mat.clone()
        }
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let drone_body_mat = materials.add(Color::srgb(1.0, 0.5, 0.5));
    let sparse_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.3, 0.8, 0.1, 0.8),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    let ground_mat = materials.add(Color::srgb(0.3, 0.3, 0.3));
    let solid_mat = materials.add(Color::srgb(0.4, 0.3, 0.3));
    let target_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.8, 0.1, 0.3, 0.3),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    let drone_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.3, 0.1, 0.8, 0.3),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });

    let action_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.7, 0.7, 0.9, 0.2),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });

    let action_origin_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.9, 0.9, 0.9, 0.1),
        alpha_mode: AlphaMode::Blend,
        ..default()
    });

    let cube_mesh = meshes.add(Cuboid::default());
    let drone_mesh = meshes.add(Torus::default());
    commands.insert_resource(CellAssets {
        cube_mesh: cube_mesh.clone(),
        drone_mesh,
        action_mesh: cube_mesh.clone(),
        drone_body_mat,
        sparse_mat,
        ground_mat,
        solid_mat,
        drone_mat,
        target_mat,
        action_mat,
        action_origin_mat,
    });

    commands.spawn((
        DirectionalLight {
            illuminance: 100.0,
            ..Default::default()
        },
        Transform::from_xyz(8.0, 16.0, 8.0),
    ));
    commands.insert_resource(AmbientLight {
        brightness: 1200.0,
        affects_lightmapped_meshes: true,
        ..Default::default()
    });
    commands.spawn((
        PanOrbitCamera::default(),
        Transform::from_xyz(0.0, 7., 14.0).looking_at(Vec3::new(0., 0., 0.), Vec3::Y),
    ));
}

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

#[derive(Resource)]
struct FocusedAgent(usize);

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
        println!("RERENDERING WORLD!!");
        for (entity, mut cell, mut material) in cell_query.iter_mut() {
            if let Some(v) = world.cells().get(&cell.coord).copied() {
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
        for (coord, cell) in world.cells().iter() {
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
