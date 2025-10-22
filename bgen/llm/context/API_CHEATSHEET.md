## VoxelSim Python API Cheatsheet (Current)

Import
- `import voxelsim as vxs`

World & Cells
- `vg = vxs.VoxelGrid.from_dict_py({(x,y,z): vxs.Cell.filled(), ...})`
- `vg.to_dict_py() -> Dict[Tuple[int,int,int], vxs.Cell]`
- `vg.as_numpy() -> (coords: np.ndarray[n,3], vals: np.ndarray[n])`
- `vg.collisions_py([x,y,z], [dx,dy,dz]) -> List[((i32,i32,i32), vxs.Cell)]`
- `vg.dense_snapshot_py([cx,cy,cz], [hx,hy,hz]) -> vxs.DenseSnapshot`
- `vxs.Cell.filled()`, `vxs.Cell.sparse()`; `cell.is_filled_py()`, `cell.is_sparse_py()`

Agents, Actions, Planning, View
- `agent = vxs.Agent(0)`
- `agent.set_hold_py([i32;i3], yaw)`; `agent.get_pos() -> [f64;3]`; `agent.get_coord_py() -> [i32;3]`
- `agent.camera_view_py(orientation) -> vxs.CameraView`
- `agent.perform_oneshot_py(vxs.ActionIntent(...))`; `agent.push_back_intent_py(...)`
- `action = agent.get_action_py()`; `action.get_intent_queue() -> List[vxs.ActionIntent]`
- `intent = vxs.ActionIntent(urgency, yaw, [vxs.MoveDir.Forward, ...], None)`
- `planner = vxs.AStarActionPlanner(padding)`
- `planner.plan_action_py(vg, [ox,oy,oz], [dx,dy,dz], yaw, urgency) -> vxs.ActionIntent`
- `vxs.MoveDir` enum: `Forward/Back/Left/Right/Up/Down/None/Undecided`

POV Compute
- `fw = vxs.FilterWorld()`
- `noise = vxs.NoiseParams.default_with_seed_py([f32;3])`
- `renderer = vxs.AgentVisionRenderer(world, [w,h], noise)`
- `proj = vxs.CameraProjection.default_py()`; `cam = vxs.CameraOrientation.vertical_tilt_py(angle)`
- Send POV updates at max speed but avoid overlap:
  - Gate on `fw.is_updating_py(last_timestamp)`; if False:
    - `fw.send_pov_py(client, stream_idx, agent_id, proj, cam)`
    - `renderer.update_filter_world_py(agent.camera_view_py(cam), proj, fw, now_ts, callback)`
    - `last_timestamp = now_ts`

Networking
- `client = vxs.RendererClient.default_localhost_py(pov_count)`
- `client.send_world_py(world)`; `client.send_agents_py({id: agent})`

Simulator
- `gen = vxs.TerrainGenerator()`; `cfg = vxs.TerrainConfig.default_py()`
- `gen.generate_terrain_py(cfg)`; `world = gen.generate_world_py()`
- `params = vxs.QuadParams.default_py()`; `dyn = vxs.QuadDynamics(params)`
- `env = vxs.EnvState.default_py()`
- `chaser = vxs.FixedLookaheadChaser.default_py()`; `chaser.step_chase_py(agent, dt)`

Notes
- Renderer is optional; the sim can run headless.
- `px4` dynamics is feature-gated; use `vxs.px4.Px4Dynamics.default_py()` only if built with `--features px4`.

Recommended for Behaviors
- Prefer path generation with `vxs.AStarActionPlanner(...).plan_action_py(...)` to build `ActionIntent`s.
- Use short move sequences and replan as needed; avoid long blocking loops inside `act()`.
