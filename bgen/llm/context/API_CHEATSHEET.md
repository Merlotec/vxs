## VoxelSim Python API Cheatsheet (Current)

## CRITICAL CONSTRAINTS - READ FIRST

### Integer Coordinates Only
ALL coordinate parameters MUST be integers (int), NOT floats:

```python
# CORRECT:
target = [60, 60, -20]
helpers.plan_to(world, origin, target, urgency=0.9, yaw=0.0, padding=1)

# WRONG - Will crash with TypeError:
target = [60.5, 60.3, -20.0]  # float coordinates!
helpers.plan_to(world, origin, target, ...)  # ERROR!
```

If you calculate coordinates with math.cos/sin, convert to int:
```python
import math
angle = math.radians(30)
x = 60 + 5 * math.cos(angle)  # This is a float!
y = 60 + 5 * math.sin(angle)  # This is a float!
target = [int(x), int(y), -20]  # Must convert to int!
```

### A* Planner Generates STRAIGHT LINE Paths Only
- `helpers.plan_to()` uses A* which creates direct straight-line paths
- For circular or curved paths, use MULTIPLE waypoints visited sequentially
- Example: 8 waypoints around a circle = approximate circular patrol

### Allowed Exceptions in Sandbox
The sandbox allows these exceptions in try/except blocks:
- `Exception` (general)
- `TypeError`
- `ValueError`
- `KeyError`

### Distance-Based Checks for Position
Never use exact equality for position checks:

```python
# WRONG - Will almost never match:
if agent.get_coord_py() == target:

# CORRECT - Check distance:
import math
origin = agent.get_coord_py()
dist = math.sqrt((origin[0]-target[0])**2 + (origin[1]-target[1])**2 + (origin[2]-target[2])**2)
if dist < 2:  # Within 2 voxels
    # Reached target
```

### Stage/Phase Progression
Never use time-based modulo conditions:

```python
# WRONG - Will never match (t is continuous float):
if t % 360 == 0:

# CORRECT - Use distance or counter:
if waypoint_counter >= num_waypoints:
if distance_to_target < 2:
```

### Urgency Parameter
Urgency is a speed multiplier: `actual_speed = base_speed * urgency` where base_speed = 20 m/s.

**Speed reference:**
- `urgency=1.0` → 20 m/s
- `urgency=0.9` → 18 m/s (recommended default)
- `urgency=0.8` → 16 m/s
- `urgency=0.5` → 10 m/s (slow, risk of timeout in long missions)
- `urgency=0.1` → 2 m/s (extremely slow, will timeout)

Urgency also affects lookahead distance and acceleration proportionally.

**Typical usage:**
```python
# Most situations
helpers.plan_to(world, origin, target, urgency=0.9, yaw=0.0, padding=1)

# Obstacle-dense areas: increase padding, not reduce urgency
helpers.plan_to(world, origin, target, urgency=0.9, yaw=0.0, padding=3)
```

Low urgency (<0.6) makes the drone very slow and often causes episode timeouts before reaching objectives.

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
  - `urgency`: speed multiplier where `v_max = 20.0 * urgency` m/s (0.9 = 18m/s recommended, <0.6 = too slow)
- `planner = vxs.AStarActionPlanner(padding)`
- `planner.plan_action_py(vg, [ox,oy,oz], [dx,dy,dz], urgency, yaw) -> vxs.ActionIntent` (urgency before yaw)
- `vxs.MoveDir` enum: `Forward/Back/Left/Right/Up/Down/Undecided`
  - IMPORTANT: To represent no movement, use `vxs.MoveDir.none()` (classmethod) or empty list `[]`
  - DO NOT use `vxs.MoveDir.None` - this causes a Python syntax error because None is a reserved keyword

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

Policy Runner Return Convention

- `act(...)` may return either an `ActionIntent`, or `(ActionIntent, cmd)` where `cmd` is one of:
  - `"Replace"`: overwrite current action with the new intent (default behavior)
  - `"Push"`: append the intent behind the current action using `agent.push_back_intent_py(...)`
