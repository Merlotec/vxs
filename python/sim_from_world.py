import argparse
import json
import time
import math
from pathlib import Path

import voxelsim as vxs
from pynput import keyboard


def load_voxelgrid_from_world(path: Path) -> vxs.VoxelGrid:
    data = json.loads(Path(path).read_text())
    # Expect schema: { dims, meters_per_voxel, cells: [x,y,z,1], ... }
    cells = {}
    filled = vxs.Cell.filled()
    for item in data.get("cells", []):
        x, y, z = int(item[0]), int(item[1]), int(item[2])
        cells[(x, y, z)] = filled
    if hasattr(vxs.VoxelGrid, "from_dict_py"):
        return vxs.VoxelGrid.from_dict_py(cells)
    vg = vxs.VoxelGrid()
    for k, cell in cells.items():
        vg.set_cell_py(k, cell)
    return vg


def main():
    ap = argparse.ArgumentParser(description="Run simulator on saved world JSON")
    ap.add_argument("world_json", nargs="?", default=str(Path("runs/osm_voxel/world.vxs.json")), help="Path to world.vxs.json")
    args = ap.parse_args()

    world_path = Path(args.world_json)
    if not world_path.exists():
        print(f"File not found: {world_path}")
        return

    # Load world
    world = load_voxelgrid_from_world(world_path)

    # Agent and sim setup (based on povtest.py)
    agent = vxs.Agent(0)
    agent.set_hold_py([50, 50, -20], 0.0)

    fw = vxs.FilterWorld()
    dynamics = vxs.px4.Px4Dynamics.default_py()
    chaser = vxs.FixedLookaheadChaser.default_py()
    proj = vxs.CameraProjection.default_py()
    env = vxs.EnvState.default_py()
    camera_orientation = vxs.CameraOrientation.vertical_tilt_py(-0.5)

    noise = vxs.NoiseParams.default_with_seed_py([0.0, 0.0, 0.0])
    renderer = vxs.AgentVisionRenderer(world, [200, 150], noise)

    client = vxs.RendererClient.default_localhost_py()
    client.connect_py(1)
    client.send_world_py(world)
    client.send_agents_py({0: agent})

    pressed = set()
    just_pressed = set()

    def on_press(key):
        try:
            key_str = key.char.lower()
        except AttributeError:
            if key == keyboard.Key.space:
                key_str = 'space'
            elif key == keyboard.Key.shift:
                key_str = 'shift'
            elif key == keyboard.Key.esc:
                return False
            else:
                return
        if key_str not in pressed:
            just_pressed.add(key_str)
        pressed.add(key_str)

    def on_release(key):
        try:
            key_str = key.char.lower()
        except AttributeError:
            if key == keyboard.Key.space:
                key_str = 'space'
            elif key == keyboard.Key.shift:
                key_str = 'shift'
            else:
                return
        pressed.discard(key_str)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    delta = 0.01
    last_view_time = time.time()
    FRAME_DELTA_MAX = 0.13
    upd_start = 0.0

    def world_update(world, timestamp):
        dtime = time.time() - upd_start
        print(f"upd_time: {dtime}")

    YAW_STEP = math.pi * 0.125
    yaw_delta = 0.0

    print("Controls: WASD=move, Space=up, Shift=down, Q/E=rotate, ESC=quit")
    while listener.running:
        t0 = time.time()
        view_delta = t0 - last_view_time
        if fw.is_updating_py(last_view_time):
            if view_delta >= FRAME_DELTA_MAX:
                continue
        else:
            if view_delta >= FRAME_DELTA_MAX:
                fw.send_pov_py(client, 0, 0, proj, camera_orientation)
                upd_start = time.time()
                renderer.update_filter_world_py(agent.camera_view_py(camera_orientation), proj, fw, t0, world_update)
                last_view_time = t0

        action = agent.get_action_py()
        commands = []
        if action:
            commands = action.get_intent().get_move_sequence()
        commands_cl = list(commands)

        if 'q' in just_pressed:
            yaw_delta -= YAW_STEP
        if 'e' in just_pressed:
            yaw_delta += YAW_STEP
        if 'w' in just_pressed: commands.append(vxs.MoveDir.Forward)
        if 's' in just_pressed: commands.append(vxs.MoveDir.Back)
        if 'a' in just_pressed: commands.append(vxs.MoveDir.Left)
        if 'd' in just_pressed: commands.append(vxs.MoveDir.Right)
        if 'space' in just_pressed: commands.append(vxs.MoveDir.Up)
        if 'shift' in just_pressed: commands.append(vxs.MoveDir.Down)
        if not action or commands != commands_cl:
            if len(commands) > 0:
                intent = vxs.ActionIntent(0.8, yaw_delta, commands, None)
                agent.perform_py(intent)

        chase_target = chaser.step_chase_py(agent, delta)
        dynamics.update_agent_dynamics_py(agent, env, chase_target, delta)
        just_pressed.clear()
        collisions = world.collisions_py(agent.get_pos(), [0.5, 0.5, 0.3])
        if len(collisions) > 0:
            print (f"{len(collisions)}, collisions")
        client.send_agents_py({0: agent})
        d = time.time() - t0
        if d < delta:
            time.sleep(delta - d)

    listener.join()


if __name__ == "__main__":
    main()

