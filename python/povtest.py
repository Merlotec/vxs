from pynput import keyboard, mouse
import voxelsim as vxs, time

# dynamics = vxs.AgentDynamics.default_drone()
agent = vxs.Agent(0)
agent.set_pos([50.0, 50.0, -20.0])

fw = vxs.FilterWorld()
dynamics = vxs.px4.Px4Dynamics.default_py()

chaser = vxs.FixedLookaheadChaser.default_py()

generator = vxs.TerrainGenerator()
generator.generate_terrain_py(vxs.TerrainConfig.default_py())
world = generator.generate_world_py()
proj = vxs.CameraProjection.default_py()
env = vxs.EnvState.default_py()

AGENT_CAMERA_TILT = -0.5
camera_orientation = vxs.CameraOrientation.vertical_tilt_py(AGENT_CAMERA_TILT)
# Renderer
noise = vxs.NoiseParams.default_with_seed_py([0.0, 0.0, 0.0])
renderer = vxs.AgentVisionRenderer(world, [200, 150], noise)

# Client

client = vxs.RendererClient.default_localhost_py()
# Specify the number of agent renderers we want to connect to.
client.connect_py(1)
print("Controls: WASD=move, Space=up, Shift=down, ESC=quit")

client.send_world_py(world)
client.send_agents_py({0: agent})


# Another client created for filter world output
client_fw = vxs.RendererClient("127.0.0.1", 8084, 8085, 8092, 9092)
client_fw.connect_py(0)

pressed = set()
just_pressed = set()

def on_press(key):
    # normalize all keys to a string
    try:
        key_str = key.char.lower()
    except AttributeError:
        if key == keyboard.Key.space:
            key_str = 'space'
        elif key == keyboard.Key.shift:
            key_str = 'shift'
        elif key == keyboard.Key.esc:
            # stop listener
            return False
        else:
            # ignore other non-char keys
            return

    # if this is the first time we've seen it down, mark as just_pressed
    if key_str not in pressed:
        just_pressed.add(key_str)
    # always record that it's down
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

from threading import Thread
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

delta = 0.01

last_view_time = time.time()

FRAME_DELTA_MAX = 0.13

upd_start = 0.0
def world_update(world, timestamp):
    dtime = time.time() - upd_start
    print(f"upd_time: {dtime}")

    coords, vals = fw.as_numpy() 

    # Sparse to dense

    # Encode

    # Decode

    # Show

YAW_STEP = 0.3  # radians per key press (about 17°)

while listener.running:
    t0 = time.time()
    view_delta = t0 - last_view_time
    # Rendering
    if fw.is_updating_py(last_view_time):
        if view_delta >= FRAME_DELTA_MAX:
            # Do not update the simulation because we want to await the view.
            continue;
    else:
        # Our next frame is ready, so wait for the frame delta time to hit then update the world
        # used for inference.
        # Here we just send the new world over to the renderer.
        if view_delta >= FRAME_DELTA_MAX:
            fw.send_pov_py(client, 0, 0, proj, camera_orientation)
            upd_start = time.time()
            renderer.update_filter_world_py(agent.camera_view_py(camera_orientation), proj, fw, t0, world_update)
            last_view_time = t0

    action = agent.get_action()
    commands = []
    if action:
        commands = action.get_commands()
    commands_cl = list(commands)

    # Compute yaw delta from input this frame; Q = left (negative), E = right (positive)
    yaw_delta = 0.0
    if 'q' in just_pressed:
        yaw_delta -= YAW_STEP
    if 'e' in just_pressed:
        yaw_delta += YAW_STEP

    # Apply yaw_delta to any move commands created this frame
    if 'w' in just_pressed: commands.append(vxs.MoveCommand(vxs.MoveDir.Forward, 0.8, yaw_delta))
    if 's' in just_pressed: commands.append(vxs.MoveCommand(vxs.MoveDir.Back, 0.8, yaw_delta))
    if 'a' in just_pressed: commands.append(vxs.MoveCommand(vxs.MoveDir.Left, 0.8, -3.14))
    if 'd' in just_pressed: commands.append(vxs.MoveCommand(vxs.MoveDir.Right, 0.8, 3.14))
    if 'space' in just_pressed: commands.append(vxs.MoveCommand(vxs.MoveDir.Up, 0.8, yaw_delta))
    if 'shift' in just_pressed: commands.append(vxs.MoveCommand(vxs.MoveDir.Down, 0.8, yaw_delta))
    if not action or commands != commands_cl:
        if len(commands) > 0:
            agent.perform_sequence_py(commands)

    # The point in space that the drone should be chasing.
    chase_target = chaser.step_chase_py(agent, delta)
    dynamics.update_agent_dynamics_py(agent, env, chase_target, delta)
    just_pressed.clear()
    collisions = world.collisions_py(agent.get_pos(), [0.5, 0.5, 0.3])
    if len(collisions) > 0:
        print (f"{len(collisions)}, collisions")
    # collisions = env.update_py(dynamics, delta)
    # if len(collisions) > 0:
    #     print("Collision!")
    # im = env.update_pov_py()
    
    client.send_agents_py({0: agent})
    # env.send_pov(client, 0, 0)
    d = time.time() - t0
    if d < delta:
        time.sleep(delta - d)

listener.join()
