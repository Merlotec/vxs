from pynput import keyboard, mouse
import voxelsim, time

# dynamics = voxelsim.AgentDynamics.default_drone()
agent = voxelsim.Agent(0)
agent.set_pos([50.0, 50.0, 20.0])

fw = voxelsim.FilterWorld()
dynamics = voxelsim.QuadDynamics.default_py()

chaser = voxelsim.FixedLookaheadChaser.default_py()

generator = voxelsim.TerrainGenerator()
generator.generate_terrain_py(voxelsim.TerrainConfig.default_py())
world = generator.generate_world_py()
proj = voxelsim.CameraProjection.default_py()
env = voxelsim.EnvState.default_py()

AGENT_CAMERA_TILT = -0.5
camera_orientation = voxelsim.CameraOrientation.vertical_tilt_py(-0.5)
# Renderer
noise = voxelsim.NoiseParams.default_with_seed_py([0.0, 0.0, 0.0])
renderer = voxelsim.AgentVisionRenderer(world, [200, 150], noise)

# Client

client = voxelsim.RendererClient.default_localhost_py()
# Specify the number of agent renderers we want to connect to.
client.connect_py(1)
print("Controls: WASD=move, Space=up, Shift=down, ESC=quit")

client.send_world_py(world)
client.send_agents_py({0: agent})

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
    if 'w' in just_pressed: commands.append(voxelsim.MoveCommand(voxelsim.MoveDir.Forward, 0.8, 0.0))
    if 's' in just_pressed: commands.append(voxelsim.MoveCommand(voxelsim.MoveDir.Back, 0.8, 0.0))
    if 'a' in just_pressed: commands.append(voxelsim.MoveCommand(voxelsim.MoveDir.Left, 0.8, 0.0))
    if 'd' in just_pressed: commands.append(voxelsim.MoveCommand(voxelsim.MoveDir.Right, 0.8, 0.0))
    if 'space' in just_pressed: commands.append(voxelsim.MoveCommand(voxelsim.MoveDir.Up, 0.8, 0.0))
    if 'shift' in just_pressed: commands.append(voxelsim.MoveCommand(voxelsim.MoveDir.Down, 0.8, 0.0))
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
