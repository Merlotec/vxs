from pynput import keyboard, mouse
import voxelsim, time

world = voxelsim.VoxelGrid()
world.generate_default_terrain(100)
dynamics = voxelsim.AgentDynamics.default_drone()
agent = voxelsim.Agent(0)
fw = voxelsim.FilterWorld()
env = voxelsim.GlobalEnv(world, {0: agent})
proj = voxelsim.CameraProjection.default_py()


# Renderer
renderer = voxelsim.AgentVisionRenderer(world, [400, 300])

# Client

client = voxelsim.RendererClient("127.0.0.1", 8080, 8081, 8090, 9090)
# Specify the number of agent renderers we want to connect to.
client.connect_py(1)
print("Controls: WASD=move, Space=up, Shift=down, ESC=quit")

env.send_world(client)

env.set_agent_pos(0, [50.0, 20.0, 50.0])

pressed = set()

def on_press(key):
    try:
        pressed.add(key.char.lower())
    except AttributeError:
        if key == keyboard.Key.space:
            pressed.add('space')
        elif key == keyboard.Key.shift:
            pressed.add('shift')
        elif key == keyboard.Key.esc:
            # stop listener
            return False

def on_release(key):
    try:
        pressed.discard(key.char.lower())
    except AttributeError:
        if key == keyboard.Key.space:
            pressed.discard('space')
        elif key == keyboard.Key.shift:
            pressed.discard('shift')

from threading import Thread
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

delta = 0.01

last_view_time = time.time()

FRAME_DELTA_MAX = 0.13

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
            fw.send_pov_py(client, 0, 0, proj)
            renderer.update_filter_world_py(env.get_agent(0).camera_view_py(), proj, fw, t0)
            last_view_time = t0

    commands = []
    if 'w' in pressed: commands.append(voxelsim.MoveCommand.forward(0.8))
    if 's' in pressed: commands.append(voxelsim.MoveCommand.back(0.8))
    if 'a' in pressed: commands.append(voxelsim.MoveCommand.left(0.8))
    if 'd' in pressed: commands.append(voxelsim.MoveCommand.right(0.8))
    if 'space' in pressed: commands.append(voxelsim.MoveCommand.up(0.8))
    if 'shift' in pressed: commands.append(voxelsim.MoveCommand.down(0.5))

    if commands:
        env.perform_sequence_on_agent(0, commands)

    collisions = env.update_py(dynamics, delta)
    # if len(collisions) > 0:
    #     print("Collision!")
    # im = env.update_pov_py()
    
    env.send_agents(client)
    # env.send_pov(client, 0, 0)
    d = time.time() - t0
    if d < delta:
        time.sleep(delta - d)

listener.join()
