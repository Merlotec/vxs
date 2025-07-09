from pynput import keyboard, mouse
import voxelsim, time

world = voxelsim.VoxelGrid()
world.generate_default_terrain(100)
dynamics = voxelsim.AgentDynamics.default_drone()
agent = voxelsim.Agent(0)
env = voxelsim.GlobalEnv(world, {0: agent})

# … your client setup …

client = voxelsim.RendererClient("127.0.0.1", 8080, 8081, 8090)
client.connect_py(1)
print("Controls: WASD=move, Space=up, Shift=down, ESC=quit")

env.send_world(client)

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
while listener.running:
    t0 = time.time()
    commands = []
    if 'w' in pressed: commands.append(voxelsim.MoveCommand.forward(0.8))
    if 's' in pressed: commands.append(voxelsim.MoveCommand.back(0.8))
    if 'a' in pressed: commands.append(voxelsim.MoveCommand.left(0.8))
    if 'd' in pressed: commands.append(voxelsim.MoveCommand.right(0.8))
    if 'space' in pressed: commands.append(voxelsim.MoveCommand.up(0.8))
    if 'shift' in pressed: commands.append(voxelsim.MoveCommand.down(0.5))

    if commands:
        env.perform_sequence_on_agent(0, commands)

    env.update_with_callback(dynamics, delta, lambda: None, lambda i: print("Collision!"))
    env.update_povs_py()
    env.send_agents(client)
    env.send_pov(client, 0, 0)
    d = time.time() - t0
    if d < delta:
        time.sleep(delta - d)

listener.join()
