import voxelsim
import time
import keyboard

world = voxelsim.VoxelGrid()
world.generate_default_terrain(100)

dynamics = voxelsim.AgentDynamics.default_drone()
agent = voxelsim.Agent(0)
env = voxelsim.GlobalEnv(world, [agent])

client = voxelsim.RendererClient("127.0.0.1", 8080, 8081)
client.connect_py()

print("connected!")

env.send_world(client)
print("send client data")

def step():
    pass

def collide(agent_id):
    print(f"Collision!")

delta = 0.01
realtime = True
print("Control process started")
print("Controls: WASD=move, Space=up, Shift=down, ESC=quit")

while True:
    t0 = time.time()
    
    # Build command list based on keys pressed
    commands = []
    if keyboard.is_pressed('w'):
        commands.append(voxelsim.MoveCommand.forward(0.8))
    if keyboard.is_pressed('s'):
        commands.append(voxelsim.MoveCommand.back(0.8))
    if keyboard.is_pressed('a'):
        commands.append(voxelsim.MoveCommand.left(0.8))
    if keyboard.is_pressed('d'):
        commands.append(voxelsim.MoveCommand.right(0.8))
    if keyboard.is_pressed('space'):
        commands.append(voxelsim.MoveCommand.up(0.8))
    if keyboard.is_pressed('shift'):
        commands.append(voxelsim.MoveCommand.down(0.5))
    
    if keyboard.is_pressed('esc'):
        break
    
    if commands:
        env.perform_sequence_on_agent(0, commands)
    
    env.update_with_callback(dynamics, delta, step, collide)
    env.send_agents(client)
     
    t1 = time.time()
    d = t1 - t0
    if realtime and d < delta:
        time.sleep(delta - (t1 - t0))