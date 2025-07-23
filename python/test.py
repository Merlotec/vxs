import voxelsim
import time

world = voxelsim.VoxelGrid()
world.generate_default_terrain(100)

dynamics = voxelsim.AgentDynamics.default_drone()
agent = voxelsim.Agent(0)
env = voxelsim.GlobalEnv(world, {0: agent})


client = voxelsim.RendererClient("127.0.0.1", 8080, 8081, 8090)
client.connect_py(0)

print("connected!")

env.send_world(client)
print("send client data")


def step():
    # Send the entire agent list through the pipe
    try:
        env.send_agents(client)
        print("agents sent")
    except BrokenPipeError:
        print("broken pipe")
        pass  # Render process may have closed

def collide(agent_id):
    pass

delta = 0.01
realtime = True
print("Control process started")
while True:
    t0 = time.time()
    
    env.perform_sequence_on_agent(0, [
        voxelsim.MoveCommand.up(0.8),
        voxelsim.MoveCommand.up(0.8),
        voxelsim.MoveCommand.right(0.8),
        voxelsim.MoveCommand.forward(0.8)
    ])
    env.update_with_callback(dynamics, delta, step, collide)
    env.send_agents(client)
     
    t1 = time.time()
    d = t1 - t0
    if realtime and d < delta:
        time.sleep(delta - (t1 - t0))
