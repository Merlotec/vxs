import voxelsim

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
