
import voxelsim, time

# dynamics = voxelsim.AgentDynamics.default_drone()
agent = voxelsim.Agent(0)
agent.set_pos([50.0, 50.0, 20.0])

fw = voxelsim.FilterWorld()
dynamics = voxelsim.PengQuadDynamics.default_py()

chaser = voxelsim.FixedLookaheadChaser.default_py()

generator = voxelsim.TerrainGenerator()
generator.generate_terrain_py(voxelsim.TerrainConfig.default_py())
world = generator.generate_world_py()
proj = voxelsim.CameraProjection.default_py()
env = voxelsim.EnvState.default_py()

AGENT_CAMERA_TILT = -0.5
camera_orientation = voxelsim.CameraOrientation.vertical_tilt_py(-0.5)
# Renderer
renderer = voxelsim.AgentVisionRenderer(world, [400, 300])

# Client



delta = 0.01

last_view_time = time.time()

FRAME_DELTA_MAX = 0.13

while True:
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
            renderer.update_filter_world_py(agent.camera_view_py(camera_orientation), proj, fw, t0)
            last_view_time = t0

    action = agent.get_action()
    # The point in space that the drone should be chasing.
    chase_target = chaser.step_chase_py(agent, delta)
    dynamics.update_agent_dynamics_py(agent, env, chase_target, delta)
    # collisions = env.update_py(dynamics, delta)
    # if len(collisions) > 0:
    #     print("Collision!")
    # im = env.update_pov_py()
    
    # env.send_pov(client, 0, 0)
    d = time.time() - t0
    if d < delta:
        time.sleep(delta - d)

