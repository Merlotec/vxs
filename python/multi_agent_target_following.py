from pynput import keyboard
import voxelsim as vxs
import time
import math

WORLD_SIZE = 100
NUM_AGENTS = 10
SAFE_RADIUS = 5
VIEW_ANGLE = (120 *  math.pi / 180)  
generator = vxs.TerrainGenerator()
config = vxs.TerrainConfig.default_py()
config.set_world_dimensions_py(WORLD_SIZE, 30 , WORLD_SIZE)  # (x, y/height, z)
generator.generate_terrain_py(config)
world = generator.generate_world_py()
 
target_pos = [50, 50, -20]


target_agent = vxs.Agent(999)
target_agent.set_hold_py(target_pos, 0.0)

agents = {}
dynamics_list = {}
chasers = {}


agents[999] = target_agent

for i in range(NUM_AGENTS):
    agent = vxs.Agent(i)

    x = 50 + (i % 3) - 1  
    y = 50 + (i // 3) - 1
    z = -20
    agent.set_hold_py([x, y, z], 0.0)
    agents[i] = agent
    dynamics_list[i] = vxs.px4.Px4Dynamics.default_py()
    chasers[i] = vxs.FixedLookaheadChaser.default_py()

planner = vxs.AStarActionPlanner(1)
env = vxs.EnvState.default_py()

client = vxs.AsyncRendererClient.default_localhost_py(1)
client.send_world_py(world)
client.send_agents_py(agents)


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
            
    if key_str in pressed:
        pressed.remove(key_str)
        
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

def calculate_avoidance(my_agent, my_id, all_agents):
    my_pos = my_agent.get_pos()
    avoid = [0.0, 0.0, 0.0]

    for other_id, other_agent in all_agents.items():
        if other_id == my_id or other_id == 999:  
            continue
        other_pos = other_agent.get_pos()
        dx = other_pos[0] - my_pos[0]
        dy = other_pos[1] - my_pos[1]
        dz = other_pos[2] - my_pos[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        if dist < SAFE_RADIUS and dist > 0.1:
            avoid[0] -= (SAFE_RADIUS - dist) * (dx / dist)
            avoid[1] -= (SAFE_RADIUS - dist) * (dy / dist)
            avoid[2] -= (SAFE_RADIUS - dist) * (dz / dist)
    return avoid
    
    
    
def apply_avoidance(target, avoid_vector):
    MAX_AVOID = 5.0
    mag = math.sqrt(sum(v*v for v in avoid_vector))
    if mag > MAX_AVOID:
        scale = MAX_AVOID / mag
        avoid_vector = [v * scale for v in avoid_vector]

    return [
        int(target[0] + avoid_vector[0]),
        int(target[1] + avoid_vector[1]),
        int(target[2] + avoid_vector[2])
    ]

delta = 0.01
print(f"Agents: {NUM_AGENTS}, Safe Radius: {SAFE_RADIUS}")
print("Controls: WASD=move target, Space=up, Shift=down, ESC=quit")

while listener.running:
    t0 = time.time()
    
    MOVE_SPEED = 2
    old_target = target_pos.copy()

    if 'w' in just_pressed: target_pos[0] += MOVE_SPEED
    if 's' in just_pressed: target_pos[0] -= MOVE_SPEED
    if 'a' in just_pressed: target_pos[1] -= MOVE_SPEED
    if 'd' in just_pressed: target_pos[1] += MOVE_SPEED
    if 'space' in just_pressed: target_pos[2] -= MOVE_SPEED
    if 'shift' in just_pressed: target_pos[2] += MOVE_SPEED


    cols = world.collisions_py(target_pos, [0.5, 0.5, 0.5])
    if len(cols) > 0:
        target_pos = old_target
        print(f"Target blocked! Staying at {target_pos}")


    target_agent.set_hold_py(target_pos, 0.0)

    for agent_id, agent in agents.items():

        if agent_id == 999:
            continue

        if agent.get_action_py() is None:
            origin = agent.get_coord_py()

            avoid_vec = calculate_avoidance(agent, agent_id, agents)
            adjusted_target = apply_avoidance(target_pos, avoid_vec)


            avoid_mag = math.sqrt(sum(v*v for v in avoid_vec))
            if avoid_mag > 1.0:
                print(f"Agent {agent_id}: avoid_mag={avoid_mag:.2f}, vec={[round(v,1) for v in avoid_vec]}")


            target_cols = world.collisions_py(adjusted_target, [0.5, 0.5, 0.5])
            if len(target_cols) > 0:

                adjusted_target[2] -= 3  

                if len(world.collisions_py(adjusted_target, [0.5, 0.5, 0.5])) > 0:
                    continue

            try:
                intent = planner.plan_action_py(
                    world,
                    origin,
                    adjusted_target,
                    0.8, # urgency
                    0.0 # yaw
                )
                agent.perform_oneshot_py(intent)
            except Exception as e:
                print(f"Agent {agent_id} planning FAILED: {e}")
                print(f"  origin={origin}, adjusted_target={adjusted_target}")
        chase_target = chasers[agent_id].step_chase_py(agent, delta)
        dynamics_list[agent_id].update_agent_dynamics_py(
            agent, env, chase_target, delta
        )
        
    just_pressed.clear()
    
    client.send_agents_py(agents)
    d = time.time() - t0
    if d < delta:
        time.sleep(delta - d)
        
listener.join()
