from typing import Any, Dict, Optional
import math
import voxelsim as vxs

# Mission: Patrol in a circular pattern
# Strategy: Pre-calculate waypoints in a circle, visit them sequentially

CENTER = [60, 60, -20]
RADIUS = 10
NUM_WAYPOINTS = 8

# Pre-calculate integer waypoints around the circle
WAYPOINTS = []
for i in range(NUM_WAYPOINTS):
    angle = 2 * math.pi * i / NUM_WAYPOINTS
    # CRITICAL: Convert float to int!
    x = int(CENTER[0] + RADIUS * math.cos(angle))
    y = int(CENTER[1] + RADIUS * math.sin(angle))
    z = CENTER[2]
    WAYPOINTS.append([x, y, z])

current_waypoint = 0

def init(config: Dict[str, Any]) -> None:
    global current_waypoint
    current_waypoint = 0

def act(t: float, agent: vxs.Agent, world: vxs.VoxelGrid,
        fw: vxs.FilterWorld, env: vxs.EnvState, helpers: Any) -> Optional[object]:
    global current_waypoint

    if agent.get_action_py() is None:
        origin = agent.get_coord_py()
        target = WAYPOINTS[current_waypoint]

        # Check if reached current waypoint using distance
        # IMPORTANT: Don't use == for position checks!
        dist = math.sqrt(
            (origin[0] - target[0])**2 +
            (origin[1] - target[1])**2 +
            (origin[2] - target[2])**2
        )

        if dist < 2:  # Within 2 voxels = reached
            # Move to next waypoint (loop back to start)
            current_waypoint = (current_waypoint + 1) % NUM_WAYPOINTS
            target = WAYPOINTS[current_waypoint]

        try:
            return helpers.plan_to(world, origin, target, urgency=0.9, yaw=0.0, padding=1), "Replace"
        except Exception:
            # Fallback if planning fails
            return helpers.intent(urgency=0.7, yaw=0.0, moves=[vxs.MoveDir.Forward]), "Replace"

    return None

def collect(step_ctx: Dict[str, Any]) -> Dict[str, str]:
    global current_waypoint
    pos = step_ctx.get("agent_pos")
    return {
        "pos": f"{pos}",
        "waypoint": str(current_waypoint),
        "note": "circular patrol"
    }

def finalize(ep_ctx: Dict[str, Any]) -> Dict[str, str]:
    global current_waypoint
    success = ep_ctx.get("success", False)
    steps = ep_ctx.get("steps", 0)
    return {
        "summary": f"success={success}, steps={steps}, completed_waypoints={current_waypoint}"
    }
