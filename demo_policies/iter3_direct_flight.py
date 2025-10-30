"""
Demo Policy 3: DIRECT FLIGHT SCANNING
Flies directly to waypoints without A* pathfinding (simple demo)
"""
from typing import Any, Dict, Optional
import voxelsim as vxs
import math

# Turbine base positions
TURBINE_BASES = [
    (30, 30),
    (120, 120),
    (210, 210),
]

# Scanning parameters
SCAN_RADIUS = 10
SCAN_POINTS = 8
SCAN_HEIGHT = -66

# Generate all waypoints
WAYPOINTS = []
for tx, tz in TURBINE_BASES:
    for i in range(SCAN_POINTS):
        angle = 2 * math.pi * i / SCAN_POINTS
        x = int(tx + SCAN_RADIUS * math.cos(angle))
        z = int(tz + SCAN_RADIUS * math.sin(angle))
        WAYPOINTS.append([x, z, SCAN_HEIGHT])

# Add home
WAYPOINTS.append([110, 110, -80])

current_idx = 0

def init(config: Dict[str, Any]) -> None:
    global current_idx
    current_idx = 0

def act(t: float, agent: vxs.Agent, world: vxs.VoxelGrid, fw: vxs.FilterWorld, env: vxs.EnvState, helpers: Any) -> Optional[object]:
    global current_idx

    if current_idx >= len(WAYPOINTS):
        return None

    if agent.get_action_py() is None:
        origin = agent.get_coord_py()
        target = WAYPOINTS[current_idx]

        # Check if reached
        dist = math.sqrt(sum((origin[i] - target[i])**2 for i in range(3)))
        if dist < 5:
            current_idx += 1
            return None

        # Calculate direction vector
        dx = target[0] - origin[0]
        dy = target[1] - origin[1]
        dz = target[2] - origin[2]
        dist_3d = math.sqrt(dx*dx + dy*dy + dz*dz)

        if dist_3d < 1:
            current_idx += 1
            return None

        # Normalize and create movement sequence
        moves = []

        # X direction
        if abs(dx) > 2:
            moves.append(vxs.MoveDir.Right if dx > 0 else vxs.MoveDir.Left)

        # Y direction
        if abs(dy) > 2:
            moves.append(vxs.MoveDir.Forward if dy > 0 else vxs.MoveDir.Back)

        # Z direction
        if abs(dz) > 2:
            moves.append(vxs.MoveDir.Up if dz > 0 else vxs.MoveDir.Down)

        if not moves:
            current_idx += 1
            return None

        return helpers.intent(0.8, 0.0, moves), "Replace"

    return None

def collect(step_ctx: Dict[str, Any]) -> Dict[str, str]:
    global current_idx
    if current_idx < len(WAYPOINTS) - 1:
        turbine_num = current_idx // SCAN_POINTS + 1
        scan_point = (current_idx % SCAN_POINTS) + 1
        status = f"T{turbine_num}_P{scan_point}"
    else:
        status = "Home"

    return {
        "status": status,
        "progress": f"{current_idx}/{len(WAYPOINTS)}",
    }

def finalize(ep_ctx: Dict[str, Any]) -> Dict[str, str]:
    global current_idx
    return {"summary": f"Waypoints: {current_idx}/{len(WAYPOINTS)}. Steps: {ep_ctx.get('steps', 0)}"}
