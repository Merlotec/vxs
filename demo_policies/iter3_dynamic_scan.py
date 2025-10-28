"""
Iteration 3: Ground scan first, then detailed turbine scan
"""
import math
from typing import Any, Dict, Optional, Tuple, List
import voxelsim as vxs

# Hardcoded turbine locations (we know where they are)
TURBINE_CENTERS = [(30, 30), (120, 120), (210, 210)]

# Ground scan parameters - full coverage
GROUND_HEIGHT = -25  # Higher to avoid planning failures
GROUND_GRID = []
# Scan in strips for better coverage
for x in range(15, 225, 20):  # Every 20 voxels
    for z in range(15, 225, 20):
        GROUND_GRID.append([x, z, GROUND_HEIGHT])

# Scanning parameters - ultra detailed
SCAN_HEIGHTS = list(range(-35, -125, -5))  # Every 5 voxels from -35 to -120 = 18 heights
SCAN_RADIUS = 10
SCAN_POINTS_PER_HEIGHT = 16  # 16 points per circle (very dense)

# State machine
state = "GROUND_SCAN"  # GROUND_SCAN -> TURBINE_SCAN -> HOME
ground_waypoint_idx = 0
current_turbine_idx = 0
scan_waypoints = []
scan_waypoint_idx = 0


def generate_scan_waypoints(center: Tuple[int, int]) -> list:
    """Generate detailed spiral scan waypoints around turbine center"""
    cx, cz = center
    waypoints = []

    # Ground scan first (more points)
    for offset in [(-15, -15), (15, -15), (15, 15), (-15, 15), (-15, 0), (0, -15), (15, 0), (0, 15)]:
        waypoints.append([cx + offset[0], cz + offset[1], -15])

    # Vertical scan at multiple heights with three radii (very close + close + far)
    for height in SCAN_HEIGHTS:
        # Very close circle (almost touching turbine)
        for i in range(SCAN_POINTS_PER_HEIGHT):
            angle = 2 * math.pi * i / SCAN_POINTS_PER_HEIGHT
            x = int(cx + 4 * math.cos(angle))  # Radius 4 (very close)
            z = int(cz + 4 * math.sin(angle))
            waypoints.append([x, z, height])

        # Medium circle
        for i in range(SCAN_POINTS_PER_HEIGHT):
            angle = 2 * math.pi * i / SCAN_POINTS_PER_HEIGHT
            x = int(cx + 7 * math.cos(angle))  # Radius 7
            z = int(cz + 7 * math.sin(angle))
            waypoints.append([x, z, height])

        # Outer circle
        for i in range(SCAN_POINTS_PER_HEIGHT):
            angle = 2 * math.pi * i / SCAN_POINTS_PER_HEIGHT
            x = int(cx + SCAN_RADIUS * math.cos(angle))  # Radius 10 (farther)
            z = int(cz + SCAN_RADIUS * math.sin(angle))
            waypoints.append([x, z, height])

    return waypoints

def init(config: Dict[str, Any]) -> None:
    global state, ground_waypoint_idx, current_turbine_idx, scan_waypoints, scan_waypoint_idx
    state = "GROUND_SCAN"
    ground_waypoint_idx = 0
    current_turbine_idx = 0
    scan_waypoints = []
    scan_waypoint_idx = 0
    print(f"Ground-first scan initialized: {len(GROUND_GRID)} ground waypoints, {len(TURBINE_CENTERS)} turbines")

def act(t: float, agent: vxs.Agent, world: vxs.VoxelGrid, fw: vxs.FilterWorld, env: vxs.EnvState, helpers: Any) -> Optional[object]:
    global state, ground_waypoint_idx, current_turbine_idx, scan_waypoints, scan_waypoint_idx

    if agent.get_action_py() is None:
        origin = agent.get_coord_py()

        if state == "GROUND_SCAN":
            # Do ground scan first
            if ground_waypoint_idx >= len(GROUND_GRID):
                # Ground scan complete, move to turbine scanning
                print(f"Ground scan complete. Starting turbine scans.")
                state = "TURBINE_SCAN"
                current_turbine_idx = 0
                return None

            target = GROUND_GRID[ground_waypoint_idx]

            # Check if reached
            dist = math.sqrt(sum((origin[i] - target[i])**2 for i in range(3)))
            if dist < 6:
                ground_waypoint_idx += 1
                return None

            # Plan to waypoint with higher urgency to be more aggressive
            try:
                intent = helpers.plan_to(world, origin, target, urgency=0.3, yaw=0.0, padding=0)
                if intent:
                    return intent, "Replace"
            except Exception as e:
                # Don't skip, just print error
                print(f"Planning failed to {target}: {e}")

            # Only skip if reached or explicitly failed
            ground_waypoint_idx += 1

        elif state == "TURBINE_SCAN":
            if current_turbine_idx >= len(TURBINE_CENTERS):
                # All turbines scanned, go home
                state = "HOME"
                return None

            # Generate scan waypoints if needed
            if not scan_waypoints:
                turbine_center = TURBINE_CENTERS[current_turbine_idx]
                scan_waypoints = generate_scan_waypoints(turbine_center)
                scan_waypoint_idx = 0
                print(f"Scanning turbine {current_turbine_idx + 1}/{len(TURBINE_CENTERS)} at {turbine_center}: {len(scan_waypoints)} waypoints")

            if scan_waypoint_idx >= len(scan_waypoints):
                # Done with this turbine
                current_turbine_idx += 1
                scan_waypoints = []
                scan_waypoint_idx = 0
                return None

            target = scan_waypoints[scan_waypoint_idx]

            # Check if reached
            dist = math.sqrt(sum((origin[i] - target[i])**2 for i in range(3)))
            if dist < 6:
                scan_waypoint_idx += 1
                return None

            # Plan to waypoint
            try:
                intent = helpers.plan_to(world, origin, target, urgency=0.3, yaw=0.0, padding=0)
                if intent:
                    return intent, "Replace"
            except Exception:
                pass

            # Skip if planning fails
            scan_waypoint_idx += 1

        elif state == "HOME":
            home = [110, 110, -80]
            dist = math.sqrt(sum((origin[i] - home[i])**2 for i in range(3)))
            if dist < 10:
                return None

            try:
                intent = helpers.plan_to(world, origin, home, urgency=0.3, yaw=0.0, padding=0)
                if intent:
                    return intent, "Replace"
            except Exception:
                pass

    return None

def collect(step_ctx: Dict[str, Any]) -> Dict[str, str]:
    global state, ground_waypoint_idx, current_turbine_idx, scan_waypoint_idx, scan_waypoints

    if state == "GROUND_SCAN":
        status = f"Ground_Scan_{ground_waypoint_idx}/{len(GROUND_GRID)}"
        detail = "Searching"
    elif state == "TURBINE_SCAN":
        status = f"Turbine_{current_turbine_idx + 1}/{len(TURBINE_CENTERS)}"
        if scan_waypoints:
            detail = f"Scanning_{scan_waypoint_idx}/{len(scan_waypoints)}"
        else:
            detail = "Planning"
    elif state == "HOME":
        status = "Returning_Home"
        detail = "Done"
    else:
        status = "Unknown"
        detail = ""

    return {
        "state": state,
        "status": status,
        "detail": detail
    }

def finalize(ep_ctx: Dict[str, Any]) -> Dict[str, str]:
    global current_turbine_idx
    return {"summary": f"Ground scan + {current_turbine_idx}/{len(TURBINE_CENTERS)} turbines scanned. Steps: {ep_ctx.get('steps', 0)}"}
