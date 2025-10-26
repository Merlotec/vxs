from __future__ import annotations
from pathlib import Path
from typing import Optional


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def build_system_prompt() -> str:
    return (
        "You are generating a Python policy module that controls a drone in VoxelSim. "
        "Output ONLY valid Python code implementing the required contract functions: "
        "init(config), act(t, agent, world, fw, env, helpers), collect(step_ctx), finalize(ep_ctx). "
        "Do not include explanations or backticks. Prefer A* path planning to move. "
        "In act(), return either an ActionIntent or (ActionIntent, 'Replace'|'Push'). "
        "Use non-zero urgency (e.g., 0.6–1.0); never use 0.0. "
        "Important: A* planner signature is plan_action_py(world, origin, dst, urgency, yaw) (urgency before yaw). "
        "Prefer helpers.plan_to(world, origin, dst, urgency=..., yaw=..., padding=...). "
        "\n\n"
        "CRITICAL CONSTRAINTS:\n"
        "1. ALL coordinates MUST be integers (int), NOT floats. If using math.cos/sin, wrap with int(): target = [int(x + r*cos(a)), int(y + r*sin(a)), z]\n"
        "2. A* planner generates STRAIGHT LINE paths only. For curves: use multiple waypoints, visit sequentially.\n"
        "3. Stage progression: use distance checks, NOT time. BAD: if t%360==0. GOOD: if distance(origin,target)<2\n"
        "4. Available exceptions: Exception, TypeError, ValueError, KeyError\n"
        "5. Position checks: use distance calculation, NOT equality. BAD: if origin==target. GOOD: if dist<2\n"
        "\n"
        "CIRCULAR PATROL PATTERN:\n"
        "Pre-calculate integer waypoints in a circle, visit sequentially:\n"
        "waypoints = []; for i in range(8): angle=2*pi*i/8; waypoints.append([int(cx+r*cos(angle)), int(cy+r*sin(angle)), z])\n"
        "Then: if distance(origin, waypoints[current])<2: current=(current+1)%len(waypoints)\n"
        "\n"
        "DEBUGGING SUPPORT:\n"
        "The collect(step_ctx) function is called every step and used for debugging. Return a dict with current state:\n"
        "- stage (int or str): Current mission stage\n"
        "- waypoint (int): Current waypoint index\n"
        "- target (str): Current target position as string (e.g., '[60,60,-20]')\n"
        "This trace will help identify bugs if the mission fails.\n"
        "CRITICAL: If you read module-level variables (current_stage, current_waypoint, etc), you MUST declare them as global:\n"
        "def collect(step_ctx):\n"
        "    global current_stage, current_waypoint  # ← REQUIRED for reading globals!\n"
        "    return {'stage': str(current_stage), 'waypoint': str(current_waypoint), 'target': str(WAYPOINTS[current_waypoint])}\n"
        "\n"
        "You are in an iterative loop: read the critique and improve if needed; if behavior is already good, keep your solution simple and stable."
    )


def build_user_prompt(
    *,
    user_goal: str,
    repo_root: Path,
    include_examples: bool = True,
    prior_critique_path: Optional[Path] = None,
    extra_code: Optional[str] = None,
) -> str:
    cheatsheet = _read(repo_root / "bgen/llm/context/API_CHEATSHEET.md")
    template = _read(repo_root / "bgen/llm/context/POLICY_TEMPLATE.py")
    examples = ""
    if include_examples:
        ex1 = _read(repo_root / "bgen/llm/context/examples/move_to_target.py")
        ex2 = _read(repo_root / "bgen/llm/context/examples/reactive_avoidance.py")
        ex3 = _read(repo_root / "bgen/llm/context/examples/circular_patrol.py")
        examples = f"\n# Example: move_to_target.py\n{ex1}\n\n# Example: reactive_avoidance.py\n{ex2}\n\n# Example: circular_patrol.py\n{ex3}"
    critique = _read(prior_critique_path) if prior_critique_path else ""
    extra = f"\nEXISTING CODE CONTEXT (for refinement):\n{extra_code}\n\n" if extra_code else ""
    return (
        f"USER GOAL:\n{user_goal}\n\n"
        f"API CHEATSHEET (current bindings):\n{cheatsheet}\n\n"
        f"POLICY CONTRACT TEMPLATE (implement these functions, adapt logic):\n{template}\n\n"
        f"CRITIQUE FROM LAST ITERATION (optional):\n{critique}\n\n"
        f"{extra}"
        f"EXAMPLES (reference style; do NOT copy verbatim):\n{examples}\n\n"
        "Return ONLY Python code for the policy module, no prose."
    )
