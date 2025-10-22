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
        "Do not include explanations or backticks. Prefer A* path planning to move."
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
        examples = f"\n# Example: move_to_target.py\n{ex1}\n\n# Example: reactive_avoidance.py\n{ex2}"
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
