from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
import subprocess
import sys

from ..llm.client import LLVPNAnthropicClient, LLVPNOpenAIClient, DirectOpenAIClient
from .prompt import build_system_prompt, build_user_prompt


"""
Scaffold for iteration without binding to a specific LLM vendor.

Usage examples:
  # Use an existing policy file
  python -m bgen.orchestrator.iterate --policy bgen/llm/context/examples/move_to_target.py \
      --iterations 1 --episodes 3 --outdir runs/bgen_iter

  # (Future) Provide prompt and connect to an LLM to generate new policy code
  # Then call the runner to evaluate it.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Iterate policy generation and evaluation")
    ap.add_argument("--policy", type=str, required=False, help="Path to an existing policy .py (skip LLM)")
    ap.add_argument("--outdir", type=str, default="runs/bgen_iter")
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--target", type=str, default="60,60,-20", help="Target position as x,y,z")
    # LLM options
    ap.add_argument("--use-llm", action="store_true", help="Generate the policy via LLM")
    ap.add_argument("--user-goal", type=str, default="Reach the target while minimizing collisions.")
    ap.add_argument("--provider", type=str, choices=["anthropic", "openai", "openai-direct"], default="anthropic")
    ap.add_argument("--model", type=str, default="claude-3-5-sonnet-20241022")
    ap.add_argument("--max-tokens", type=int, default=2400)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--llvpn-url", type=str, default="https://llvpn.io/v1/service/1bd998cc32e3e5d34b2f09ec104f7439e0498c22358cdf8c1c11a258c3157f84")
    ap.add_argument("--openai-url", type=str, default="https://llvpn.io/v1/service/15695372f78172f965ebf2879254099e1f02a80b3222ea52161dc31eb2cdf7db")
    ap.add_argument("--api-key-env", type=str, default="LLVPN_API_KEY")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # For now, iterate either using a given policy or generating one with LLM each iteration.
    for it in range(args.iterations):
        iter_dir = outdir / f"iter_{it}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        # Choose policy path: either provided or generated via LLM
        policy_path: Optional[str] = args.policy
        if args.use_llm:
            # Build prompt pack
            system = build_system_prompt()
            prior_crit = iter_dir.parent / f"iter_{it-1}" / "critique.txt" if it > 0 else None
            user = build_user_prompt(user_goal=args.user_goal, repo_root=Path.cwd(), prior_critique_path=prior_crit)
            if args.provider == "openai":
                client = LLVPNOpenAIClient(url=args.openai_url, api_key_env=args.api_key_env)
            elif args.provider == "openai-direct":
                client = DirectOpenAIClient(api_key_env="OPENAI_API_KEY")
            else:
                client = LLVPNAnthropicClient(url=args.llvpn_url, api_key_env=args.api_key_env)
            # Generate code
            code = client.generate(system=system, user=user, model=args.model, max_tokens=args.max_tokens, temperature=args.temperature)
            gen_path = iter_dir / "policy.generated.py"
            gen_path.write_text(code, encoding="utf-8")
            policy_path = str(gen_path)

        if not policy_path:
            raise SystemExit("No policy path provided and --use-llm not set")

        cmd = [
            sys.executable,
            "-m",
            "bgen.runner.run_policy",
            "--policy",
            policy_path,
            "--outdir",
            str(iter_dir),
            "--episodes",
            str(args.episodes),
            "--target",
            args.target,
        ]
        if args.render:
            cmd.append("--render")
        subprocess.run(cmd, check=False)

        # Evaluate this iteration
        eval_cmd = [
            sys.executable,
            "-m",
            "bgen.orchestrator.evaluator",
            "--root",
            str(iter_dir),
        ]
        subprocess.run(eval_cmd, check=False)


if __name__ == "__main__":
    main()
