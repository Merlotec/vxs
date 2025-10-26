from __future__ import annotations
import argparse
import json
import statistics as stats
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _find_episode_summaries(root: Path) -> List[Path]:
    return list(root.rglob("summary.*.json"))


def _load_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def aggregate(root: Path) -> Tuple[Dict[str, Any], str]:
    summaries = _find_episode_summaries(root)
    if not summaries:
        return ({"episodes": 0}, "No episodes found to evaluate.")

    succ: List[int] = []
    steps: List[int] = []
    collisions: List[int] = []
    path_len: List[float] = []
    policy_summaries: List[str] = []

    for sp in summaries:
        s = _load_json(sp)
        if not s:
            continue
        succ.append(1 if s.get("success") else 0)
        steps.append(int(s.get("steps", 0)))
        collisions.append(int(s.get("collisions_total", 0)))
        ple = s.get("path_len_est")
        if isinstance(ple, (int, float)):
            path_len.append(float(ple))
        ps = s.get("policy_summary", {})
        if isinstance(ps, dict):
            txt = ps.get("summary")
            if isinstance(txt, str) and txt.strip():
                policy_summaries.append(txt.strip())

    n = len(steps)
    success_rate = sum(succ) / n if n else 0.0
    mean_steps = stats.mean(steps) if steps else 0.0
    med_steps = stats.median(steps) if steps else 0.0
    mean_collisions = stats.mean(collisions) if collisions else 0.0
    med_collisions = stats.median(collisions) if collisions else 0.0
    mean_path = stats.mean(path_len) if path_len else None

    agg: Dict[str, Any] = {
        "episodes": n,
        "success_rate": success_rate,
        "steps": {
            "mean": mean_steps,
            "median": med_steps,
            "min": min(steps) if steps else 0,
            "max": max(steps) if steps else 0,
        },
        "collisions_total": {
            "mean": mean_collisions,
            "median": med_collisions,
            "min": min(collisions) if collisions else 0,
            "max": max(collisions) if collisions else 0,
        },
        "path_len_est": {
            "mean": mean_path,
            "min": min(path_len) if path_len else None,
            "max": max(path_len) if path_len else None,
        },
        "sample_policy_summaries": policy_summaries[:5],
    }

    # Generate a concise critique for LLM feedback
    notes: List[str] = []
    if success_rate < 0.5:
        notes.append(
            "Low success rate: ensure planning triggers when idle and replan when stuck; prefer A* with sufficient padding."
        )
    elif success_rate < 0.9:
        notes.append(
            "Moderate success rate: consider increasing A* padding near obstacles and adding simple recovery steps on failure."
        )
    else:
        notes.append("High success rate: refine path efficiency and collision avoidance further.")

    if mean_collisions > 0.0:
        notes.append(
            "Non-zero collisions: increase planner padding, replan on collision spikes, or bias paths away from dense voxels."
        )

    if mean_steps > 0 and mean_path and mean_path / max(mean_steps, 1) > 0.75:
        notes.append(
            "Path efficiency could be improved: prefer shorter sequences and re-evaluate yaw; inject waypoints if needed."
        )

    critique_lines: List[str] = [
        f"Episodes: {n}, Success Rate: {success_rate:.2f}",
        f"Steps (mean/median/min/max): {mean_steps:.1f}/{med_steps:.1f}/{min(steps) if steps else 0}/{max(steps) if steps else 0}",
        f"Collisions total (mean/median/min/max): {mean_collisions:.1f}/{med_collisions:.1f}/{min(collisions) if collisions else 0}/{max(collisions) if collisions else 0}",
    ]
    if mean_path is not None:
        critique_lines.append(
            f"Path length est. (mean/min/max): {mean_path:.1f}/{min(path_len):.1f}/{max(path_len):.1f}"
        )
    critique_lines.append("Recommendations:")
    for nline in notes:
        critique_lines.append(f"- {nline}")

    if policy_summaries:
        critique_lines.append("Sample policy summaries:")
        for s in policy_summaries[:3]:
            critique_lines.append(f"- {s}")

    critique = "\n".join(critique_lines)

    # Add execution trace from failed episodes for LLM debugging
    trace_files = list(root.glob("ep_*/trace.*.json"))
    if trace_files:
        trace_path = trace_files[0]
        try:
            trace_data = json.loads(trace_path.read_text())
            # Sample trace to avoid token limit
            if len(trace_data) > 150:
                sampled = trace_data[:50] + trace_data[-100:]
                critique += "\n\nEXECUTION TRACE (first 50 + last 100 steps):\n"
                critique += json.dumps(sampled, indent=2)
            else:
                critique += "\n\nEXECUTION TRACE (full):\n"
                critique += json.dumps(trace_data, indent=2)
            critique += "\n\nAnalyze the trace above to identify why the agent failed and what bug caused it. Look for patterns like stuck positions, unchanging targets, or stage transition issues."
        except Exception:
            pass

    return agg, critique


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate episode summaries and produce a critique")
    ap.add_argument("--root", type=str, required=True, help="Path to an iteration dir (e.g., runs/.../iter_0)")
    args = ap.parse_args()
    root = Path(args.root)
    agg, critique = aggregate(root)
    (root / "aggregate.json").write_text(json.dumps(agg, indent=2))
    (root / "critique.txt").write_text(critique)
    print(critique)


if __name__ == "__main__":
    main()

