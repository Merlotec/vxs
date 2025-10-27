from __future__ import annotations
import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Callable
import threading

import voxelsim as vxs

from .helpers import Helpers
from .metrics import distance_to
from .schema import StepLog, EpisodeSummary
from .sandbox import load_policy_module, SandboxError


def _load_policy_from_path(path: Path) -> str:
    return path.read_text()


def run_episode(
    policy_code: str,
    outdir: Path,
    seed: int,
    max_steps: int,
    delta: float,
    enable_render: bool,
    pov_size: tuple[int, int],
    target: Optional[tuple[float, float, float]],
    use_px4: bool,
    pov_min_dt_world: float = 0.05,
    on_step: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_summary: Optional[Callable[[Dict[str, Any]], None]] = None,
    stop_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    # Sandbox load policy
    policy = load_policy_module(policy_code)

    # World & sim setup
    gen = vxs.TerrainGenerator()
    cfg = vxs.TerrainConfig.default_py()
    cfg.set_world_size_py(200) 
    # Set world seed per episode to ensure reset and variability
    try:
        cfg.set_seed_py(int(seed) & 0xFFFFFFFF)
    except Exception:
        pass
    gen.generate_terrain_py(cfg)
    world = gen.generate_world_py()

    agent = vxs.Agent(0)
    agent.set_hold_py([50, 50, -20], 0.0)

    fw = vxs.FilterWorld()
    proj = vxs.CameraProjection.default_py()
    cam = vxs.CameraOrientation.vertical_tilt_py(-0.5)
    noise = vxs.NoiseParams.default_with_seed_py([0.0, 0.0, 0.0])
    renderer = vxs.AgentVisionRenderer(world, [pov_size[0], pov_size[1]], noise)

    if use_px4:
        try:
            dyn = vxs.px4.Px4Dynamics.default_py()  # type: ignore[attr-defined]
        except Exception:
            dyn = vxs.QuadDynamics(vxs.QuadParams.default_py())
    else:
        dyn = vxs.QuadDynamics(vxs.QuadParams.default_py())

    env = vxs.EnvState.default_py()

    client = None
    if enable_render:
        client = vxs.RendererClient.default_localhost_py(1)
        client.send_world_py(world)
        last_world_send = time.time()
        WORLD_REFRESH_INTERVAL_S = 1.0
        client.send_agents_py({0: agent})

    # Policy init
    config: Dict[str, Any] = {}
    if target is not None:
        config["target"] = target
    if hasattr(policy, "init") and callable(policy.init):
        policy.init(config)

    helpers = Helpers()
    logs_path = outdir / f"steps.{seed}.jsonl"
    outdir.mkdir(parents=True, exist_ok=True)
    steps_file = logs_path.open("w", encoding="utf-8")

    trace_log = []

    steps = 0
    collisions_total = 0
    last_view_time = time.time()
    last_pov_world_t = -1e9
    success = False
    path_len_accum = 0.0
    last_pos = tuple(agent.get_pos())
    t0 = time.time()
    chaser = vxs.FixedLookaheadChaser.default_py()

    while steps < max_steps:
        if stop_event is not None and stop_event.is_set():
            break
        now = time.time()
        t = now - t0
        sim_t = steps * delta

        # Policy control
        try:
            result = policy.act(t, agent, world, fw, env, helpers)
        except Exception as e:
            # Fail closed: stop episode on policy error
            break
        # Support returning either an intent, or (intent, command) where command is "Replace" or "Push".
        intent = None
        queue_cmd = "Replace"
        if result is not None:
            if isinstance(result, tuple) and len(result) == 2:
                intent, queue_cmd = result
            else:
                intent = result
        if intent is not None:
            try:
                if str(queue_cmd).lower().startswith("push"):
                    agent.push_back_intent_py(intent)
                else:
                    agent.perform_oneshot_py(intent)
            except Exception:
                # Ignore invalid intents
                pass

        # Dynamics step
        # Guard against empty/no action trajectories (can cause Rust-side bspline panics)
        if agent.get_action_py() is not None:
            chase_target = chaser.step_chase_py(agent, delta)
            dyn.update_agent_dynamics_py(agent, env, chase_target, delta)

        # Optional POV & render
        if enable_render and client is not None:
            try:
                # Gate submissions to avoid overlapping POV updates
                if not fw.is_updating_py(last_view_time):
                    # Also gate by simulation time to achieve a minimum world-time cadence
                    if (sim_t - last_pov_world_t) >= pov_min_dt_world:
                        fw.send_pov_py(client, 0, 0, proj, cam)
                        renderer.update_filter_world_py(
                            agent.camera_view_py(cam), proj, fw, now, lambda *_: None
                        )
                        last_view_time = now
                        last_pov_world_t = sim_t
                # Periodically refresh the world snapshot to keep renderer in sync
                if (now - last_world_send) >= WORLD_REFRESH_INTERVAL_S:
                    client.send_world_py(world)
                    last_world_send = now
                client.send_agents_py({0: agent})
            except Exception:
                pass

        # Collisions and metrics
        cols = world.collisions_py(agent.get_pos(), [0.5, 0.5, 0.3])
        collisions_total += len(cols)
        pos = tuple(agent.get_pos())
        coord = tuple(agent.get_coord_py())
        path_len_accum += math.dist(pos, last_pos)
        last_pos = pos

        dist_to_target = None
        if target is not None:
            dist_to_target = distance_to(target, pos)
            if dist_to_target < 1.0:
                success = True
                break

        # Action info
        action = agent.get_action_py()
        cmd_len = 0
        if action:
            q = action.get_intent_queue()
            if len(q) > 0:
                cmd_len = q[0].len()

        # Policy collect
        step_ctx: Dict[str, Any] = {
            "t": t,
            "agent_pos": pos,
            "agent_coord": coord,
            "collisions_count": len(cols),
            "distance_to_target": dist_to_target,
        }
        policy_log: Dict[str, str]
        try:
            policy_log = policy.collect(step_ctx)
        except Exception as e:
            import traceback
            print(f"WARNING: policy.collect() failed: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            policy_log = {"collect_error": str(e)}

        # Collect trace for debugging
        trace_log.append({
            "step": steps,
            "origin": list(coord),
            "policy_log": policy_log,
        })

        # Emit step log
        s = StepLog(
            t=t,
            agent_pos=pos,
            agent_coord=coord,
            yaw=0.0,
            command_len=cmd_len,
            collisions_count=len(cols),
            distance_to_target=dist_to_target,
            frame_time_ms=(time.time() - now) * 1000.0,
            policy_log=policy_log,
        )
        s_dict = s.to_dict()
        steps_file.write(json.dumps(s_dict) + "\n")
        if on_step:
            try:
                on_step(s_dict)
            except Exception:
                pass
        steps += 1

    steps_file.close()

    # Finalize
    # Compute timing
    wall_time_s = time.time() - t0
    sim_time_s = float(steps) * float(delta)

    ep_ctx: Dict[str, Any] = {
        "success": success,
        "steps": steps,
        "collisions_total": collisions_total,
        "avg_command_len": None,
        "path_len_est": path_len_accum,
        "coverage": None,
        "time_to_goal": None,
        "reasons": None,
        "sim_time_s": sim_time_s,
        "wall_time_s": wall_time_s,
    }
    try:
        policy_summary = policy.finalize(ep_ctx)
    except Exception:
        policy_summary = {}
    ep = EpisodeSummary(
        success=success,
        steps=steps,
        collisions_total=collisions_total,
        avg_command_len=0.0,
        path_len_est=path_len_accum,
        coverage=None,
        time_to_goal=None,
        reasons=None,
        policy_summary=policy_summary,
    )
    # Save summary
    ep_dict = ep.to_dict()
    # Attach timing for downstream consumers (WS server)
    ep_dict["sim_time_s"] = sim_time_s
    ep_dict["wall_time_s"] = wall_time_s
    (outdir / f"summary.{seed}.json").write_text(json.dumps(ep_dict, indent=2))
    (outdir / f"summary.{seed}.txt").write_text(policy_summary.get("summary", ""))

    # Save trace for failed episodes (for LLM debugging)
    if not success:
        trace_path = outdir / f"trace.{seed}.json"
        trace_path.write_text(json.dumps(trace_log, indent=2))
    if on_summary:
        try:
            on_summary(ep_dict)
        except Exception:
            pass

    return ep_dict


def main() -> None:
    ap = argparse.ArgumentParser(description="Run an LLM-generated policy against voxelsim")
    ap.add_argument("--policy", type=str, required=True, help="Path to policy .py file")
    ap.add_argument("--outdir", type=str, default="runs/bgen", help="Output directory")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--render", action="store_true", help="Enable renderer updates")
    ap.add_argument("--pov-size", type=str, default="200x150")
    ap.add_argument("--target", type=str, default=None, help="Target coord as x,y,z (grid coord with z negative above ground)")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--px4", dest="px4", action="store_true", help="Use PX4 dynamics if available (default)")
    grp.add_argument("--no-px4", dest="px4", action="store_false", help="Disable PX4; use QuadDynamics")
    ap.set_defaults(px4=True)
    args = ap.parse_args()

    pov_wh = tuple(map(int, args.pov_size.lower().split("x")))
    outdir = Path(args.outdir)
    policy_code = _load_policy_from_path(Path(args.policy))

    target = None
    if args.target:
        tx, ty, tz = map(int, args.target.split(","))
        target = (tx, ty, tz)

    for i in range(args.episodes):
        seed = args.seed_start + i
        run_episode(
            policy_code=policy_code,
            outdir=outdir / f"ep_{seed}",
            seed=seed,
            max_steps=args.max_steps,
            delta=args.delta,
            enable_render=args.render,
            pov_size=pov_wh,  # type: ignore[arg-type]
            target=target,
            use_px4=args.px4,
        )


if __name__ == "__main__":
    main()
