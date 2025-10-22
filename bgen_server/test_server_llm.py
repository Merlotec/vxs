from __future__ import annotations
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import threading

import requests
import websockets


SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8098
WS_PORT = 8099
REPO_ROOT = Path(__file__).resolve().parents[1]


async def _ws_listen(run_id: str, timeout_s: float = 90.0) -> list[dict]:
    uri = f"ws://{SERVER_HOST}:{WS_PORT}/ws/progress/{run_id}"
    msgs: list[dict] = []
    end = time.time() + timeout_s
    async with websockets.connect(uri) as ws:
        # Proactively subscribe in case path parsing differs
        await ws.send(json.dumps({"type": "subscribe", "topic": f"progress:{run_id}"}))
        while time.time() < end:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            try:
                data = json.loads(msg)
            except Exception:
                continue
            msgs.append(data)
            if data.get("type") in ("done", "terminated"):
                break
    return msgs


async def _ws_listen_frames(run_id: str, channel: str, timeout_s: float = 8.0) -> int:
    uri = f"ws://{SERVER_HOST}:{WS_PORT}/ws/render/{run_id}/{channel}"
    count = 0
    end = time.time() + timeout_s
    try:
        async with websockets.connect(uri) as ws:
            while time.time() < end:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                try:
                    data = json.loads(msg)
                except Exception:
                    continue
                if data.get("type") == "frame":
                    count += 1
    except Exception:
        pass
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM server integration test (OpenAI)")
    ap.add_argument("--render", action="store_true", help="Enable rendering during the run")
    ap.add_argument("--iterations", type=int, default=4, help="Max LLM iterations (server may conclude early)")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--proxy", dest="proxy", action="store_true", help="Enable TCP→WS proxy (default if --render)")
    grp.add_argument("--no-proxy", dest="proxy", action="store_false", help="Disable proxy (use local voxelsim-renderer over TCP)")
    ap.set_defaults(proxy=True)
    args = ap.parse_args()
    if not os.environ.get("LLVPN_API_KEY"):
        print("Skipping LLM test: LLVPN_API_KEY not set.")
        sys.exit(0)

    env = os.environ.copy()
    env["BGEN_SERVER_LOG"] = "debug"

    cmd = [
        sys.executable,
        "-m",
        "bgen_server.server",
        "--host",
        SERVER_HOST,
        "--port",
        str(SERVER_PORT),
        "--ws-port",
        str(WS_PORT),
        "--no-ws-proxy",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(REPO_ROOT), env=env)

    # Stream server logs
    def _reader():
        if not proc.stdout:
            return
        for line in proc.stdout:
            print("SERVER:", line.rstrip())
    threading.Thread(target=_reader, daemon=True).start()

    try:
        # Wait for health
        health_url = f"http://{SERVER_HOST}:{SERVER_PORT}/health"
        deadline = time.time() + 20.0
        while time.time() < deadline:
            try:
                hr = requests.get(health_url, timeout=0.5)
                if hr.ok:
                    break
            except Exception:
                time.sleep(0.2)
        else:
            raise SystemExit("Server did not start (/health unreachable)")

        # Kick off a run using the LLM (no policy_code), headless, 0 episodes to skip dynamics
        url = f"http://{SERVER_HOST}:{SERVER_PORT}/run"
        body = {
            "user_goal": "Generate a minimal valid policy using A*, which moves the agent to the block with coords (0, 0, 0).",
            "iterations": int(args.iterations),
            "episodes": 1,
            "render": bool(args.render),
            # Request world-time duration; server converts to steps via delta
            "duration_seconds": 30,
            "provider": "openai",
            "provider_url": "https://llvpn.io/v1/service/15695372f78172f965ebf2879254099e1f02a80b3222ea52161dc31eb2cdf7db",
            "model": "gpt-4o-mini",
        }
        if args.render:
            body["ws_proxy"] = bool(args.proxy)
        r = requests.post(url, json=body, timeout=10)
        r.raise_for_status()
        run_id = r.json()["run_id"]
        print("Run ID:", run_id)

        # Listen for progress until done; optionally collect render frames
        async def runner():
            tasks = [
                _ws_listen(run_id, timeout_s=90.0),
            ]
            if args.render and args.proxy:
                tasks.append(_ws_listen_frames(run_id, "world", timeout_s=10.0))
                tasks.append(_ws_listen_frames(run_id, "pov_world_0", timeout_s=10.0))
            results = await asyncio.gather(*tasks)
            return results

        results = asyncio.run(runner())
        msgs = results[0]
        print("WS messages received:", len(msgs))
        if not any(m.get("type") == "done" for m in msgs):
            raise SystemExit("Did not receive 'done' message from LLM run")
        # Check for any error messages
        errs = [m for m in msgs if m.get("type") == "error"]
        if errs:
            raise SystemExit(f"LLM run reported error: {errs}")
        # Print timing if provided
        done_msgs = [m for m in msgs if m.get("type") == "done"]
        if done_msgs and isinstance(done_msgs[-1].get("timing"), dict):
            timing = done_msgs[-1]["timing"]
            sim = timing.get("sim_time_s")
            wall = timing.get("wall_time_s")
            speed = timing.get("speedup_x")
            concluded = done_msgs[-1].get("concluded_early")
            print(f"Timing: sim={sim:.3f}s, wall={wall:.3f}s, speedup={speed:.1f}x, concluded_early={concluded}" if (sim is not None and wall is not None and speed is not None) else f"Timing: {timing}")

        if args.render and args.proxy:
            world_frames = results[1]
            pov_frames = results[2]
            print(f"Render frames: world={world_frames}, pov_world_0={pov_frames}")
            if world_frames == 0:
                raise SystemExit("No world frames observed over WS proxy")
        elif args.render and not args.proxy:
            print("Render enabled without proxy: expecting local voxelsim-renderer to consume TCP streams on 8080/8081/8090/9090.")

        # Fetch generated code
        cr = requests.get(f"http://{SERVER_HOST}:{SERVER_PORT}/code/{run_id}", timeout=10)
        cr.raise_for_status()
        code = cr.json().get("code", "")
        assert code and "def act(" in code
        print("LLM code retrieval OK, size:", len(code))
        print("LLM test passed.")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
