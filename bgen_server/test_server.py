from __future__ import annotations
import asyncio
import json
import os
import signal
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


MINIMAL_POLICY = """
import voxelsim as vxs
from typing import Any, Dict, Optional

def init(config: Dict[str, Any]) -> None:
    pass

def act(t: float, agent: vxs.Agent, world: vxs.VoxelGrid, fw: vxs.FilterWorld, env: vxs.EnvState, helpers: Any) -> Optional[vxs.ActionIntent]:
    if agent.get_action_py() is None:
        origin = agent.get_coord_py()
        try:
            return helpers.plan_to(world, origin, [55,55,-20], yaw=0.0, urgency=0.8, padding=1)
        except Exception:
            return helpers.intent(0.6, 0.0, [vxs.MoveDir.Forward])
    return None

def collect(step_ctx: Dict[str, Any]) -> Dict[str, str]:
    return {"note": "ok"}

def finalize(ep_ctx: Dict[str, Any]) -> Dict[str, str]:
    return {"summary": f"steps={ep_ctx.get('steps')}, success={ep_ctx.get('success')}"}
"""


async def _ws_listen(run_id: str, timeout_s: float = 30.0) -> list[dict]:
    uri = f"ws://{SERVER_HOST}:{WS_PORT}/ws/progress/{run_id}"
    msgs: list[dict] = []
    end = time.time() + timeout_s
    async with websockets.connect(uri) as ws:
        # Fallback: also send an explicit subscribe message in case path parsing fails
        await ws.send(json.dumps({"type": "subscribe", "topic": f"progress:{run_id}"}))
        while time.time() < end:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
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


def main() -> None:
    # Start server as a subprocess on test ports with WS proxy disabled by default
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
    # Run from repo root so that 'bgen' and 'bgen_server' packages resolve
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(REPO_ROOT))
    # Stream server output for debugging
    def _reader():
        if not proc.stdout:
            return
        for line in proc.stdout:
            print("SERVER:", line.rstrip())
    t_reader = threading.Thread(target=_reader, daemon=True)
    t_reader.start()
    try:
        # Wait for /health to respond or timeout
        health_url = f"http://{SERVER_HOST}:{SERVER_PORT}/health"
        deadline = time.time() + 10.0
        while time.time() < deadline:
            try:
                hr = requests.get(health_url, timeout=0.5)
                if hr.ok:
                    break
            except Exception:
                time.sleep(0.2)
        else:
            # Print server logs for debugging
            try:
                out = proc.stdout.read() if proc.stdout else ""
                print("Server logs:\n", out)
            except Exception:
                pass
            raise SystemExit("Server did not start (/health unreachable)")

        # Kick off a run with inline policy_code (bypasses LLM) and no rendering
        url = f"http://{SERVER_HOST}:{SERVER_PORT}/run"
        body = {
            "user_goal": "Test run",
            "iterations": 1,
            "episodes": 0,
            "render": False,
            "policy_code": MINIMAL_POLICY,
        }
        r = requests.post(url, json=body, timeout=10)
        r.raise_for_status()
        run_id = r.json()["run_id"]
        print("Run ID:", run_id)

        # Listen to progress websocket until done
        msgs = asyncio.run(_ws_listen(run_id))
        print("WS messages received:", len(msgs))
        if not any(m.get("type") == "done" for m in msgs):
            raise SystemExit("Did not receive 'done' message")

        # Fetch generated/used code
        cr = requests.get(f"http://{SERVER_HOST}:{SERVER_PORT}/code/{run_id}", timeout=10)
        cr.raise_for_status()
        code = cr.json().get("code", "")
        assert "def act(" in code
        print("Code retrieval OK, size:", len(code))

        print("Test passed.")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
