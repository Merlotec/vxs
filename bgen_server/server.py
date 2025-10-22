from __future__ import annotations
import argparse
import asyncio
import base64
import json
import os
import socket
import struct
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

from flask import Flask, request, jsonify
import logging
import websockets

from bgen.orchestrator.prompt import build_system_prompt, build_user_prompt
from bgen.llm.client import LLVPNAnthropicClient, LLVPNOpenAIClient
from bgen.orchestrator.evaluator import aggregate
from bgen.runner.run_policy import run_episode


# ---------------- Websocket Hub -----------------

class WSHub:
    def __init__(self) -> None:
        self._topics: Dict[str, list[websockets.WebSocketServerProtocol]] = {}
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def subscribe(self, topic: str, ws: websockets.WebSocketServerProtocol) -> None:
        with self._lock:
            lst = self._topics.setdefault(topic, [])
            if ws not in lst:
                lst.append(ws)
        logger.debug(f"WS subscribe topic={topic} conn={id(ws)}")

    def unsubscribe(self, topic: str, ws: websockets.WebSocketServerProtocol) -> None:
        with self._lock:
            if topic in self._topics:
                lst = self._topics[topic]
                if ws in lst:
                    lst.remove(ws)
                if not lst:
                    self._topics.pop(topic, None)
        logger.debug(f"WS unsubscribe topic={topic} conn={id(ws)}")

    async def broadcast(self, topic: str, message: Dict[str, Any]) -> None:
        payload = json.dumps(message)
        conns = []
        with self._lock:
            conns = list(self._topics.get(topic, []))
        # Send concurrently; drop failures
        results = await asyncio.gather(*(self._safe_send(ws, payload) for ws in conns), return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logger.debug(f"WS broadcast error: {r}")

    async def _safe_send(self, ws: websockets.WebSocketServerProtocol, payload: str) -> None:
        try:
            await ws.send(payload)
        except Exception:
            # Connection likely closed; no action
            logger.debug(f"WS send failed for conn={id(ws)}")

    # Configure the event loop used by this hub for thread-safe scheduling.
    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    # Thread-safe broadcast: schedule onto the configured event loop.
    def broadcast_sync(self, topic: str, message: Dict[str, Any]) -> None:
        if self._loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self.broadcast(topic, message), self._loop)
        except Exception as e:
            logger.error(f"broadcast_sync schedule failed: {e}")
        # Optionally wait or ignore; here we ignore result to avoid blocking


hub = WSHub()


# --------------- TCP Renderer Proxy ----------------

class RendererTCPProxy:
    """Listens on renderer ports and forwards length-prefixed frames to websocket topics.

    Ports by default:
      - world: 8080
      - agents: 8081
      - pov world streams start: 8090
      - pov agent streams start: 9090
    """

    def __init__(self, run_id: str, pov_count: int = 1, host: str = "127.0.0.1", world_port: int = 8080, agent_port: int = 8081, pov_start: int = 8090, pov_agent_start: int = 9090) -> None:
        self.run_id = run_id
        self.host = host
        self.world_port = world_port
        self.agent_port = agent_port
        self.pov_start = pov_start
        self.pov_agent_start = pov_agent_start
        self.pov_count = pov_count
        self._threads: list[threading.Thread] = []
        self._sockets: list[socket.socket] = []
        self._stop = threading.Event()

    def start(self) -> None:
        ports = [(self.world_port, f"world"), (self.agent_port, f"agents")]
        for i in range(self.pov_count):
            ports.append((self.pov_start + i, f"pov_world_{i}"))
            ports.append((self.pov_agent_start + i, f"pov_agents_{i}"))
        for port, channel in ports:
            t = threading.Thread(target=self._listen, args=(port, channel), daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        for s in self._sockets:
            try:
                s.close()
            except Exception:
                pass

    def _listen(self, port: int, channel: str) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, port))
        srv.listen(1)
        self._sockets.append(srv)
        while not self._stop.is_set():
            try:
                conn, _ = srv.accept()
            except Exception:
                continue
            # Handle one connection per port sequentially
            with conn:
                while not self._stop.is_set():
                    # Read 4-byte little endian length
                    hdr = self._recv_all(conn, 4)
                    if not hdr:
                        break
                    (length,) = struct.unpack("<I", hdr)
                    payload = self._recv_all(conn, length)
                    if payload is None:
                        break
                    # Broadcast frame
                    hub.broadcast_sync(
                        f"render:{self.run_id}:{channel}",
                        {
                            "type": "frame",
                            "channel": channel,
                            "len": length,
                            "payload_b64": base64.b64encode(payload).decode("ascii"),
                            "ts": time.time(),
                        },
                    )

    def _recv_all(self, conn: socket.socket, n: int) -> Optional[bytes]:
        buf = b""
        while len(buf) < n:
            try:
                chunk = conn.recv(n - len(buf))
            except Exception:
                return None
            if not chunk:
                return None
            buf += chunk
        return buf


# ---------------- HTTP + WS Server -----------------

app = Flask(__name__)
logger = logging.getLogger("bgen_server")
_lvl = os.environ.get("BGEN_SERVER_LOG", "info").lower()
logger.setLevel(logging.DEBUG if _lvl == "debug" else logging.INFO)
_sh = logging.StreamHandler()
_sh.setLevel(logging.DEBUG if _lvl == "debug" else logging.INFO)
_sh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s"))
if not logger.handlers:
    logger.addHandler(_sh)
DEFAULT_WS_PROXY = True


@app.route("/health", methods=["GET"])
def health() -> Any:
    return {"status": "ok"}


_runs: Dict[str, Dict[str, Any]] = {}


def _start_training_thread(
    run_id: str,
    user_goal: str,
    iterations: int,
    episodes: int,
    enable_render: bool,
    existing_code: Optional[str],
    enable_ws_proxy: bool,
    policy_code: Optional[str],
    use_px4: bool = True,
    max_steps: int = 500,
    delta: float = 0.01,
    pov_min_dt_world: float = 0.05,
    provider: str = "anthropic",
    provider_url: Optional[str] = None,
    model: Optional[str] = None,
    success_threshold: float = 0.95,
) -> None:
    def on_step(step: Dict[str, Any]) -> None:
        hub.broadcast_sync(f"progress:{run_id}", {"type": "step", "step": step})

    def on_summary(summary: Dict[str, Any]) -> None:
        hub.broadcast_sync(f"progress:{run_id}", {"type": "episode_summary", "summary": summary})

    def worker() -> None:
        logger.info(f"Run {run_id} starting: iterations={iterations}, episodes={episodes}, render={enable_render}, ws_proxy={enable_ws_proxy}")
        # Optionally start renderer proxy
        proxy: Optional[RendererTCPProxy] = None
        # Start WS proxy only if explicitly enabled; this allows testing against a normal renderer.
        if enable_render and enable_ws_proxy:
            proxy = RendererTCPProxy(run_id=run_id, pov_count=1)
            proxy.start()

        outdir = Path("runs") / "bgen_server" / run_id
        # LLM client
        # Select LLM provider
        if provider == "openai":
            client = LLVPNOpenAIClient(
                url=provider_url or "https://llvpn.io/v1/service/15695372f78172f965ebf2879254099e1f02a80b3222ea52161dc31eb2cdf7db",
                api_key_env="LLVPN_API_KEY",
            )
        else:
            client = LLVPNAnthropicClient(
                url=provider_url or "https://llvpn.io/v1/service/1bd998cc32e3e5d34b2f09ec104f7439e0498c22358cdf8c1c11a258c3157f84",
                api_key_env="LLVPN_API_KEY",
            )
        # Iteration loop
        prior_crit: Optional[Path] = None
        stop_event: threading.Event = _runs[run_id]["stop"]
        total_sim_time_s = 0.0
        total_wall_time_s = 0.0
        concluded_early = False
        for it in range(iterations):
            if stop_event.is_set():
                break
            iter_dir = outdir / f"iter_{it}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            system = build_system_prompt()
            user = build_user_prompt(user_goal=user_goal, repo_root=Path.cwd(), prior_critique_path=prior_crit, extra_code=existing_code)
            if policy_code:
                code = policy_code
            else:
                try:
                    # Choose default model if not provided
                    default_model = model or ("gpt-4o-mini" if provider == "openai" else "claude-3-5-sonnet-20241022")
                    code = client.generate(system=system, user=user, model=default_model, max_tokens=2400, temperature=0.2)
                except Exception as e:
                    hub.broadcast_sync(f"progress:{run_id}", {"type": "error", "message": str(e)})
                    break
            policy_path = iter_dir / "policy.generated.py"
            policy_path.write_text(code, encoding="utf-8")
            _runs[run_id]["last_code_path"] = str(policy_path)
            _runs[run_id]["last_code"] = code
            logger.info(f"Run {run_id} iteration {it}: policy saved to {policy_path}")

            # Run episodes
            for ep in range(episodes):
                if stop_event.is_set():
                    break
                seed = ep
                ep_result = run_episode(
                    policy_code=code,
                    outdir=iter_dir / f"ep_{seed}",
                    seed=seed,
                    max_steps=max_steps,
                    delta=delta,
                    enable_render=enable_render,
                    pov_size=(200, 150),
                    target=None,
                    use_px4=use_px4,
                    pov_min_dt_world=pov_min_dt_world,
                    on_step=on_step,
                    on_summary=on_summary,
                    stop_event=stop_event,
                )
                try:
                    total_sim_time_s += float(ep_result.get("sim_time_s", 0.0))
                    total_wall_time_s += float(ep_result.get("wall_time_s", 0.0))
                except Exception:
                    pass

            # Evaluate
            if stop_event.is_set():
                break
            agg, critique = aggregate(iter_dir)
            (iter_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))
            (iter_dir / "critique.txt").write_text(critique)
            hub.broadcast_sync(f"progress:{run_id}", {"type": "critique", "critique": critique, "aggregate": agg})
            prior_crit = iter_dir / "critique.txt"
            # Early stop if success meets threshold
            try:
                sr = float(agg.get("success_rate", 0.0))
                if sr >= success_threshold:
                    concluded_early = True
                    break
            except Exception:
                pass

        if proxy is not None:
            proxy.stop()
        _runs[run_id]["status"] = "done"
        timing = {
            "sim_time_s": total_sim_time_s,
            "wall_time_s": total_wall_time_s,
            "speedup_x": (total_sim_time_s / total_wall_time_s) if total_wall_time_s > 0 else None,
        }
        hub.broadcast_sync(f"progress:{run_id}", {"type": "done", "timing": timing, "concluded_early": concluded_early})
        logger.info(f"Run {run_id} finished")

    threading.Thread(target=worker, daemon=True).start()


@app.route("/run", methods=["POST"])
def run_training() -> Any:
    data = request.get_json(force=True) or {}
    logger.info(f"/run payload: {data}")
    user_goal = data.get("user_goal", "Reach target while minimizing collisions")
    iterations = int(data.get("iterations", 1))
    episodes = int(data.get("episodes", 3))
    enable_render = bool(data.get("render", True))
    # Simulation pacing
    # World time per step is `delta`; if caller does not provide max_steps, compute from desired duration.
    # `duration_seconds` means world seconds (steps * delta), not wall time.
    max_steps = data.get("max_steps")
    try:
        delta = float(data.get("delta", 0.01))
    except Exception:
        delta = 0.01
    # Derive steps from duration if needed
    if max_steps is None:
        try:
            duration_world_s = float(data.get("duration_seconds", data.get("duration", 30.0)))
        except Exception:
            duration_world_s = 30.0
        # Ceil to ensure at least the requested duration
        import math as _math
        max_steps = int(max(1, _math.ceil(duration_world_s / max(delta, 1e-6))))
    else:
        max_steps = int(max_steps)
    provider = str(data.get("provider", "anthropic")).lower()
    provider_url = data.get("provider_url")
    model = data.get("model")
    # Optional: POV cadence in world seconds (min interval between POV updates)
    try:
        pov_min_dt_world = float(data.get("pov_min_dt_world", 0.05))
    except Exception:
        pov_min_dt_world = 0.05
    success_threshold = float(data.get("success_threshold", 0.95))
    # Dynamics selection flag (support both keys for compatibility)
    use_px4 = bool(data.get("px4", data.get("use_px4", True)))
    # Allow request to override; otherwise fall back to server default
    if "ws_proxy" in data:
        enable_ws_proxy = bool(data.get("ws_proxy"))
    else:
        enable_ws_proxy = DEFAULT_WS_PROXY
    existing_code = data.get("existing_code")  # Optional seed code for refinement
    policy_code = data.get("policy_code")      # Optional: bypass LLM and use this policy directly

    run_id = uuid.uuid4().hex[:8]
    _runs[run_id] = {"status": "running", "started": time.time(), "stop": threading.Event(), "last_code": None, "last_code_path": None}
    _start_training_thread(
        run_id,
        user_goal,
        iterations,
        episodes,
        enable_render,
        existing_code,
        enable_ws_proxy,
        policy_code,
        use_px4=use_px4,
        max_steps=max_steps,
        delta=delta,
        pov_min_dt_world=pov_min_dt_world,
        provider=provider,
        provider_url=provider_url,
        model=model,
        success_threshold=success_threshold,
    )
    return jsonify({"run_id": run_id})


@app.route("/code/<run_id>", methods=["GET"])
def get_code(run_id: str) -> Any:
    info = _runs.get(run_id)
    if not info:
        return jsonify({"error": "run not found"}), 404
    # Prefer in-memory; fallback to file
    code = info.get("last_code")
    if not code and info.get("last_code_path"):
        try:
            code = Path(info["last_code_path"]).read_text(encoding="utf-8")
        except Exception:
            code = None
    if not code:
        return jsonify({"error": "code not available yet"}), 404
    return jsonify({"run_id": run_id, "code": code})


async def ws_handler(websocket: websockets.WebSocketServerProtocol) -> None:
    # Supports either path-based subscription (/ws/progress/<run_id>) or
    # message-based subscription: {"type": "subscribe", "topic": "progress:<run_id>"}.
    topic: Optional[str] = None
    try:
        path = getattr(websocket, "path", None)
        if not path and hasattr(websocket, "request"):
            req = getattr(websocket, "request")
            path = getattr(req, "path", "")
        path = path or ""
        parts = [p for p in str(path).split("/") if p]
        if len(parts) >= 3 and parts[0] == "ws" and parts[1] == "progress":
            run_id = parts[2]
            topic = f"progress:{run_id}"
        elif len(parts) >= 4 and parts[0] == "ws" and parts[1] == "render":
            run_id = parts[2]
            channel = parts[3]
            topic = f"render:{run_id}:{channel}"
        else:
            # Fallback: wait for a subscribe message
            try:
                first = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                data = json.loads(first)
                if isinstance(data, dict) and data.get("type") == "subscribe" and "topic" in data:
                    topic = str(data["topic"]) 
                else:
                    await websocket.close()
                    return
            except Exception:
                await websocket.close()
                return
        hub.subscribe(topic, websocket)
        # Send initial ready signal
        try:
            await websocket.send(json.dumps({"type": "connected", "topic": topic}))
        except Exception:
            return
        logger.debug(f"WS connected path={path} topic={topic} conn={id(websocket)}")
        # If progress subscription and run already done, emit done immediately
        if topic.startswith("progress:"):
            rid = topic.split(":", 1)[1]
            info = _runs.get(rid)
            if info and info.get("status") == "done":
                try:
                    await websocket.send(json.dumps({"type": "done"}))
                except Exception:
                    pass
        # Keep alive and accept control messages
        async for message in websocket:
            try:
                data = json.loads(message)
            except Exception as e:
                logger.debug(f"WS received non-JSON message: {e}")
                continue
            if not isinstance(data, dict):
                continue
            # Allow termination for progress topic
            if topic.startswith("progress:"):
                if data.get("type") == "terminate":
                    run_id = topic.split(":", 1)[1]
                    info = _runs.get(run_id)
                    if info and "stop" in info:
                        info["stop"].set()
                        # Acknowledge to sender
                        try:
                            await websocket.send(json.dumps({"type": "terminated"}))
                        except Exception:
                            pass
                    else:
                        await hub.broadcast(topic, {"type": "error", "message": "run not found"})
    except Exception as e:
        logger.error(f"WS handler error: {e}")
        return
    finally:
        try:
            if topic:
                hub.unsubscribe(topic, websocket)
        except Exception:
            pass


async def _ws_main(host: str, ws_port: int) -> None:
    loop = asyncio.get_running_loop()
    hub.set_loop(loop)
    logger.info(f"WebSocket server starting on {host}:{ws_port}")
    async with websockets.serve(ws_handler, host, ws_port):
        # Run indefinitely until cancelled
        await asyncio.Future()


def main() -> None:
    ap = argparse.ArgumentParser(description="bgen-server: LLM-in-the-loop training service")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--ws-port", type=int, default=8089)
    # Toggle default WS proxy behavior; can still be overridden per-run via POST /run ws_proxy flag
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--ws-proxy", dest="ws_proxy", action="store_true", help="Enable WS proxy by default (can be overridden per run)")
    grp.add_argument("--no-ws-proxy", dest="ws_proxy", action="store_false", help="Disable WS proxy by default (can be overridden per run)")
    ap.set_defaults(ws_proxy=True)
    args = ap.parse_args()

    global DEFAULT_WS_PROXY
    DEFAULT_WS_PROXY = bool(args.ws_proxy)

    # Launch Flask (HTTP) in one thread
    http_thread = threading.Thread(target=lambda: app.run(host=args.host, port=args.port, debug=False, use_reloader=False), daemon=True)
    http_thread.start()

    # Launch websockets server (WS) in main thread using asyncio.run
    try:
        asyncio.run(_ws_main(args.host, args.ws_port))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
