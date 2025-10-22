# Behavior Generator (bgen)

This module hosts an LLM-in-the-loop behavior generation workflow for VoxelSim. The runner mirrors the control/render loop style in `python/povtest.py`, but executes at max speed (no realtime sleeps) and gates POV updates using `FilterWorld.is_updating_py` to avoid overlapping frame work.

- LLM receives a compact API cheatsheet and examples, then outputs a Python policy.
- Policies obey a small contract: control via `act(...)` and data logging via `collect(...)` and `finalize(...)`.
- A sandboxed runner executes policies against `voxelsim` and emits step logs and episode summaries.
- An (optional) orchestrator iterates policies across episodes and seeds; summaries feed back to the LLM.

Contents
- `llm/context/`: Docs, policy template, and few-shot examples given to the LLM
- `runner/`: Sandbox, helpers, metrics, schemas, and a CLI runner for episodes
- `orchestrator/`: Iteration utilities (scaffold) to facilitate LLM loops without hard-coding a vendor

See `llm/context/API_CHEATSHEET.md` for the Python API surface and `llm/context/POLICY_TEMPLATE.py` for the required policy contract.

## LLM Integration

The orchestrator can generate policy code via LLVPN-backed Anthropic or OpenAI endpoints.

Env setup:
- Export your token (provided: `b8phulmolutozl5jy3nqh`) into `LLVPN_API_KEY`:
  - `export LLVPN_API_KEY=b8phulmolutozl5jy3nqh`

Run iteration with code generation (Anthropic default):
- `python -m bgen.orchestrator.iterate --use-llm --user-goal "Reach target safely" --iterations 1 --episodes 3`

Use OpenAI provider instead:
- `python -m bgen.orchestrator.iterate --use-llm --provider openai --model gpt-4o-mini --iterations 1 --episodes 3`

Flags:
- `--provider` anthropic | openai (default anthropic)
- `--llvpn-url` Anthropic URL (default wired to provided)
- `--openai-url` OpenAI URL (default wired to provided)
- `--api-key-env` (default `LLVPN_API_KEY`)
- `--model` (Anthropic default `claude-3-5-sonnet-20241022`; set appropriate model for OpenAI)
- `--max-tokens`, `--temperature`

Notes:
- The client uses the Anthropic messages schema with `system` + `messages` and parses the first text block.
- If the model returns fenced code, the client extracts the first Python block.

## bgen-server (HTTP + WebSockets)

Run an HTTP API and WebSocket service to kick off training loops and stream progress and render frames.

Start server:
- `python -m bgen_server.server --host 127.0.0.1 --port 8088 --ws-port 8089` (WS proxy on by default)
- Disable WS proxy by default: `python -m bgen_server.server --no-ws-proxy`

HTTP endpoints:
- `POST /run` with JSON `{ "user_goal": "...", "iterations": 1, "episodes": 3, "render": true, "ws_proxy": true }`
  - Returns `{ "run_id": "<id>" }`
  - Optional: include `"existing_code": "<python module text>"` to guide/refine the generated behavior.

WebSockets:
- Progress: `ws://<host>:8089/ws/progress/<run_id>`
  - Messages: `{ type: "step" | "episode_summary" | "critique" | "done", ... }`
  - Control: send `{ "type": "terminate" }` to request run termination.
- Render frames via TCP→WS proxy (uses renderer ports 8080/8081/8090/9090):
  - World stream: `ws://<host>:8089/ws/render/<run_id>/world`
  - Agents stream: `ws://<host>:8089/ws/render/<run_id>/agents`
  - POV world stream 0: `.../pov_world_0`
  - POV agents stream 0: `.../pov_agents_0`
  - Frame payload: `{ type: "frame", channel, len, payload_b64, ts }`

Caveats:
- To test with the normal Bevy renderer (no WS frame proxy), either start the server with `--no-ws-proxy` or set `"ws_proxy": false` in the `/run` request. The runner will still emit to the usual TCP ports and your renderer can consume them directly.
- If `"ws_proxy": true` (default), the server binds to ports 8080/8081/8090/9090 to capture frames and forward them to WebSockets; do not run the Bevy renderer at the same time.
- Ensure `LLVPN_API_KEY` is exported for LLM generation.

Fetching Generated Code
- `GET /code/<run_id>` returns the latest generated policy code for that run.

## Testing the Server

Quick integration test (uses a minimal inline policy and disables rendering):

- Ensure `voxelsim` Python bindings are installed (see repo README) and `Flask`, `websockets`, and `requests` are available (`pip install Flask websockets requests`).
- Run: `python bgen_server/test_server.py`
- What it does:
  - Starts the server on ports 8098 (HTTP) and 8099 (WS) with `--no-ws-proxy`.
  - POSTs `/run` with a minimal `policy_code` (bypasses LLM) and `render: false`.
  - Listens on the progress WebSocket until it receives `done`.
  - Fetches the policy via `GET /code/<run_id>` and asserts it contains `def act`.
  - Prints a short summary and exits.
