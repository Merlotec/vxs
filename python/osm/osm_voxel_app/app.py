import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

from flask import Flask, render_template, request, jsonify

# Support running as a script (python app.py) and as a module (python -m osm_voxel_app.app)
try:
    # Package-relative import
    from .osm2voxel import generate_voxel_grid_from_point
except Exception:
    # Fallback for direct execution without package context
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from osm2voxel import generate_voxel_grid_from_point


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/generate")
    def generate():
        try:
            data: Dict[str, Any] = request.get_json(force=True)
            lat = float(data.get("lat"))
            lon = float(data.get("lon"))
            dims = data.get("dims") or {}
            nx = int(dims.get("x"))
            ny = int(dims.get("y"))
            nz = int(dims.get("z"))
            meters_per_voxel = float(data.get("meters_per_voxel", 1.0))
            backend = (data.get("backend") or "python").lower()

            # Optional output directory
            out_dir = Path(data.get("out_dir") or Path.cwd() / "runs" / "osm_voxel")
            out_dir.mkdir(parents=True, exist_ok=True)

            vg, serial = generate_voxel_grid_from_point(
                lat=lat,
                lon=lon,
                dims=(nx, ny, nz),
                meters_per_voxel=meters_per_voxel,
                work_dir=out_dir,
                backend=backend,
            )
            # Persist in a JSON-friendly schema
            out_json = out_dir / "world.vxs.json"
            with open(out_json, "w") as f:
                json.dump(serial, f)

            response = {
                "status": "ok",
                "message": "Voxel grid generated",
                "paths": {
                    "world_json": str(out_json),
                },
                "meta": serial,
            }
            return jsonify(response)

        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 400

    return app


if __name__ == "__main__":
    # Allow overriding host/port via env vars
    host = os.environ.get("OSM_VOXEL_HOST", "127.0.0.1")
    port = int(os.environ.get("OSM_VOXEL_PORT", 5000))
    app = create_app()
    # Debug can be toggled via FLASK_DEBUG or OSM_VOXEL_DEBUG
    debug = os.environ.get("OSM_VOXEL_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)
