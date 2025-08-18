import json
import sys
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict

import webview

# Support running directly from this folder by adding the parent to sys.path
try:
    from osm_voxel_app.osm2voxel import generate_voxel_grid_from_point
except Exception:
    parent = Path(__file__).resolve().parent.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    from osm_voxel_app.osm2voxel import generate_voxel_grid_from_point


class API:
    def __init__(self):
        self.last_path: Path | None = None

    def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            lat = float(params.get("lat"))
            lon = float(params.get("lon"))
            nx = int(params.get("x"))
            ny = int(params.get("y"))
            nz = int(params.get("z"))
            m = float(params.get("meters_per_voxel", 1.0))
            backend = (params.get("backend") or "python").lower()

            out_dir = Path(params.get("out_dir") or Path.cwd() / "runs" / "osm_voxel")
            out_dir.mkdir(parents=True, exist_ok=True)

            vg, serial = generate_voxel_grid_from_point(
                lat=lat, lon=lon, dims=(nx, ny, nz), meters_per_voxel=m, work_dir=out_dir, backend=backend
            )

            out_json = out_dir / "world.vxs.json"
            with open(out_json, "w") as f:
                json.dump(serial, f)

            self.last_path = out_json

            # Launch povtest.py with the saved world JSON
            try:
                repo_root = Path(__file__).resolve().parent.parent.parent
                pov_path = (repo_root / "povtest.py").resolve()
                if pov_path.exists():
                    print("Opening file in default renderer.")
                    subprocess.Popen([sys.executable, str(pov_path), str(out_json)], cwd=str(repo_root))
            except Exception as e:
                # Non-fatal: still return success
                pass

            return {
                "status": "ok",
                "world_json": str(out_json),
                "meta": serial,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}


def run():
    api = API()
    html_path = (Path(__file__).parent / "desktop_index.html").resolve()
    window = webview.create_window("OSM → Voxel World", url=str(html_path), js_api=api)
    webview.start(debug=False, http_server=False, func=None, gui=None)


if __name__ == "__main__":
    run()
