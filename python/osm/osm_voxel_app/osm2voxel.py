import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
import time
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import requests
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


def meters_to_degrees(lat_deg: float, dx_m: float, dy_m: float) -> Tuple[float, float]:
    lat_rad = math.radians(lat_deg)
    dlat = dy_m / 111_320.0
    dlon = dx_m / (111_320.0 * max(math.cos(lat_rad), 1e-6))
    return dlat, dlon


def bbox_around(lat: float, lon: float, width_m: float, height_m: float) -> Tuple[float, float, float, float]:
    """Return bbox (south, west, north, east) in degrees around center lat/lon."""
    dlat, dlon = meters_to_degrees(lat, width_m / 2.0, height_m / 2.0)
    south = lat - dlat
    north = lat + dlat
    west = lon - dlon
    east = lon + dlon
    return south, west, north, east


DEFAULT_OVERPASS_ENDPOINTS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]


def post_overpass_query(query: str, endpoints: Optional[List[str]] = None, timeout: int = 60) -> requests.Response:
    endpoints = endpoints or DEFAULT_OVERPASS_ENDPOINTS
    errors = []
    for url in endpoints:
        try:
            resp = requests.post(url, data={"data": query}, timeout=timeout)
            if resp.status_code == 200:
                return resp
            errors.append(f"{url} -> HTTP {resp.status_code}")
            # Small delay before trying next endpoint
            time.sleep(0.5)
        except Exception as e:
            errors.append(f"{url} -> {e}")
            time.sleep(0.5)
    raise RuntimeError("All Overpass endpoints failed: " + "; ".join(errors))


def fetch_osm_xml_for_bbox(bbox: Tuple[float, float, float, float], out_file: Path,
                           overpass_endpoints: Optional[List[str]] = None) -> None:
    south, west, north, east = bbox
    query = f"""
    [out:xml][timeout:60];
    (
      node({south},{west},{north},{east});
      way({south},{west},{north},{east});
      relation({south},{west},{north},{east});
    );
    out body;
    >;
    out skel qt;
    """
    resp = post_overpass_query(query, endpoints=overpass_endpoints)
    out_file.write_bytes(resp.content)


def fetch_buildings_geojson(
    bbox: Tuple[float, float, float, float],
    overpass_endpoints: Optional[List[str]] = None,
) -> Dict[str, Any]:
    south, west, north, east = bbox
    query = f"""
    [out:json][timeout:60];
    (
      way["building"]({south},{west},{north},{east});
      relation["building"]({south},{west},{north},{east});
    );
    (._;>;);
    out body;
    """
    resp = post_overpass_query(query, endpoints=overpass_endpoints)
    return resp.json()


def local_xy_from_latlon(lat0: float, lon0: float, lat: float, lon: float) -> Tuple[float, float]:
    lat_rad = math.radians(lat0)
    dx = (lon - lon0) * 111_320.0 * math.cos(lat_rad)
    dy = (lat - lat0) * 111_320.0
    return dx, dy


def parse_height(tags: Dict[str, Any]) -> float:
    # Prefer explicit height, then building:levels
    h = tags.get("height") if tags else None
    if isinstance(h, str):
        try:
            h = float(h.split()[0])
        except Exception:
            h = None
    if isinstance(h, (int, float)):
        return float(h)
    levels = tags.get("building:levels") if tags else None
    try:
        lv = float(levels)
        return lv * 3.0  # meters per level heuristic
    except Exception:
        return 10.0  # default height


def buildings_mesh_from_overpass(
    lat: float, lon: float, bbox: Tuple[float, float, float, float], data: Dict[str, Any]
):
    import trimesh
    elements = data.get("elements", [])
    nodes = {el["id"]: (el["lat"], el["lon"]) for el in elements if el.get("type") == "node"}

    meshes: List[trimesh.Trimesh] = []
    for el in elements:
        t = el.get("type")
        if t not in ("way",):
            continue
        if not el.get("tags", {}).get("building"):
            continue
        nds = el.get("nodes") or []
        if len(nds) < 3:
            continue
        coords_latlon = []
        ok = True
        for nid in nds:
            n = nodes.get(nid)
            if not n:
                ok = False
                break
            coords_latlon.append(n)
        if not ok:
            continue
        # Ensure closed polygon
        if coords_latlon[0] != coords_latlon[-1]:
            coords_latlon.append(coords_latlon[0])

        # Convert to local XY meters
        xy = [local_xy_from_latlon(lat, lon, la, lo) for la, lo in coords_latlon]
        try:
            poly = Polygon(xy)
        except Exception:
            continue
        if not poly.is_valid or poly.area <= 0:
            continue
        height = parse_height(el.get("tags", {}))
        try:
            mesh = trimesh.creation.extrude_polygon(poly, height)
            meshes.append(mesh)
        except Exception:
            continue

    if not meshes:
        return None
    scene = trimesh.util.concatenate(meshes)
    return scene


def voxelize_buildings_python(lat: float, lon: float, dims: Tuple[int, int, int], meters_per_voxel: float,
                              overpass_endpoints: Optional[List[str]] = None):
    import trimesh
    nx, ny, nz = dims
    width_m = nx * meters_per_voxel
    height_m = ny * meters_per_voxel
    bbox = bbox_around(lat, lon, width_m, height_m)
    # Try single request first
    try:
        data = fetch_buildings_geojson(bbox, overpass_endpoints=overpass_endpoints)
        mesh = buildings_mesh_from_overpass(lat, lon, bbox, data)
        if mesh is None:
            dense = np.zeros((nx, ny, nz), dtype=bool)
            return dense, bbox
    except Exception:
        # Fallback: subdivide bbox into 2x2 and merge meshes
        south, west, north, east = bbox
        mid_lat = (south + north) / 2.0
        mid_lon = (west + east) / 2.0
        tiles = [
            (south, west, mid_lat, mid_lon),
            (south, mid_lon, mid_lat, east),
            (mid_lat, west, north, mid_lon),
            (mid_lat, mid_lon, north, east),
        ]
        meshes = []
        for tb in tiles:
            try:
                d = fetch_buildings_geojson(tb, overpass_endpoints=overpass_endpoints)
                m = buildings_mesh_from_overpass(lat, lon, tb, d)
                if m is not None:
                    meshes.append(m)
            except Exception:
                continue
        if not meshes:
            dense = np.zeros((nx, ny, nz), dtype=bool)
            return dense, bbox
        mesh = trimesh.util.concatenate(meshes)
    vox = mesh.voxelized(pitch=meters_per_voxel)
    vox = vox.fill()
    dense_buildings = vox.matrix
    # Return raw building occupancy; caller merges with terrain / crops
    return dense_buildings, bbox

def run_osm2world(osm_file: Path, out_obj: Path) -> None:
    jar = os.environ.get("OSM2WORLD_JAR")
    if not jar:
        raise RuntimeError("OSM2WORLD_JAR env var not set; point it to OSM2World.jar")
    if not Path(jar).exists():
        raise RuntimeError(f"OSM2WORLD_JAR not found: {jar}")
    # Try running headless to avoid GUI issues on macOS
    base_cmd = [
        "java", "-Djava.awt.headless=true", "-jar", jar,
        "-i", str(osm_file),
        "-o", str(out_obj),
    ]
    # First attempt
    proc = subprocess.run(base_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # Attempt a fallback without headless flag
        alt_cmd = ["java", "-jar", jar, "-i", str(osm_file), "-o", str(out_obj)]
        proc2 = subprocess.run(alt_cmd, capture_output=True, text=True)
        if proc2.returncode != 0:
            # Write logs alongside output for debugging
            log_path = out_obj.with_suffix(".osm2world.log")
            log_path.write_text(
                "Command 1:\n" + " ".join(base_cmd) + "\n\n" +
                "stdout 1:\n" + (proc.stdout or '') + "\n\n" +
                "stderr 1:\n" + (proc.stderr or '') + "\n\n" +
                "Command 2:\n" + " ".join(alt_cmd) + "\n\n" +
                "stdout 2:\n" + (proc2.stdout or '') + "\n\n" +
                "stderr 2:\n" + (proc2.stderr or '') + "\n"
            )
            raise RuntimeError(
                f"OSM2World failed (see {log_path}). Last stderr: {proc2.stderr.strip() or proc.stderr.strip()}"
            )


def load_mesh_as_trimesh(obj_path: Path):
    import trimesh
    mesh = trimesh.load_mesh(str(obj_path))
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("Failed to load mesh as Trimesh")
    return mesh


def voxelize_mesh(mesh, pitch: float):
    # Voxelize and fill interior for solid representation
    vox = mesh.voxelized(pitch=pitch)
    vox = vox.fill()
    # dense boolean occupancy and origin transform
    mat = vox.matrix  # (nx, ny, nz) bool
    origin = vox.origin  # world coordinates of matrix[0,0,0] corner
    return mat, origin


def dense_to_coords(dense: np.ndarray) -> np.ndarray:
    # Return N x 3 integer coordinates for True entries
    coords = np.argwhere(dense)
    # coords already in (x,y,z) index order
    return coords.astype(np.int32)

def _enu_to_ned_coords(coords: np.ndarray) -> np.ndarray:
    """Map voxel indices from ENU (x=east, y=north, z=up) to NED (x=north, y=east, z=down).
    In NED, up is negative, so we negate z-indices. We also swap x/y.
    """
    if coords.size == 0:
        return coords
    mapped = np.empty_like(coords)
    mapped[:, 0] = coords[:, 1]       # x_north = y_north
    mapped[:, 1] = coords[:, 0]       # y_east  = x_east
    mapped[:, 2] = -coords[:, 2]      # z_down  = -z_up
    return mapped


def coords_to_vxs(coords: np.ndarray):
    import voxelsim as vxs
    coords = _enu_to_ned_coords(coords)
    # Build sparse dict of filled cells
    cells = {}
    filled = vxs.Cell.filled()
    for x, y, z in coords:
        cells[(int(x), int(y), int(z))] = filled
    if hasattr(vxs.VoxelGrid, "from_dict_py"):
        vg = vxs.VoxelGrid.from_dict_py(cells)
    else:
        vg = vxs.VoxelGrid()
        for k, cell in cells.items():
            vg.set_cell_py(k, cell)
    return vg


def coords_to_json_list(coords: np.ndarray) -> list:
    # Represent cells as [x, y, z, 1] where 1 indicates a filled voxel
    coords = _enu_to_ned_coords(coords)
    return [[int(x), int(y), int(z), 1] for x, y, z in coords]


def crop_or_pad(dense: np.ndarray, target: Tuple[int, int, int]) -> np.ndarray:
    """Center-crop or pad the dense grid to target shape (nx, ny, nz)."""
    out = np.zeros(target, dtype=bool)
    src = dense
    sx, sy, sz = src.shape
    tx, ty, tz = target

    # Compute start indices to center align
    ox = max((sx - tx) // 2, 0)
    oy = max((sy - ty) // 2, 0)
    oz = max((sz - tz) // 2, 0)

    dx = max((tx - sx) // 2, 0)
    dy = max((ty - sy) // 2, 0)
    dz = max((tz - sz) // 2, 0)

    cx = min(sx, tx)
    cy = min(sy, ty)
    cz = min(sz, tz)

    out[dx:dx+cx, dy:dy+cy, dz:dz+cz] = src[ox:ox+cx, oy:oy+cy, oz:oz+cz]
    return out


def crop_or_pad_bottom_aligned(dense: np.ndarray, target: Tuple[int, int, int]) -> np.ndarray:
    """XY centered, Z bottom-aligned crop/pad to target (nx, ny, nz)."""
    out = np.zeros(target, dtype=bool)
    src = dense
    sx, sy, sz = src.shape
    tx, ty, tz = target

    # XY centered
    ox = max((sx - tx) // 2, 0)
    oy = max((sy - ty) // 2, 0)
    dx = max((tx - sx) // 2, 0)
    dy = max((ty - sy) // 2, 0)
    cx = min(sx, tx)
    cy = min(sy, ty)

    # Z bottom-aligned
    oz = 0
    dz = 0
    cz = min(sz, tz)

    out[dx:dx+cx, dy:dy+cy, dz:dz+cz] = src[ox:ox+cx, oy:oy+cy, oz:oz+cz]
    return out


def try_load_srtm():
    try:
        import srtm
        return srtm.get_data()
    except Exception:
        return None


def terrain_heights_dem(lat_center: float, lon_center: float, nx: int, ny: int, meters_per_voxel: float,
                        dem_stride: int = 2) -> Optional[np.ndarray]:
    data = try_load_srtm()
    if data is None:
        return None
    xs = np.arange(nx)
    ys = np.arange(ny)
    offset_x = (xs - nx // 2) * meters_per_voxel
    offset_y = (ys - ny // 2) * meters_per_voxel
    dlat_pm, dlon_pm = meters_to_degrees(lat_center, 1.0, 1.0)
    xs_coarse = xs[::dem_stride]
    ys_coarse = ys[::dem_stride]
    elev_coarse = np.zeros((xs_coarse.size, ys_coarse.size), dtype=float)
    for i, ix in enumerate(xs_coarse):
        for j, iy in enumerate(ys_coarse):
            lat = lat_center + offset_y[iy] * dlat_pm
            lon = lon_center + offset_x[ix] * dlon_pm
            h = data.get_elevation(lat, lon)
            if h is None:
                h = 0.0
            elev_coarse[i, j] = float(h)
    # Nearest-neighbor upsample using repeat
    elev = np.kron(elev_coarse, np.ones((dem_stride, dem_stride)))
    elev = elev[:nx, :ny]
    return elev


def merge_buildings_with_terrain(dense_buildings: np.ndarray, elev: Optional[np.ndarray],
                                 nx: int, ny: int, nz: int, meters_per_voxel: float) -> np.ndarray:
    out = np.zeros((nx, ny, nz), dtype=bool)
    if elev is not None:
        elev = elev - np.nanmin(elev)
        h_idx = np.clip((elev / meters_per_voxel).astype(int), 0, nz - 1)
        # Terrain fill and place buildings atop
        bx, by, bz = dense_buildings.shape
        bx = min(bx, nx); by = min(by, ny)
        for ix in range(nx):
            for iy in range(ny):
                hi = h_idx[ix, iy]
                out[ix, iy, :hi+1] = True
                if ix < bx and iy < by:
                    col = dense_buildings[ix, iy, :]
                    if col.any():
                        z_on = np.nonzero(col)[0] + hi
                        z_on = z_on[z_on < nz]
                        out[ix, iy, z_on] = True
        return out
    else:
        # Bottom-align buildings and add flat ground
        dense = dense_buildings
        if dense.any():
            occ = np.any(np.any(dense, axis=0), axis=0)
            z_idx = np.nonzero(occ)[0]
            if z_idx.size > 0:
                zmin = int(z_idx.min())
                if zmin > 0:
                    dense = dense[:, :, zmin:]
        dense = crop_or_pad_bottom_aligned(dense, (nx, ny, nz))
        out |= dense
        if out.size > 0:
            out[:, :, 0] = True
        return out


def generate_voxel_grid_from_point(lat: float, lon: float, dims: Tuple[int, int, int],
                                   meters_per_voxel: float, work_dir: Path,
                                   backend: str = "python") -> Tuple["Any", Dict[str, Any]]:
    """
    End-to-end: fetch OSM via Overpass for a bbox around (lat,lon),
    convert to OBJ via OSM2World, voxelize to target dims and return vxs.VoxelGrid.
    Returns (VoxelGrid, meta_dict).
    """
    nx, ny, nz = dims
    width_m = nx * meters_per_voxel
    height_m = ny * meters_per_voxel
    bbox = bbox_around(lat, lon, width_m, height_m)

    work_dir.mkdir(parents=True, exist_ok=True)

    if backend == "python":
        # Pure-Python path: Overpass buildings → extrude → voxelize
        endpoints_env = os.environ.get("OSM_OVERPASS_URLS")
        endpoints = None
        if endpoints_env:
            endpoints = [u.strip() for u in endpoints_env.split(",") if u.strip()]
        dense_buildings, bbox_used = voxelize_buildings_python(lat, lon, (nx, ny, nz), meters_per_voxel, overpass_endpoints=endpoints)
        # DEM-based terrain (optional, default on). Disable with OSM_USE_DEM=0
        use_dem = os.environ.get("OSM_USE_DEM", "1") != "0"
        elev = terrain_heights_dem(lat, lon, nx, ny, meters_per_voxel) if use_dem else None
        dense = merge_buildings_with_terrain(dense_buildings, elev, nx, ny, nz, meters_per_voxel)
        coords = dense_to_coords(dense)
        vg = coords_to_vxs(coords)
        serial = {
            "backend": backend,
            "center": {"lat": lat, "lon": lon},
            "bbox": {"south": bbox_used[0], "west": bbox_used[1], "north": bbox_used[2], "east": bbox_used[3]},
            "meters_per_voxel": meters_per_voxel,
            "dims": {"x": nx, "y": ny, "z": nz},
            "occupied_voxels": int(coords.shape[0]),
            "paths": {},
            "cells": coords_to_json_list(coords),
            "dem": bool(elev is not None),
        }
        return vg, serial
    else:
        # OSM2World Java path
        osm_path = work_dir / "region.osm"
        obj_path = work_dir / "region.obj"
        fetch_osm_xml_for_bbox(bbox, osm_path)
        run_osm2world(osm_path, obj_path)
        mesh = load_mesh_as_trimesh(obj_path)
        dense_buildings, origin = voxelize_mesh(mesh, pitch=meters_per_voxel)
        use_dem = os.environ.get("OSM_USE_DEM", "1") != "0"
        elev = terrain_heights_dem(lat, lon, nx, ny, meters_per_voxel) if use_dem else None
        dense = merge_buildings_with_terrain(dense_buildings, elev, nx, ny, nz, meters_per_voxel)
        coords = dense_to_coords(dense)
        vg = coords_to_vxs(coords)
        serial = {
            "backend": backend,
            "center": {"lat": lat, "lon": lon},
            "bbox": {"south": bbox[0], "west": bbox[1], "north": bbox[2], "east": bbox[3]},
            "meters_per_voxel": meters_per_voxel,
            "dims": {"x": nx, "y": ny, "z": nz},
            "origin_world": list(map(float, np.asarray(origin).tolist())),
            "occupied_voxels": int(coords.shape[0]),
            "paths": {"osm": str(osm_path), "obj": str(obj_path)},
            "cells": coords_to_json_list(coords),
            "dem": bool(elev is not None),
        }
        return vg, serial
