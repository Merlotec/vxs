#!/usr/bin/env python3
"""
Test script to generate simple terrain and send to renderer

This tests the complete pipeline:
Python (this script) → TCP → Proxy → WebSocket → WASM Renderer

Usage:
1. Start the proxy: python proxy.py
2. Run this script: python test_terrain.py
3. Open browser with WASM renderer at http://localhost:8000
"""

import sys
import os
import time

# Add parent directory to path to import voxelsim
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import voxelsim as vxs

def create_terrain():
    """Create a simple terrain with floor, walls, and some features"""
    world = vxs.VoxelGrid()

    # Create a 10x10 ground floor
    print("Creating ground floor...")
    for x in range(-5, 6):
        for z in range(-5, 6):
            world.set(vxs.Coord(x, 0, z), vxs.Cell.GROUND | vxs.Cell.FILLED)

    # Create walls around the perimeter
    print("Creating walls...")
    for y in range(1, 4):
        for x in range(-5, 6):
            # Front and back walls
            world.set(vxs.Coord(x, y, -5), vxs.Cell.FILLED)
            world.set(vxs.Coord(x, y, 5), vxs.Cell.FILLED)

        for z in range(-5, 6):
            # Left and right walls
            world.set(vxs.Coord(-5, y, z), vxs.Cell.FILLED)
            world.set(vxs.Coord(5, y, z), vxs.Cell.FILLED)

    # Add some target blocks inside
    print("Adding target blocks...")
    world.set(vxs.Coord(0, 1, 0), vxs.Cell.TARGET | vxs.Cell.FILLED)
    world.set(vxs.Coord(-2, 1, -2), vxs.Cell.TARGET | vxs.Cell.FILLED)
    world.set(vxs.Coord(2, 1, -2), vxs.Cell.TARGET | vxs.Cell.FILLED)
    world.set(vxs.Coord(-2, 1, 2), vxs.Cell.TARGET | vxs.Cell.FILLED)
    world.set(vxs.Coord(2, 1, 2), vxs.Cell.TARGET | vxs.Cell.FILLED)

    # Add a tower in one corner
    print("Adding tower...")
    for y in range(1, 6):
        world.set(vxs.Coord(3, y, 3), vxs.Cell.FILLED)

    # Add some sparse obstacles
    print("Adding obstacles...")
    world.set(vxs.Coord(-3, 1, 0), vxs.Cell.FILLED)
    world.set(vxs.Coord(3, 1, 0), vxs.Cell.FILLED)
    world.set(vxs.Coord(0, 1, -3), vxs.Cell.FILLED)
    world.set(vxs.Coord(0, 1, 3), vxs.Cell.FILLED)

    return world

def main():
    print("=" * 60)
    print("VoxelSim Terrain Test")
    print("=" * 60)

    # Create renderer client (connects to TCP ports)
    print("\nCreating RendererClient (TCP)...")
    try:
        client = vxs.RendererClient.default_localhost_py(num_pov_streams=0)
        print("✓ Connected to TCP ports 8080 (world)")
    except Exception as e:
        print(f"✗ Failed to create RendererClient: {e}")
        print("\nMake sure the proxy is running: python proxy.py")
        return

    # Create terrain
    print("\nGenerating terrain...")
    world = create_terrain()
    cell_count = len(list(world.cells()))
    print(f"✓ Created world with {cell_count} cells")

    # Send terrain continuously
    print("\nSending terrain to renderer...")
    print("(Press Ctrl+C to stop)")
    print("-" * 60)

    frame_count = 0
    try:
        while True:
            # Send the world
            client.send_world(world)
            frame_count += 1

            if frame_count % 10 == 0:
                print(f"Sent {frame_count} frames...")

            # Send at ~30 FPS
            time.sleep(1.0 / 30.0)

    except KeyboardInterrupt:
        print(f"\n✓ Sent {frame_count} frames total")
        print("Shutting down...")

if __name__ == "__main__":
    main()
