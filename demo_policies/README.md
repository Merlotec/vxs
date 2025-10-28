# Demo Policies for Investor Video

These 4 policies demonstrate "LLM evolution" for the investor demo video. They are **hardcoded** to guarantee reliable behavior.

## The 4 Iterations

### Iteration 0: `iter0_crash.py`
- **Behavior**: Flies forward, no planning, crashes immediately
- **Expected result**: ~50 steps, many collisions, 0% success
- **Demo narrative**: "Total failure"

### Iteration 1: `iter1_first_turbine.py`
- **Behavior**: Uses A* to reach first turbine at [30, 30, -50], then stops
- **Expected result**: Reaches turbine 1, idles, ~200 steps
- **Demo narrative**: "Learned planning, partial success"

### Iteration 2: `iter2_all_turbines.py`
- **Behavior**: Visits all 3 turbines but doesn't return home
- **Expected result**: All 3 turbines visited, stops at turbine 3
- **Demo narrative**: "Waypoint navigation working, forgot to return"

### Iteration 3: `iter3_perfect.py`
- **Behavior**: Visits all 3 turbines AND returns to [100, 100, -40]
- **Expected result**: 100% mission success
- **Demo narrative**: "Perfect execution"

## How to Test

### Quick test (no rendering):
```bash
./demo_policies/run_demo.sh
```

### Test individual policy with rendering:
```bash
# Start renderer first
cargo run -p voxelsim-renderer --release

# In another terminal, run policy
python -m bgen.runner.run_policy \
  --policy demo_policies/iter3_perfect.py \
  --use-wind-farm \
  --render \
  --episodes 1
```

### Test with browser renderer (WASM):
```bash
# Terminal 1: Start proxy
cd proxy && npm start

# Terminal 2: Start frontend
cd voxelsim-frontend && npm run dev

# Terminal 3: Run policy
python -m bgen.runner.run_policy \
  --policy demo_policies/iter3_perfect.py \
  --use-wind-farm \
  --render \
  --episodes 1
```

## Recording the Demo Video

1. **Start renderer** (native or browser)
2. **Set up screen capture** (OBS Studio, QuickTime, etc.)
3. **Run each policy one by one** while recording
4. **In video editing**:
   - Add "Iteration X" titles
   - Add fake metrics overlays
   - Add "LLM generating code" scenes between iterations
   - Speed up iterations 0-2 (2x), show iteration 3 at real-time

## Troubleshooting

**Q: Agent doesn't move?**
- Check that `--use-wind-farm` flag is set (creates proper terrain)
- Verify voxelsim module is installed: `python -c "import voxelsim"`

**Q: Crashes with "list indices must be integers"?**
- This shouldn't happen - all coordinates are hardcoded as integers

**Q: Agent gets stuck?**
- A* may fail if turbine position is inside terrain
- Adjust turbine heights if needed (currently -50)

**Q: Want to change turbine positions?**
- Edit the `TURBINES` list in each policy file
- Use positions from your actual wind farm terrain

## Wind Farm Turbine Positions

Based on `wind_turbine_inspection.py`, turbines are at:
- Turbine 1: (30, 30) - bottom-left
- Turbine 2: (120, 120) - center
- Turbine 3: (210, 210) - top-right

Altitude: -50 (50 voxels below origin)
Start position: [100, 100, -40]
