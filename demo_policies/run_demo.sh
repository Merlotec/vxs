#!/bin/bash
# Run all 4 demo policies in sequence

echo "========================================="
echo "DEMO POLICY TEST SUITE"
echo "========================================="

# Policy 0: Crash
echo ""
echo ">>> ITERATION 0: Crash Policy <<<"
python -m bgen.runner.run_policy \
  --policy demo_policies/iter0_crash.py \
  --use-wind-farm \
  --episodes 1 \
  --outdir runs/demo_iter0 \
  --target 30,30,-50

echo ""
echo ">>> ITERATION 1: First Turbine Only <<<"
python -m bgen.runner.run_policy \
  --policy demo_policies/iter1_first_turbine.py \
  --use-wind-farm \
  --episodes 1 \
  --outdir runs/demo_iter1 \
  --target 30,30,-50

echo ""
echo ">>> ITERATION 2: All Turbines (No Return) <<<"
python -m bgen.runner.run_policy \
  --policy demo_policies/iter2_all_turbines.py \
  --use-wind-farm \
  --episodes 1 \
  --outdir runs/demo_iter2 \
  --target 210,210,-50

echo ""
echo ">>> ITERATION 3: Perfect Mission <<<"
python -m bgen.runner.run_policy \
  --policy demo_policies/iter3_perfect.py \
  --use-wind-farm \
  --episodes 1 \
  --outdir runs/demo_iter3 \
  --target 100,100,-40

echo ""
echo "========================================="
echo "ALL TESTS COMPLETE"
echo "========================================="
echo ""
echo "Check results in runs/demo_iter*/"
