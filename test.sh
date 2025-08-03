cd voxelsim-py
maturin build --release 
cd ../python
pip install /Users/brodieknight/dev/drone/vxs/voxelsim-py/target/wheels/voxelsim_py-0.1.0-cp313-cp313-macosx_11_0_arm64.whl --force-reinstall
python3 povtest.py
