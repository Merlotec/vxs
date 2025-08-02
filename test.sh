rm -r cuda_octree/build
mkdir cuda_octree/build
cd cuda_octree/build
cmake ..
cmake --build .
cd ../../voxelsim-py
cargo clean -p octree-gpu
maturin build --release --features cuda-octree
cd ../python
pip install /home/box/dev/drone/vxs/voxelsim-py/target/wheels/voxelsim_py-0.1.0-cp312-cp312-manylinux_2_35_x86_64.whl --force-reinstall
python3 test.py
