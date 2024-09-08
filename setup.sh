git submodule init && git submodule update
cd tools
cp ../third_party/depth_anything_v2/depth_anything_v2/dpt.py ../third_party/depth_anything_v2/depth_anything_v2/dpt_old.py
cp dpt.py ../third_party/depth_anything_v2/depth_anything_v2/dpt.py
cd ..
mkdir third_party/depth_anything_v2/depth_anything_v2/checkpoints
wget https://pub-da6ae3cf12bc4de49e659943f4080da6.r2.dev/da_v2_checkpoints/depth_anything_v2_vitg.pth -O third_party/depth_anything_v2/depth_anything_v2/checkpoints/depth_anything_v2_vitg.pth
wget https://pub-da6ae3cf12bc4de49e659943f4080da6.r2.dev/da_v2_checkpoints/depth_anything_v2_vitl.pth -O third_party/depth_anything_v2/depth_anything_v2/checkpoints/depth_anything_v2_vitl.pth

pip uninstall onnx
python3 -m pip uninstall tensorrt tensorrt-cu12 tensorrt-cu12-bindings tensorrt-cu12-libs
sudo apt-get purge "libnvinfer*"
sudo apt-get purge "nv-tensorrt-local-repo*"
python3 -m pip install tensorrt
pip install onnx