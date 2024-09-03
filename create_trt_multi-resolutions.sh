#!/bin/bash

# Starting size
size=602

# Number of iterations
iterations=15

for i in $(seq 1 $iterations); do
    echo "Running iteration $i with size $size"

    mkdir depth_anything_v2_vitl_4090_${size}

    # Export ONNX
    PYTHONPATH=. python tools/export_onnx.py --checkpoint /app/myrepo/depth-anything-tensorrt/third_party/depth_anything_v2/depth_anything_v2/checkpoints/depth_anything_v2_vitl.pth --onnx depth_anything_v2_vitl_4090_${size}/depth_anything_v2_vitl_4090_${size}.onnx --input_size ${size} --encoder vitl 

    # Convert ONNX to TRT
    PYTHONPATH=. python trt_build_engine.py --onnx depth_anything_v2_vitl_4090_${size}/depth_anything_v2_vitl_4090_${size}.onnx --engine depth_anything_v2_vitl_4090_${size}.trt --fp16 --workspace 20

    # Run inference
    git pull
    python infer_video.py --video flowers.mov --engine depth_anything_v2_vitl_4090_${size}.trt --size ${size} --output vitl_flowers_single_${size}_04.mp4 --method single --use_gpu --sample_rate 10

    # Delete the .trt file
    rm depth_anything_v2_vitl_4090_${size}.trt
    echo "Deleted depth_anything_v2_vitl_4090_${size}.trt"
    rm -r depth_anything_v2_vitl_4090_${size}
    rm depth_anything_v2_vitl_4090_${size}.onnx
    echo "Deleted depth_anything_v2_vitl_4090_${size}.onnx"
    # Increment size by 14
    size=$((size + 14))
done