#!/bin/bash

# Check if both vit and starting size arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <vit> <starting_size>"
    echo "  vit: 'vitl' or 'vitg'"
    echo "  starting_size: integer value for the starting size"
    exit 1
fi

vit=$1
size=$2

# Validate the vit argument
if [ "$vit" != "vitl" ] && [ "$vit" != "vitg" ]; then
    echo "Invalid vit argument. Please use either 'vitl' or 'vitg'"
    exit 1
fi

# Validate the starting size argument
if ! [[ "$size" =~ ^[0-9]+$ ]]; then
    echo "Invalid starting size. Please provide a positive integer."
    exit 1
fi

# Number of iterations
iterations=40

for i in $(seq 1 $iterations); do
    echo "Running iteration $i with size $size"

    mkdir depth_anything_v2_${vit}_4090_${size}

    # Export ONNX
    PYTHONPATH=. python tools/export_onnx.py --checkpoint /app/myrepo/depth-anything-tensorrt/third_party/depth_anything_v2/depth_anything_v2/checkpoints/depth_anything_v2_${vit}.pth --onnx depth_anything_v2_${vit}_4090_${size}/depth_anything_v2_${vit}_4090_${size}.onnx --input_size ${size} --encoder ${vit} 

    # Convert ONNX to TRT
    PYTHONPATH=. python trt_build_engine.py --onnx depth_anything_v2_${vit}_4090_${size}/depth_anything_v2_${vit}_4090_${size}.onnx --engine depth_anything_v2_${vit}_4090_${size}.trt --tf32 --workspace 20

    # Run inference
    git pull
    python infer_video.py --video flowers.mov --engine depth_anything_v2_${vit}_4090_${size}.trt --size ${size} --output ${vit}_flowers_single_${size}_04.mp4 --method single --use_gpu --sample_rate 10

    # Delete the .trt file
    # rm depth_anything_v2_${vit}_4090_${size}.trt
    # echo "Deleted depth_anything_v2_${vit}_4090_${size}.trt"
    rm -r depth_anything_v2_${vit}_4090_${size}
    rm depth_anything_v2_${vit}_4090_${size}.onnx
    echo "Deleted depth_anything_v2_${vit}_4090_${size}.onnx"
    # Increment size by 14
    size=$((size + 14))
done