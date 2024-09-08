#!/bin/bash

# Check if all required arguments are provided
if [ $# -lt 4 ]; then
    echo "Usage: $0 <vit> <starting_size> <precision> <gpu>"
    echo "  vit: 'vitl' or 'vitg'"
    echo "  starting_size: integer value for the starting size"
    echo "  precision: 'tf32' or 'fp16'"
    echo "  gpu: GPU model (e.g., '4090')"
    exit 1
fi

vit=$1
size=$2
precision=$3
gpu=$4

# Validate the vit argument
if [ "$vit" != "vitl" ] && [ "$vit" != "vitg" ]; then
    echo "Invalid vit argument. Please use either 'vitl' or 'vitg'"
    exit 1
fi

# Validate the starting size argument
if ! echo "$size" | grep -q '^[0-9]\+$'; then
    echo "Invalid starting size. Please provide a positive integer."
    exit 1
fi

# Validate the precision argument
if [ "$precision" != "tf32" ] && [ "$precision" != "fp16" ]; then
    echo "Invalid precision argument. Please use either 'tf32' or 'fp16'"
    exit 1
fi

# Number of iterations
iterations=40

for i in $(seq 1 $iterations); do
    echo "Running iteration $i with size $size"

    mkdir depth_anything_v2_${vit}_${gpu}_${size}_${precision}

    # Export ONNX
    PYTHONPATH=. python tools/export_onnx.py --checkpoint /app/depth-anything-tensorrt/third_party/depth_anything_v2/depth_anything_v2/checkpoints/depth_anything_v2_${vit}.pth --onnx depth_anything_v2_${vit}_${gpu}_${size}_${precision}/depth_anything_v2_${vit}_${gpu}_${size}_${precision}.onnx --input_size ${size} --encoder ${vit} 

    # Convert ONNX to TRT
    PYTHONPATH=. python trt_build_engine.py --onnx depth_anything_v2_${vit}_${gpu}_${size}_${precision}/depth_anything_v2_${vit}_${gpu}_${size}_${precision}.onnx --engine depth_anything_v2_${vit}_${gpu}_${size}_${precision}.trt --${precision} --workspace 20

    # Run inference
    git pull
    python infer_video.py --video test.mp4 --engine depth_anything_v2_${vit}_${gpu}_${size}_${precision}.trt --size ${size} --output ${vit}_${gpu}_test_single_${size}_${precision}.mp4 --method single --use_gpu --sample_rate 10

    # Delete the .trt file
    # rm depth_anything_v2_${vit}_${gpu}_${size}.trt
    # echo "Deleted depth_anything_v2_${vit}_${gpu}_${size}.trt"
    rm -r depth_anything_v2_${vit}_${gpu}_${size}_${precision}
    echo "Deleted depth_anything_v2_${vit}_${gpu}_${size}_${precision}"
    # Increment size by 14
    size=$((size + 14))
done