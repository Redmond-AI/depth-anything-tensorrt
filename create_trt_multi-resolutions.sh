#!/bin/bash
rm log.txt
# Check if all required arguments are provided
if [ $# -lt 5 ]; then
    echo -e "\e[31mUsage: $0 <vit> <starting_size> <precision> <gpu> <workspace>\e[0m" | tee -a log.txt
    echo -e "\e[31m  vit: 'vitl' or 'vitg'\e[0m" | tee -a log.txt
    echo -e "\e[31m  starting_size: integer value for the starting size\e[0m" | tee -a log.txt
    echo -e "\e[31m  precision: 'tf32', 'fp16', or 'fp32'\e[0m" | tee -a log.txt
    echo -e "\e[31m  gpu: GPU model (e.g., '4090')\e[0m" | tee -a log.txt
    echo -e "\e[31m  workspace: workspace size in GB (e.g., '20')\e[0m" | tee -a log.txt
    exit 1
fi

vit=$1
size=$2
precision=$3
gpu=$4
workspace=$5

# Validate the vit argument
if [ "$vit" != "vitl" ] && [ "$vit" != "vitg" ]; then
    echo -e "\e[31mInvalid vit argument. Please use either 'vitl' or 'vitg'\e[0m" | tee -a log.txt
    exit 1
fi

# Validate the starting size argument
if ! echo "$size" | grep -q '^[0-9]\+$'; then
    echo -e "\e[31mInvalid starting size. Please provide a positive integer.\e[0m" | tee -a log.txt
    exit 1
fi

# Validate the precision argument
if [ "$precision" != "tf32" ] && [ "$precision" != "fp16" ] && [ "$precision" != "fp32" ]; then
    echo -e "\e[31mInvalid precision argument. Please use 'tf32', 'fp16', or 'fp32'\e[0m" | tee -a log.txt
    exit 1
fi

# Validate the workspace argument
if ! echo "$workspace" | grep -q '^[0-9]\+$'; then
    echo -e "\e[31mInvalid workspace size. Please provide a positive integer.\e[0m" | tee -a log.txt
    exit 1
fi

# Number of iterations
iterations=40

for i in $(seq 1 $iterations); do
    echo -e "\e[31mRunning iteration $i with size $size\e[0m" | tee -a log.txt

    echo -e "\e[31mmkdir depth_anything_v2_${vit}_${gpu}_${size}_${precision}\e[0m" | tee -a log.txt
    mkdir depth_anything_v2_${vit}_${gpu}_${size}_${precision} 2>&1 | tee -a log.txt

    # Export ONNX
    echo -e "\e[31mPYTHONPATH=. python tools/export_onnx.py --checkpoint /app/depth-anything-tensorrt/third_party/depth_anything_v2/depth_anything_v2/checkpoints/depth_anything_v2_${vit}.pth --onnx depth_anything_v2_${vit}_${gpu}_${size}_${precision}/depth_anything_v2_${vit}_${gpu}_${size}_${precision}.onnx --input_size ${size} --encoder ${vit}\e[0m" | tee -a log.txt
    PYTHONPATH=. python tools/export_onnx.py --checkpoint /app/depth-anything-tensorrt/third_party/depth_anything_v2/depth_anything_v2/checkpoints/depth_anything_v2_${vit}.pth --onnx depth_anything_v2_${vit}_${gpu}_${size}_${precision}/depth_anything_v2_${vit}_${gpu}_${size}_${precision}.onnx --input_size ${size} --encoder ${vit} 2>&1 | tee -a log.txt

    # Convert ONNX to TRT
    echo -e "\e[31mPYTHONPATH=. python trt_build_engine.py --onnx depth_anything_v2_${vit}_${gpu}_${size}_${precision}/depth_anything_v2_${vit}_${gpu}_${size}_${precision}.onnx --engine depth_anything_v2_${vit}_${gpu}_${size}_${precision}.trt --${precision} --workspace ${workspace}\e[0m" | tee -a log.txt
    PYTHONPATH=. python trt_build_engine.py --onnx depth_anything_v2_${vit}_${gpu}_${size}_${precision}/depth_anything_v2_${vit}_${gpu}_${size}_${precision}.onnx --engine depth_anything_v2_${vit}_${gpu}_${size}_${precision}.trt --precision ${precision} --workspace ${workspace} 2>&1 | tee -a log.txt

    # Run inference
    echo -e "\e[31mgit pull\e[0m" | tee -a log.txt
    git pull 2>&1 | tee -a log.txt
    echo -e "\e[31mpython infer_video.py --video test.mp4 --engine depth_anything_v2_${vit}_${gpu}_${size}_${precision}.trt --size ${size} --output ${vit}_${gpu}_test_single_${size}_${precision}.mp4 --method single --use_gpu --sample_rate 10\e[0m" | tee -a log.txt
    python infer_video.py --video flowers.mov --engine depth_anything_v2_${vit}_${gpu}_${size}_${precision}.trt --size ${size} --output ${vit}_${gpu}_flowers_single_${size}_${precision}.mp4 --method single --use_gpu --sample_rate 10 2>&1 | tee -a video_log.txt

    echo -e "\e[31mrm -r depth_anything_v2_${vit}_${gpu}_${size}_${precision}\e[0m" | tee -a log.txt
    rm -r depth_anything_v2_${vit}_${gpu}_${size}_${precision} 2>&1 | tee -a log.txt
    echo -e "\e[31mDeleted depth_anything_v2_${vit}_${gpu}_${size}_${precision}\e[0m" | tee -a log.txt
    # Increment size by 14
    echo -e "\e[31msize=$((size + 14))\e[0m" | tee -a log.txt
    size=$((size - 14))
done