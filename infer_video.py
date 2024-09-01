import cv2
import numpy as np
from PIL import Image
import torch
import argparse
from dpt.dpt import DptTrtInference
import time

def process_frame(frame, dpt, size):
    original_size = frame.shape[:2][::-1]  # (width, height)
    
    # Resize and preprocess the frame
    frame_resized = cv2.resize(frame, (size, size))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_input = frame_rgb.transpose(2, 0, 1)
    frame_input = np.expand_dims(frame_input, axis=0)
    frame_input = frame_input.astype(np.float32) / 255.0

    # Run inference
    # time_start = time.time()
    depth = dpt(frame_input)
    # print(f"Inference time: {time.time() - time_start}")

    # Normalize depth map
    # print(f"Depth: {depth.min()}, {depth.max()}")
    depth_normalized = ((depth - depth.min()) / (1000 - depth.min()) * 255).astype(np.uint8)
    
    # Convert to PIL Image, resize to original size, and then to BGR for OpenCV
    depth_image = Image.fromarray(depth_normalized.squeeze(), mode='L')
    depth_image = depth_image.resize(original_size, Image.LANCZOS)
    depth_bgr = cv2.cvtColor(np.array(depth_image), cv2.COLOR_GRAY2BGR)

    return depth_bgr

def run_video(args):
    # Initialize DptTrtInference
    dpt = DptTrtInference(args.engine, 1, (args.size, args.size), (args.size, args.size), multiple_of=32)

    # Open video capture
    cap = cv2.VideoCapture(args.video)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # print(f"Input video resolution: {width}x{height}")
    # print(f"Processing frames at size: {args.size}x{args.size}")
    # print(f"Output video will maintain original resolution: {width}x{height}")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame and measure time
        start_time = time.time()
        depth_frame = process_frame(frame, dpt, args.size)
        end_time = time.time()
        frame_time = end_time - start_time
        total_time += frame_time

        # Write frame (no need to resize as it's already at original resolution)
        out.write(depth_frame)

        # Display progress
        frame_count += 1
        progress = (frame_count / total_frames) * 100
        print(f"\rProcessing: {progress:.2f}% complete | Frame time: {frame_time:.4f}s", end="")

        # Display frame (optional)
        if args.display:
            cv2.imshow('Depth', depth_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Calculate and print average time per frame
    avg_time_per_frame = total_time / frame_count
    print(f"\nVideo processing complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per frame: {avg_time_per_frame:.4f} seconds")
    print(f"Average FPS: {1/avg_time_per_frame:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video and generate depth maps")
    parser.add_argument('--video', type=str, required=True, help='input video file')
    parser.add_argument('--engine', type=str, required=True, help='TensorRT engine file')
    parser.add_argument('--size', type=int, default=798, help='input size for the model')
    parser.add_argument('--output', type=str, default='output.mp4', help='output video file')
    parser.add_argument('--display', action='store_true', help='display processed frames')
    args = parser.parse_args()

    run_video(args)