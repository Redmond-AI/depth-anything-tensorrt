import cv2
import numpy as np
from PIL import Image
import torch
import argparse
from dpt.dpt import DptTrtInference

def process_frame(frame, dpt, size):
    # Resize and preprocess the frame
    frame = cv2.resize(frame, (size, size))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.transpose(2, 0, 1)
    frame = np.expand_dims(frame, axis=0)
    frame = frame.astype(np.float32) / 255.0

    # Run inference
    depth = dpt(frame)

    # Normalize depth map
    depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    
    # Convert to PIL Image and then to BGR for OpenCV
    depth_image = Image.fromarray(depth_normalized.squeeze(), mode='L')
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

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        depth_frame = process_frame(frame, dpt, args.size)

        # Resize depth frame to match original video size
        depth_frame = cv2.resize(depth_frame, (width, height))

        # Write frame
        out.write(depth_frame)

        # Display progress
        frame_count += 1
        progress = (frame_count / total_frames) * 100
        print(f"\rProcessing: {progress:.2f}% complete", end="")

        # Display frame (optional)
        if args.display:
            cv2.imshow('Depth', depth_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\nVideo processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video and generate depth maps")
    parser.add_argument('--video', type=str, required=True, help='input video file')
    parser.add_argument('--engine', type=str, required=True, help='TensorRT engine file')
    parser.add_argument('--size', type=int, default=798, help='input size for the model')
    parser.add_argument('--output', type=str, default='output.mp4', help='output video file')
    parser.add_argument('--display', action='store_true', help='display processed frames')
    args = parser.parse_args()

    run_video(args)