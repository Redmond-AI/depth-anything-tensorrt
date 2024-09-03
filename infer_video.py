import cv2
import numpy as np
from PIL import Image
import torch
import argparse
from dpt.dpt import DptTrtInference
import time

def split_image_into_four(image):
    # Convert image to numpy array if it's not already
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Get dimensions
    height, width, channels = image.shape
    
    # Create four empty images
    images = [np.zeros((height//2, width//2, channels), dtype=image.dtype) for _ in range(4)]
    
    # Fill the four images
    images[0] = image[0::2, 0::2]  # Top-left pixels
    images[1] = image[0::2, 1::2]  # Top-right pixels
    images[2] = image[1::2, 0::2]  # Bottom-left pixels
    images[3] = image[1::2, 1::2]  # Bottom-right pixels
    
    # Convert back to PIL Images
    return [Image.fromarray(img) for img in images]

def process_frame_multi_res(frame, dpt, sizes):
    original_size = frame.shape[:2][::-1]  # (width, height)
    depths = []

    for size in sizes:
        # Resize the frame to the current size
        frame_resized = cv2.resize(frame, (size, size))
        
        # Resize again to 798x798 for DPT input
        frame_dpt = cv2.resize(frame_resized, (798, 798))
        
        # Preprocess the frame
        frame_rgb = cv2.cvtColor(frame_dpt, cv2.COLOR_BGR2RGB)
        frame_input = frame_rgb.transpose(2, 0, 1)
        frame_input = np.expand_dims(frame_input, axis=0)
        frame_input = frame_input.astype(np.float32) / 255.0

        # Run inference
        depth = dpt(frame_input)
        
        # Resize depth back to original size
        depth_resized = cv2.resize(depth.squeeze(), original_size)
        depths.append(depth_resized)

    # Combine depths using max operation
    combined_depth = np.max(depths, axis=0)
    
    return combined_depth, original_size

def process_frame_split(frame, dpt, sizes):
    original_size = frame.shape[:2][::-1]  # (width, height)
    depths = []

    # Split the frame into four images
    split_images = split_image_into_four(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    for split_image in split_images:
        split_depths = []
        for size in sizes:
            # Resize the split image to the current size
            image_resized = split_image.resize((size, size), Image.LANCZOS)
            
            # Resize again to 798x798 for DPT input
            image_dpt = image_resized.resize((798, 798), Image.LANCZOS)
            
            # Preprocess the image
            image_np = np.array(image_dpt)
            image_input = image_np.transpose(2, 0, 1)
            image_input = np.expand_dims(image_input, axis=0)
            image_input = image_input.astype(np.float32) / 255.0

            # Run inference
            depth = dpt(image_input)
            
            # Resize depth back to original split image size
            depth_resized = cv2.resize(depth.squeeze(), (original_size[0]//2, original_size[1]//2))
            split_depths.append(depth_resized)
        
        # Combine depths for this split image using max operation
        combined_split_depth = np.max(split_depths, axis=0)
        depths.append(combined_split_depth)

    # Reconstruct the full depth map
    full_depth = np.zeros(original_size[::-1], dtype=np.float32)
    full_depth[0::2, 0::2] = depths[0]
    full_depth[0::2, 1::2] = depths[1]
    full_depth[1::2, 0::2] = depths[2]
    full_depth[1::2, 1::2] = depths[3]
    
    return full_depth, original_size

def normalize_and_convert_depth(depth, original_size, global_min, global_max):
    depth_normalized = ((depth - global_min) / (global_max - global_min) * 255).astype(np.uint8)
    
    # Convert to PIL Image, resize to original size, and then to BGR for OpenCV
    depth_image = Image.fromarray(depth_normalized.squeeze(), mode='L')
    depth_image = depth_image.resize(original_size, Image.LANCZOS)
    depth_bgr = cv2.cvtColor(np.array(depth_image), cv2.COLOR_GRAY2BGR)

    return depth_bgr

def run_video(args):
    sizes = [392, 518, 630, 798]
    # Initialize a single DptTrtInference for 798x798 input
    dpt = DptTrtInference(args.engine, 1, (798, 798), (798, 798), multiple_of=32)

    # Open video capture
    cap = cv2.VideoCapture(args.video)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # First pass: determine global min and max depth values
    global_min = float('inf')
    global_max = float('-inf')
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if args.method == 'multi_res':
            depth, _ = process_frame_multi_res(frame, dpt, sizes)
        else:  # split method
            depth, _ = process_frame_split(frame, dpt, sizes)
        
        global_min = min(global_min, depth.min())
        global_max = max(global_max, depth.max())

        frame_count += 1
        progress = (frame_count / total_frames) * 100
        print(f"\rFirst pass: {progress:.2f}% complete", end="")

    print(f"\nGlobal depth range: {global_min:.2f} to {global_max:.2f}")

    # Reset video capture for second pass
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # Second pass: process and save normalized depth frames
    frame_count = 0
    total_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame and measure time
        start_time = time.time()
        if args.method == 'multi_res':
            depth, original_size = process_frame_multi_res(frame, dpt, sizes)
        else:  # split method
            depth, original_size = process_frame_split(frame, dpt, sizes)
        depth_frame = normalize_and_convert_depth(depth, original_size, global_min, global_max)
        end_time = time.time()
        frame_time = end_time - start_time
        total_time += frame_time

        # Write frame
        out.write(depth_frame)

        # Display progress
        frame_count += 1
        progress = (frame_count / total_frames) * 100
        print(f"\rSecond pass: {progress:.2f}% complete | Frame time: {frame_time:.4f}s", end="")

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
    parser.add_argument('--method', type=str, choices=['multi_res', 'split'], default='multi_res',
                        help='processing method: multi_res (multiple resolutions) or split (split image)')
    args = parser.parse_args()

    run_video(args)