import cv2
import numpy as np
from PIL import Image
import torch
import argparse
from dpt.dpt import DptTrtInference
import time

def split_image_into_four_cpu(image):
    # Convert image to numpy array if it's not already
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Get dimensions
    height, width, channels = image.shape
    
    # Create four images using advanced indexing
    images = [
        image[0::2, 0::2],  # Top-left pixels
        image[0::2, 1::2],  # Top-right pixels
        image[1::2, 0::2],  # Bottom-left pixels
        image[1::2, 1::2]   # Bottom-right pixels
    ]
    
    return images

def split_image_into_four_gpu(image):
    # Convert image to torch tensor if it's not already
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).cuda()
    elif isinstance(image, Image.Image):
        image = torch.from_numpy(np.array(image)).cuda()
    elif isinstance(image, torch.Tensor) and not image.is_cuda:
        image = image.cuda()
    
    # Get dimensions
    height, width, channels = image.shape
    
    # Create four images using advanced indexing
    images = [
        image[0::2, 0::2],  # Top-left pixels
        image[0::2, 1::2],  # Top-right pixels
        image[1::2, 0::2],  # Bottom-left pixels
        image[1::2, 1::2]   # Bottom-right pixels
    ]
    
    return images

def split_image_into_four(image, use_gpu=False):
    if use_gpu and torch.cuda.is_available():
        return split_image_into_four_gpu(image)
    else:
        return split_image_into_four_cpu(image)

def process_frame_multi_res(frame, dpt, sizes):
    original_size = frame.shape[:2][::-1]  # (width, height)
    depths = []

    for size in sizes:
        # Resize the frame to the current size
        frame_resized = cv2.resize(frame, (size, size))
        
        # Resize again to args.sizexargs.size for DPT input
        frame_dpt = cv2.resize(frame_resized, (args.size, args.size))
        
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

def process_frame_split(frame, dpt, use_gpu=False):
    original_size = frame.shape[:2][::-1]  # (width, height)
    depths = []

    # Split the frame into four images
    split_images = split_image_into_four(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_gpu)

    for split_image in split_images:
        if use_gpu:
            # Convert back to CPU for further processing
            split_image = split_image.cpu().numpy()
        
        # Resize to args.sizexargs.size for DPT input
        image_dpt = cv2.resize(split_image, (args.size, args.size))
        
        # Preprocess the image
        image_input = image_dpt.transpose(2, 0, 1)
        image_input = np.expand_dims(image_input, axis=0)
        image_input = image_input.astype(np.float32) / 255.0

        # Run inference
        depth = dpt(image_input)
        
        # Resize depth back to original split image size
        depth_resized = cv2.resize(depth.squeeze(), (original_size[0]//2, original_size[1]//2))
        depths.append(depth_resized)

    # Reconstruct the full depth map
    full_depth = np.zeros(original_size[::-1], dtype=np.float32)
    full_depth[0::2, 0::2] = depths[0]
    full_depth[0::2, 1::2] = depths[1]
    full_depth[1::2, 0::2] = depths[2]
    full_depth[1::2, 1::2] = depths[3]
    
    return full_depth, original_size

def process_frame_single(frame, dpt, size):
    original_size = frame.shape[:2][::-1]  # (width, height)
    
    # Resize to args.sizexargs.size for DPT input
    frame_dpt = cv2.resize(frame, (size, size))
    
    # Preprocess the frame
    frame_rgb = cv2.cvtColor(frame_dpt, cv2.COLOR_BGR2RGB)
    frame_input = frame_rgb.transpose(2, 0, 1)
    frame_input = np.expand_dims(frame_input, axis=0)
    frame_input = frame_input.astype(np.float32) / 255.0

    # Run inference
    depth = dpt(frame_input)
    
    # Resize depth back to original size
    depth_resized = cv2.resize(depth.squeeze(), original_size)
    
    return depth_resized, original_size

def normalize_and_convert_depth(depth, original_size, global_min, global_max):
    depth_normalized = ((depth - global_min) / (global_max - global_min) * 255).astype(np.uint8)
    
    # Convert to PIL Image, resize to original size, and then to BGR for OpenCV
    depth_image = Image.fromarray(depth_normalized.squeeze(), mode='L')
    depth_image = depth_image.resize(original_size, Image.LANCZOS)
    depth_bgr = cv2.cvtColor(np.array(depth_image), cv2.COLOR_GRAY2BGR)

    return depth_bgr

def run_video(args):
    sizes = [266, 392, 518, 798]
    # Initialize a single DptTrtInference for 798x798 input
    dpt = DptTrtInference(args.engine, 1, (args.size, args.size), (args.size, args.size), multiple_of=32)

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

        if frame_count % args.sample_rate == 0:
            if args.method == 'multi_res':
                depth, _ = process_frame_multi_res(frame, dpt, sizes)
            elif args.method == 'split':
                depth, _ = process_frame_split(frame, dpt, args.use_gpu)
            else:  # single method
                depth, _ = process_frame_single(frame, dpt, args.size)
            
            global_min = min(global_min, depth.min())
            global_max = max(global_max, depth.max())

            progress = (frame_count / total_frames) * 100
            print(f"\rFirst pass: {progress:.2f}% complete", end="")

        frame_count += 1

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
        elif args.method == 'split':
            depth, original_size = process_frame_split(frame, dpt, args.use_gpu)
        else:  # single method
            depth, original_size = process_frame_single(frame, dpt, args.size)
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
    parser.add_argument('--method', type=str, choices=['multi_res', 'split', 'single'], default='multi_res',
                        help='processing method: multi_res (multiple resolutions), split (split image), or single (single pass)')
    parser.add_argument('--sample_rate', type=int, default=10, help='process every Nth frame in the first pass')
    parser.add_argument('--use_gpu', action='store_true', help='use GPU for image splitting (if available)')
    args = parser.parse_args()

    run_video(args)