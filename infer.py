import argparse
import os
import cv2
from PIL import Image
import numpy as np
from dpt.dpt import DptTrtInference
import torch

def load_image(filepath, size=None):
    img = Image.open(filepath)
    if size:
        img = img.resize((size, size), Image.LANCZOS)
    img = np.array(img)
    img = np.transpose(img, (2, 0, 1))  # C, H, W
    img = img[None]  # B, C, H, W
    return img.astype(np.uint8)

def run(args):
    os.makedirs(args.outdir, exist_ok=True)
    input_img = load_image(args.img, args.size)

    print(f"Input image shape: {input_img.shape}")

    # Use args.size for both input and output sizes
    dpt = DptTrtInference(args.engine, 1, (args.size, args.size), (args.size, args.size), multiple_of=32)
    
    print(f"DPT input shape: {dpt.input_shape}")
    print(f"DPT output shape: {dpt.output_shape}")

    # Ensure input_img is the correct shape and type
    input_img = input_img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    input_img = np.ascontiguousarray(input_img)

    print(f"Processed input image shape: {input_img.shape}")
    print(f"Processed input image dtype: {input_img.dtype}")
    print(f"Processed input image min: {input_img.min()}, max: {input_img.max()}")

    depth = dpt(input_img)

    print(f"Depth output shape: {depth.shape}")
    print(f"Depth output dtype: {depth.dtype}")
    print(f"Depth min: {depth.min()}, max: {depth.max()}")

    # Save depth map
    img_name = os.path.basename(args.img)
    output_path = f'{args.outdir}/{os.path.splitext(img_name)[0]}_depth.png'
    
    # Consider using a different normalization method if needed
    depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    
    if args.grayscale:
        cv2.imwrite(output_path, depth_normalized)
    else:
        colored_depth = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
        cv2.imwrite(output_path, colored_depth)

    print(f"Depth saved to {output_path}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run depth estimation with a TensorRT engine.')
    parser.add_argument('--img', type=str, required=True, help='Path to the input image')
    parser.add_argument('--outdir', type=str, default='./assets', help='Output directory for the depth map')
    parser.add_argument('--engine', type=str, required=True, help='Path to the TensorRT engine')
    parser.add_argument('--grayscale', action='store_true', help='Save the depth map in grayscale')
    parser.add_argument('--size', type=int, help='Resize image to this size (width and height)')
    args = parser.parse_args()

    run(args)