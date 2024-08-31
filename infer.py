import argparse
import os
import cv2
from PIL import Image
import numpy as np
from dpt.dpt import DptTrtInference
import torch

def load_image(filepath, size=None):
    img = Image.open(filepath)
    original_size = img.size
    if size:
        img = img.resize((size, size), Image.LANCZOS)
    img = np.array(img)
    img = np.transpose(img, (2, 0, 1))  # C, H, W
    img = img[None]  # B, C, H, W
    return img.astype(np.uint8), original_size

def delete_icc_profile(image_path):
    img = Image.open(image_path)
    img.info.pop('icc_profile', None)
    img.save(image_path)

def run(args):
    os.makedirs(args.outdir, exist_ok=True)
    input_img, original_size = load_image(args.img, args.size)

    print(f"Original image size: {original_size}")
    print(f"Input image shape: {input_img.shape}")

    dpt = DptTrtInference(args.engine, 1, (args.size, args.size), (args.size, args.size), multiple_of=32)
    
    print(f"DPT input shape: {dpt.input_shape}")
    print(f"DPT output shape: {dpt.output_shape}")

    input_img = input_img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    input_img = np.ascontiguousarray(input_img)

    print(f"Processed input image shape: {input_img.shape}")
    print(f"Processed input image dtype: {input_img.dtype}")
    print(f"Processed input image min: {input_img.min()}, max: {input_img.max()}")

    depth = dpt(input_img)

    print(f"Depth output shape: {depth.shape}")
    print(f"Depth output dtype: {depth.dtype}")
    print(f"Depth min: {depth.min()}, max: {depth.max()}")

    depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    depth_image = Image.fromarray(depth_normalized.squeeze(), mode='L')
    
    # Resize depth image to original input image size
    depth_image = depth_image.resize(original_size, Image.LANCZOS)

    # Save depth map
    output_path = os.path.join(args.outdir, f"{os.path.splitext(os.path.basename(args.img))[0]}_depth.png")
    
    if args.grayscale:
        depth_image.save(output_path)
    else:
        depth_color = cv2.applyColorMap(np.array(depth_image), cv2.COLORMAP_INFERNO)
        cv2.imwrite(output_path, depth_color)

    print(f"Depth saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run depth estimation with a TensorRT engine.')
    parser.add_argument('--img', type=str, required=True, help='Path to the input image')
    parser.add_argument('--outdir', type=str, default='./assets', help='Output directory for the depth map')
    parser.add_argument('--engine', type=str, required=True, help='Path to the TensorRT engine')
    parser.add_argument('--grayscale', action='store_true', help='Save the depth map in grayscale')
    parser.add_argument('--size', type=int, default=798, help='Resize image to this size (width and height) for inference')
    args = parser.parse_args()

    run(args)