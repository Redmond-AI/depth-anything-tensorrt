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

def delete_icc_profile(image_path):
    img = Image.open(image_path)
    img.info.pop('icc_profile', None)
    img.save(image_path)

def run(args):
    os.makedirs(args.outdir, exist_ok=True)
    input_img = load_image(args.img, args.size)

    print(f"Input image shape: {input_img.shape}")

    dpt = DptTrtInference(args.engine, 1, (args.size, args.size), (args.size, args.size), multiple_of=32)
    
    print(f"DPT input shape: {dpt.input_shape}")
    print(f"DPT output shape: {dpt.output_shape}")

    input_img = input_img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    input_img = np.ascontiguousarray(input_img)

    print(f"Processed input image shape: {input_img.shape}")
    print(f"Processed input image dtype: {input_img.dtype}")
    print(f"Processed input image min: {input_img.min()}, max: {input_img.max()}")

    try:
        depth = dpt(input_img)
        depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
        depth_image = Image.fromarray(depth_normalized.squeeze(), mode='L')
        depth_image.save("./assets/cynthia_depth.png")
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        print(f"TensorRT engine input shape: {dpt._engine.get_tensor_shape('input')}")
        print(f"TensorRT engine output shape: {dpt._engine.get_tensor_shape('output')}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run depth estimation with a TensorRT engine.')
    parser.add_argument('--img', type=str, required=True, help='Path to the input image')
    parser.add_argument('--outdir', type=str, default='./assets', help='Output directory for the depth map')
    parser.add_argument('--engine', type=str, required=True, help='Path to the TensorRT engine')
    parser.add_argument('--grayscale', action='store_true', help='Save the depth map in grayscale')
    parser.add_argument('--size', type=int, help='Resize image to this size (width and height)')
    args = parser.parse_args()

    run(args)