import onnxruntime as ort
import numpy as np
from PIL import Image
import argparse

def load_image(filepath, size=798):
    img = Image.open(filepath)
    img = img.resize((size, size), Image.LANCZOS)
    img = np.array(img)
    img = np.transpose(img, (2, 0, 1))  # C, H, W
    img = img[None]  # B, C, H, W
    return img.astype(np.float32) / 255.0  # Normalize to [0, 1]

def save_depth_map(depth, output_path):
    depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    depth_image = Image.fromarray(depth_normalized.squeeze(), mode='L')
    depth_image.save(output_path)

parser = argparse.ArgumentParser(description='Validate ONNX model with image input')
parser.add_argument('--img', type=str, required=True, help='Path to the input image')
parser.add_argument('--onnx', type=str, default='depth_anything_v2_vitg_4090_798.onnx', help='Path to the ONNX model')
parser.add_argument('--output', type=str, default='onnx_depth_output.png', help='Path to save the output depth map')
parser.add_argument('--size', type=int, default=798, help='Size to resize the input image (default: 798)')
args = parser.parse_args()

# Load the ONNX model
ort_session = ort.InferenceSession(args.onnx)

# Get input name and shape
input_name = ort_session.get_inputs()[0].name
input_shape = ort_session.get_inputs()[0].shape
print(f"Input name: {input_name}")
print(f"Input shape: {input_shape}")

# Load and preprocess the image
input_data = load_image(args.img, size=args.size)
print(f"Processed input image shape: {input_data.shape}")
print(f"Processed input image min: {input_data.min()}, max: {input_data.max()}")

# Run inference
ort_outputs = ort_session.run(None, {input_name: input_data})

# Print output information
print("ONNX output shape:", ort_outputs[0].shape)
print("ONNX output dtype:", ort_outputs[0].dtype)
print("ONNX output min:", ort_outputs[0].min())
print("ONNX output max:", ort_outputs[0].max())

# Check for NaN values
if np.isnan(ort_outputs[0]).any():
    print("Warning: Output contains NaN values")
else:
    print("Output does not contain NaN values")

# Save the depth map
save_depth_map(ort_outputs[0], args.output)
print(f"Depth map saved to {args.output}")
