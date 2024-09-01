import argparse
import os
import cv2
from PIL import Image
import numpy as np
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def load_image(filepath, size=None):
    img = Image.open(filepath)
    original_size = img.size
    if size:
        img = img.resize((size, size), Image.LANCZOS)
    img = np.array(img)
    img = np.transpose(img, (2, 0, 1))  # C, H, W
    img = img[None]  # B, C, H, W
    return img.astype(np.float32) / 255.0, original_size  # Normalize to [0, 1]

def save_depth_map(depth, output_path, original_size, colormap=False):
    depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    depth_image = Image.fromarray(depth_normalized.squeeze(), mode='L')
    depth_image = depth_image.resize(original_size, Image.LANCZOS)
    
    if colormap:
        depth_color = cv2.applyColorMap(np.array(depth_image), cv2.COLORMAP_INFERNO)
        cv2.imwrite(output_path, depth_color)
    else:
        depth_image.save(output_path)

def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def run_onnx(onnx_path, input_data):
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    ort_outputs = ort_session.run(None, {input_name: input_data})
    return ort_outputs[0]

def run_trt(engine_path, input_data):
    engine = load_engine(engine_path)
    context = engine.create_execution_context()

    # Allocate memory for input and output
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_tensor_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_tensor_shape(1)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    stream = cuda.Stream()

    # Copy input data to GPU
    np.copyto(h_input, input_data.ravel())
    cuda.memcpy_htod_async(d_input, h_input, stream)

    # Run inference
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)

    # Copy output back to CPU
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    # Reshape the output
    trt_output = h_output.reshape(engine.get_tensor_shape(1))
    return trt_output

def run(args):
    os.makedirs(args.outdir, exist_ok=True)
    input_data, original_size = load_image(args.img, args.size)

    print(f"Original image size: {original_size}")
    print(f"Input image shape: {input_data.shape}")
    print(f"Input image dtype: {input_data.dtype}")
    print(f"Input image min: {input_data.min()}, max: {input_data.max()}")

    # Run ONNX inference
    onnx_output = run_onnx(args.onnx, input_data)
    print(f"ONNX output shape: {onnx_output.shape}")
    print(f"ONNX output dtype: {onnx_output.dtype}")
    print(f"ONNX output min: {onnx_output.min()}, max: {onnx_output.max()}")

    # Save ONNX depth map
    onnx_output_path = os.path.join(args.outdir, f"{os.path.splitext(os.path.basename(args.img))[0]}_onnx_depth.png")
    save_depth_map(onnx_output, onnx_output_path, original_size, not args.grayscale)
    print(f"ONNX depth map saved to {onnx_output_path}")

    # Run TensorRT inference
    trt_output = run_trt(args.engine, input_data)
    print(f"TensorRT output shape: {trt_output.shape}")
    print(f"TensorRT output dtype: {trt_output.dtype}")
    print(f"TensorRT output min: {trt_output.min()}, max: {trt_output.max()}")

    # Check for NaN values in TensorRT output
    if np.isnan(trt_output).any():
        print("Warning: TensorRT output contains NaN values")
    else:
        print("TensorRT output does not contain NaN values")

    # Save TensorRT depth map
    trt_output_path = os.path.join(args.outdir, f"{os.path.splitext(os.path.basename(args.img))[0]}_trt_depth.png")
    save_depth_map(trt_output, trt_output_path, original_size, not args.grayscale)
    print(f"TensorRT depth map saved to {trt_output_path}")

    # Compare ONNX and TensorRT outputs
    mse = np.mean((onnx_output - trt_output)**2)
    print("Mean Squared Error between ONNX and TensorRT outputs:", mse)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate TensorRT engine with ONNX model')
    parser.add_argument('--img', type=str, required=True, help='Path to the input image')
    parser.add_argument('--outdir', type=str, default='./assets', help='Output directory for the depth maps')
    parser.add_argument('--onnx', type=str, required=True, help='Path to the ONNX model')
    parser.add_argument('--engine', type=str, required=True, help='Path to the TensorRT engine')
    parser.add_argument('--grayscale', action='store_true', help='Save the depth maps in grayscale')
    parser.add_argument('--size', type=int, default=798, help='Resize image to this size (width and height) for inference')
    args = parser.parse_args()

    run(args)
