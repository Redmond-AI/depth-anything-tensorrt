import argparse
import os
import cv2
from PIL import Image
import numpy as np
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.autoprimaryctx
import time

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
    time_start = time.time()
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    ort_outputs = ort_session.run(None, {input_name: input_data})
    print(f"ONNX time: {time.time() - time_start}")
    return ort_outputs[0]

def run_trt(engine_path, input_data):
    time_start = time.time()
    engine = load_engine(engine_path)
    context = engine.create_execution_context()

    # Get input and output tensor names
    input_tensor_name = engine.get_tensor_name(0)
    output_tensor_name = engine.get_tensor_name(1)

    # Get input and output shapes
    input_shape = engine.get_tensor_shape(input_tensor_name)
    output_shape = engine.get_tensor_shape(output_tensor_name)

    print(f"TensorRT input shape: {input_shape}")
    print(f"TensorRT output shape: {output_shape}")

    h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    stream = cuda.Stream()

    # Copy input data to GPU
    np.copyto(h_input, input_data.ravel())
    cuda.memcpy_htod_async(d_input, h_input, stream)

    # Set input and output bindings
    context.set_tensor_address(input_tensor_name, int(d_input))
    context.set_tensor_address(output_tensor_name, int(d_output))

    # Run inference
    context.execute_async_v3(stream_handle=stream.handle)

    # Copy output back to CPU
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    # Reshape the output
    trt_output = h_output.reshape(output_shape)

    print(f"TensorRT input min: {input_data.min()}, max: {input_data.max()}")
    print(f"TensorRT output min: {trt_output.min()}, max: {trt_output.max()}")
    print(f"TensorRT time: {time.time() - time_start}")

    return trt_output

def compare_outputs(onnx_output, trt_output):
    print("Comparing ONNX and TensorRT outputs:")
    print(f"ONNX shape: {onnx_output.shape}, TRT shape: {trt_output.shape}")
    print(f"ONNX min: {onnx_output.min()}, max: {onnx_output.max()}")
    print(f"TRT min: {trt_output.min()}, max: {trt_output.max()}")
    
    abs_diff = np.abs(onnx_output - trt_output)
    print(f"Absolute difference - min: {abs_diff.min()}, max: {abs_diff.max()}, mean: {abs_diff.mean()}")
    
    relative_diff = abs_diff / (np.abs(onnx_output) + 1e-7)
    print(f"Relative difference - min: {relative_diff.min()}, max: {relative_diff.max()}, mean: {relative_diff.mean()}")
    
    mse = np.mean((onnx_output - trt_output)**2)
    print(f"Mean Squared Error: {mse}")

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

    # Check for NaN and Inf values in TensorRT output
    if np.isnan(trt_output).any() or np.isinf(trt_output).any():
        print("Warning: TensorRT output contains NaN or Inf values")
        print(f"Number of NaN values: {np.isnan(trt_output).sum()}")
        print(f"Number of Inf values: {np.isinf(trt_output).sum()}")
    else:
        print("TensorRT output does not contain NaN or Inf values")

    # Save TensorRT depth map
    trt_output_path = os.path.join(args.outdir, f"{os.path.splitext(os.path.basename(args.img))[0]}_trt_depth.png")
    save_depth_map(trt_output, trt_output_path, original_size, not args.grayscale)
    print(f"TensorRT depth map saved to {trt_output_path}")

    # Compare ONNX and TensorRT outputs
    mse = np.mean((onnx_output - trt_output)**2)
    print("Mean Squared Error between ONNX and TensorRT outputs:", mse)

    compare_outputs(onnx_output, trt_output)

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
