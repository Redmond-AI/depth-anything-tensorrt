import argparse
import os
import cv2
from PIL import Image
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def load_image(filepath, size):
    img = Image.open(filepath).resize((size, size), Image.LANCZOS)
    img = np.array(img).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return img

def save_depth_map(depth, output_path):
    depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_normalized.squeeze(), cv2.COLORMAP_INFERNO)
    cv2.imwrite(output_path, depth_color)

def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def run_trt(engine, input_data):
    context = engine.create_execution_context()
    input_name, output_name = engine.get_tensor_name(0), engine.get_tensor_name(1)
    input_shape, output_shape = engine.get_tensor_shape(input_name), engine.get_tensor_shape(output_name)

    d_input = cuda.mem_alloc(input_data.nbytes)
    h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()

    cuda.memcpy_htod_async(d_input, input_data, stream)
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))
    context.execute_async_v3(stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    return h_output.reshape(output_shape)

def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    input_data = load_image(args.img, args.size)
    engine = load_engine(args.engine)
    depth = run_trt(engine, input_data)
    output_path = os.path.join(args.outdir, f"{os.path.splitext(os.path.basename(args.img))[0]}_depth.png")
    save_depth_map(depth, output_path)
    print(f"Depth map saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate depth map using TensorRT')
    parser.add_argument('--img', type=str, required=True, help='Path to the input image')
    parser.add_argument('--outdir', type=str, default='./assets', help='Output directory for the depth map')
    parser.add_argument('--engine', type=str, required=True, help='Path to the TensorRT engine')
    parser.add_argument('--size', type=int, default=798, help='Resize image to this size for inference')
    args = parser.parse_args()

    main(args)
