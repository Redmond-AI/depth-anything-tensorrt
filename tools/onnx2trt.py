import tensorrt as trt
import argparse
import os
import pycuda.driver as cuda
import pycuda.autoinit

def GiB(val):
    return val * 1 << 30

def build_engine(onnx_file_path, engine_file_path, fp16_mode=False, workspace_size=16):
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    
    # Set the maximum workspace size (convert GB to bytes)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, GiB(workspace_size))
    
    # Enable TF32 inference
    config.set_flag(trt.BuilderFlag.TF32)
    
    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # Parse ONNX file
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, logger)
    
    print(f"Loading ONNX file: {onnx_file_path}")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    print("ONNX file parsed successfully.")
    
    # Print network information
    print(f"Network has {network.num_layers} layers")
    print(f"Network has {network.num_inputs} inputs:")
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        print(f"  Input {i}: {tensor.name}, shape={tensor.shape}, dtype={tensor.dtype}")
    print(f"Network has {network.num_outputs} outputs:")
    for i in range(network.num_outputs):
        tensor = network.get_output(i)
        print(f"  Output {i}: {tensor.name}, shape={tensor.shape}, dtype={tensor.dtype}")
    
    print(f"Building TensorRT engine with {workspace_size}GB workspace. This may take a few minutes.")
    plan = builder.build_serialized_network(network, config)
    
    if plan is None:
        print("Failed to create the engine")
        return None
    
    print("Engine built successfully.")
    
    with open(engine_file_path, "wb") as f:
        f.write(plan)
    
    print(f"TensorRT engine has been built and saved to {engine_file_path}")
    return engine_file_path

def main():
    parser = argparse.ArgumentParser(description='Convert ONNX model to TensorRT engine')
    parser.add_argument('--onnx', type=str, required=True, help='Path to input ONNX model')
    parser.add_argument('--engine', type=str, required=True, help='Path to output TensorRT engine')
    parser.add_argument('--fp16', action='store_true', help='Enable FP16 precision')
    parser.add_argument('--workspace', type=int, default=20, help='Max workspace size in GB')
    args = parser.parse_args()

    # Check if CUDA is available
    if not cuda.Device(0).compute_capability():
        print("CUDA is not available. Please check your installation.")
        return

    # Print CUDA and TensorRT versions
    print(f"CUDA Version: {cuda.get_version()}")
    print(f"TensorRT Version: {trt.__version__}")

    engine_file_path = build_engine(args.onnx, args.engine, args.fp16, args.workspace)
    
    if engine_file_path:
        print(f"TensorRT engine created successfully: {engine_file_path}")
    else:
        print("Failed to create TensorRT engine.")

if __name__ == '__main__':
    main()