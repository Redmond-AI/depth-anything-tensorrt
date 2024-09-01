import tensorrt as trt
import argparse
import os

def build_engine(onnx_file_path, engine_file_path, fp16_mode=False, workspace_size=16384):
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    
    # Set the maximum workspace size (convert MB to bytes)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size * (1024 * 1024))
    
    # Enable TF32 inference
    config.set_flag(trt.BuilderFlag.TF32)
    
    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # Parse ONNX file
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    print(f"Building TensorRT engine with {workspace_size}MB workspace. This may take a few minutes.")
    plan = builder.build_serialized_network(network, config)
    
    with open(engine_file_path, "wb") as f:
        f.write(plan)
    
    print(f"TensorRT engine has been built and saved to {engine_file_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert ONNX model to TensorRT engine')
    parser.add_argument('--onnx', type=str, required=True, help='Path to input ONNX model')
    parser.add_argument('--engine', type=str, required=True, help='Path to output TensorRT engine')
    parser.add_argument('--fp16', action='store_true', help='Enable FP16 precision')
    parser.add_argument('--workspace', type=int, default=16384, help='Max workspace size in MB')
    args = parser.parse_args()

    build_engine(args.onnx, args.engine, args.fp16, args.workspace)

if __name__ == '__main__':
    main()