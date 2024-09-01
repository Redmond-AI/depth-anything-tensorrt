import tensorrt as trt
import os

def build_engine(onnx_file_path, engine_file_path, fp16_mode=False, workspace_size=20):
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    
    # Use the new method to set workspace size
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size * (1 << 30))  # Convert GB to bytes
    
    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    print(f"Loading ONNX file: {onnx_file_path}")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    print("ONNX file parsed successfully. Building TensorRT engine...")
    engine = builder.build_engine(network, config)
    if engine is None:
        print('ERROR: Failed to build TensorRT engine.')
        return None

    print(f"Saving TensorRT engine to {engine_file_path}")
    with open(engine_file_path, 'wb') as f:
        f.write(engine.serialize())

    print(f'TensorRT engine built and saved to {engine_file_path}')
    return engine_file_path

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build TensorRT engine from ONNX model')
    parser.add_argument('--onnx', type=str, required=True, help='Path to input ONNX model')
    parser.add_argument('--engine', type=str, required=True, help='Path to output TensorRT engine')
    parser.add_argument('--fp16', action='store_true', help='Enable FP16 precision')
    parser.add_argument('--workspace', type=int, default=20, help='Max workspace size in GB')
    args = parser.parse_args()

    build_engine(args.onnx, args.engine, args.fp16, args.workspace)