import tensorrt as trt
import os

def build_engine(onnx_file_path, engine_file_path, precision='fp32', workspace_size=20):
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size * (1 << 30))
    
    if precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == 'tf32':
        config.set_flag(trt.BuilderFlag.TF32)
    # For FP32, we don't need to set any specific flag

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
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print('ERROR: Failed to build TensorRT engine.')
        return None

    print(f"Saving TensorRT engine to {engine_file_path}")
    with open(engine_file_path, 'wb') as f:
        f.write(serialized_engine)

    print(f'TensorRT engine built and saved to {engine_file_path}')
    return engine_file_path

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build TensorRT engine from ONNX model')
    parser.add_argument('--onnx', type=str, required=True, help='Path to input ONNX model')
    parser.add_argument('--engine', type=str, required=True, help='Path to output TensorRT engine')
    parser.add_argument('--precision', type=str, choices=['fp32', 'fp16', 'tf32'], default='fp32', help='Precision mode')
    parser.add_argument('--workspace', type=int, default=20, help='Max workspace size in GB')
    args = parser.parse_args()

    build_engine(args.onnx, args.engine, args.precision, args.workspace)