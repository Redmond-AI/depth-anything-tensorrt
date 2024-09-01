import onnx
from onnx import numpy_helper

def adjust_layernorm_epsilon(model_path, output_path, new_epsilon=1e-5):
    model = onnx.load(model_path)
    for node in model.graph.node:
        if node.op_type == 'LayerNormalization':
            for attr in node.attribute:
                if attr.name == 'epsilon':
                    attr.f = new_epsilon
    onnx.save(model, output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Adjust epsilon in LayerNormalization ops of ONNX model')
    parser.add_argument('--input', type=str, required=True, help='Path to input ONNX model')
    parser.add_argument('--output', type=str, required=True, help='Path to output ONNX model')
    parser.add_argument('--epsilon', type=float, default=1e-5, help='New epsilon value')
    args = parser.parse_args()

    adjust_layernorm_epsilon(args.input, args.output, args.epsilon)
    print(f"Adjusted ONNX model saved to {args.output}")