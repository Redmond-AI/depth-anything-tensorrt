import torch
import numpy as np
import tensorrt as trt
from .transform import DptPreProcess, DptPostProcess


class DptTrtInference:
    def __init__(self, engine_path, batch_size, input_shape, output_shape, device='cuda'):
        self._device = device
        self._engine_path = engine_path
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.output_shape = output_shape

        # Load TensorRT engine
        self._logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self._logger) as runtime:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        self._context = self._engine.create_execution_context()

        # Allocate device memory using PyTorch
        self._d_input = torch.empty(batch_size, 3, *input_shape, dtype=torch.float32, device=self._device)
        self._d_output = torch.empty(batch_size, *output_shape, dtype=torch.float32, device=self._device)

        # Create CUDA stream
        self._stream = torch.cuda.Stream()

        # Assume target_size is the same as input_shape for now
        target_size = input_shape
        self._pre_process = DptPreProcess(input_shape, target_size, device=self._device)
        self._post_process = DptPostProcess(output_shape, device=self._device)

    def __call__(self, img):
        if img.shape != (self.batch_size, 3, *self.input_shape):
            raise ValueError(f"Input shape {img.shape} does not match expected shape {(self.batch_size, 3, *self.input_shape)}")
        
        # Convert numpy array to PyTorch tensor and move to GPU
        img_tensor = torch.from_numpy(img).to(self._device)
        
        # Copy input data to device
        self._d_input.copy_(img_tensor)

        # Run inference
        bindings = [int(self._d_input.data_ptr()), int(self._d_output.data_ptr())]
        self._context.execute_async_v2(bindings, self._stream.cuda_stream)

        # Synchronize CUDA stream
        self._stream.synchronize()

        # Copy output back to host
        output = self._d_output.cpu().numpy()

        return output
