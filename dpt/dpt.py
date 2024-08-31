import torch
import numpy as np
import tensorrt as trt
from .transform import DptPreProcess, DptPostProcess


class DptTrtInference:
    def __init__(self, engine_path, batch_size, input_shape, output_shape, device='cuda', multiple_of=32):
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

        # Unpack input_shape and output_shape tuples
        input_height, input_width = input_shape
        output_height, output_width = output_shape

        self._pre_process = DptPreProcess((input_height, input_width), output_height, output_width, device=self._device, multiple_of=multiple_of)
        self._post_process = DptPostProcess((output_height, output_width))  # Removed device argument

    def __call__(self, img):
        if img.shape != (self.batch_size, 3, *self.input_shape):
            raise ValueError(f"Input shape {img.shape} does not match expected shape {(self.batch_size, 3, *self.input_shape)}")
        
        # Convert numpy array to PyTorch tensor and move to GPU
        img_tensor = torch.from_numpy(img).to(self._device)
        
        # Copy input data to device
        self._d_input.copy_(img_tensor)

        # Run inference
        bindings = [int(self._d_input.data_ptr()), int(self._d_output.data_ptr())]
        
        # Use execute_async_v3 instead of execute_async_v2
        self._context.execute_async_v3(bindings=bindings, stream_handle=self._stream.cuda_stream)

        # Synchronize CUDA stream
        self._stream.synchronize()

        # Copy output back to host
        output = self._d_output.cpu().numpy()

        return output
