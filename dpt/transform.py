import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class DptPreProcess(object):
    def __init__(self, input_size, target_height, target_width, device='cuda', multiple_of=32):
        self._device = device
        self._multiple_of = multiple_of
        self._height, self._width = self.get_input_size(input_size[0], input_size[1], target_height, target_width)
        # ... rest of the initialization ...

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self._multiple_of) * self._multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self._multiple_of) * self._multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self._multiple_of) * self._multiple_of).astype(int)

        return y

    def get_input_size(self, orig_height, orig_width, target_height, target_width):
        scale_height = target_height / orig_height
        scale_width = target_width / orig_width

        if scale_width > scale_height:
            scale_height = scale_width
        else:
            scale_width = scale_height

        new_height = self.constrain_to_multiple_of(scale_height * orig_height, min_val=target_height)
        new_width = self.constrain_to_multiple_of(scale_width * orig_width, min_val=target_width)

        return (new_height, new_width)

    def __call__(self, img):
        img = torch.as_tensor(img, device=self._device)
        img = img.reshape(-1, self._img_channel, *img.shape[-2:])
        img = self._transforms(img)
        img = img.reshape(-1, self._img_channel, self._height, self._width)
        return img
    

class DptPostProcess(object):
    def __init__(self, output_size):
        self._output_size = output_size

    def _normalize(self, x):
        """Per channel normalize
        """
        out_min = x.amin((-2, -1), keepdim=True)
        out_max = x.amax((-2, -1), keepdim=True)
        return (x - out_min) / (out_max - out_min) * 255.

    def __call__(self, depth):
        depth = depth.reshape(*self._depth_shape)
        depth = F.interpolate(depth.unsqueeze(1), size=self._target_size, mode='bilinear', align_corners=False)
        depth = self._normalize(depth)

        return depth.to(self._dtype).squeeze(1)