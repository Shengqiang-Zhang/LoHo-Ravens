import numpy as np
import cliport.models as models
from cliport.utils import utils

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransportImageGoal(nn.Module):
    """Transport module."""

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):
        """Transport module for placing.
        Args:
          in_shape: shape of input image.
          n_rotations: number of rotations of convolving kernel.
          crop_size: crop size around pick argmax used as convolving kernel.
          preprocess: function to preprocess input images.
        """
        super().__init__()

        self.iters = 0
        self.stream_fcn = stream_fcn
        self.n_rotations = n_rotations
        self.crop_size = crop_size  # crop size must be N*16 (e.g. 96)
        self.preprocess = preprocess
        self.cfg = cfg
        self.device = device
        self.batchnorm = self.cfg['train']['batchnorm']

        self.pad_size = int(self.crop_size / 2)
        self.padding = np.zeros((3, 2), dtype=int)
        self.padding[:2, :] = self.pad_size

        in_shape = np.array(in_shape)
        in_shape = tuple(in_shape)
        self.in_shape = in_shape

        # Crop before network (default for Transporters CoRL 2020).
        self.kernel_shape = (self.crop_size, self.crop_size, self.in_shape[2])

        if not hasattr(self, 'output_dim'):
            self.output_dim = 3
        if not hasattr(self, 'kernel_dim'):
            self.kernel_dim = 3

        self.rotator = utils.ImageRotator(self.n_rotations)

        self._build_nets()

    def _build_nets(self):
        stream_one_fcn, _ = self.stream_fcn
        model = models.names[stream_one_fcn]
        self.key_resnet = model(self.in_shape, self.output_dim, self.cfg, self.device)
        self.query_resnet = model(self.in_shape, self.kernel_dim, self.cfg, self.device)
        self.goal_resnet = model(self.in_shape, self.output_dim, self.cfg, self.device)
        print(f"Transport FCN: {stream_one_fcn}")

    def correlate(self, in0, in1, softmax):
        """Correlate two input tensors."""
        # The batch size IS 1 in this function, but the project code is batched.
        assert in0.shape[0] == 1
        output = F.conv2d(in0, in1, padding=(self.pad_size, self.pad_size))
        output = F.interpolate(output, size=(in0.shape[-2], in0.shape[-1]), mode='bilinear')
        output = output[:,:,self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]
        if softmax:
            output_shape = output.shape
            output = output.reshape((1, np.prod(output.shape)))
            output = F.softmax(output, dim=-1)
            output = output.view(output_shape)
        return output

    def forward(self, inp_img, goal_img, p, softmax=True):
        """Forward pass."""

        # Input image.
        pytorch_padding = tuple([int(x) for x in self.padding[::-1].flatten()])
        in_tensor = F.pad(inp_img, pytorch_padding, mode='constant')  # [B W H 6]
        in_tensor = in_tensor.permute(0, 3, 1, 2)  # [B 6 W H]

        # Goal image.
        goal_tensor = F.pad(goal_img, pytorch_padding, mode='constant')  # [B W H 6]
        goal_tensor = goal_tensor.permute(0, 3, 1, 2)  # [B 6 W H]

        # Rotation pivot.
        pv = p + self.pad_size
        hcrop = self.pad_size

        # Cropped input features.
        in_crop = self.rotator(
            x_list=[in_tensor for _ in range(self.n_rotations)], 
            pivot_list=[pv for _ in range(self.n_rotations)]
        )  # list of [B 6 W H]
        B = in_tensor.shape[0]
        in_crop = [
            torch.stack(
                [
                    _in_crop[i, :, pv[i][0]-hcrop:pv[i][0]+hcrop, pv[i][1]-hcrop:pv[i][1]+hcrop]
                    for i in range(B)
                ]
            )
            for _in_crop in in_crop
        ]
        in_crop = torch.cat(in_crop, dim=0)

        # Cropped goal features.
        goal_crop = self.rotator(
            x_list=[goal_tensor for _ in range(self.n_rotations)], 
            pivot_list=[pv for _ in range(self.n_rotations)]
        )
        goal_crop = [
            torch.stack(
                [
                    _goal_crop[i, :, pv[i][0]-hcrop:pv[i][0]+hcrop, pv[i][1]-hcrop:pv[i][1]+hcrop]
                    for i in range(B)
                ]
            )
            for _goal_crop in goal_crop
        ]
        goal_crop = torch.cat(goal_crop, dim=0)

        in_logits = self.key_resnet(in_tensor)
        goal_logits = self.goal_resnet(goal_tensor)
        kernel_crop = self.query_resnet(in_crop)
        goal_crop = self.goal_resnet(goal_crop)

        # Fuse Goal and Transport features
        goal_x_in_logits = goal_logits + in_logits # Mohit: why doesn't multiply work? :(
        goal_x_kernel = goal_crop + kernel_crop

        # TODO(Mohit): Crop after network. Broken for now
        # in_logits = self.key_resnet(in_tensor)
        # kernel_nocrop_logits = self.query_resnet(in_tensor)
        # goal_logits = self.goal_resnet(goal_tensor)

        # goal_x_in_logits = in_logits
        # goal_x_kernel_logits = goal_logits * kernel_nocrop_logits

        # goal_crop = goal_x_kernel_logits.repeat(self.n_rotations, 1, 1, 1)
        # goal_crop = self.rotator(goal_crop, pivot=pv)
        # goal_crop = torch.cat(goal_crop, dim=0)
        # goal_crop = goal_crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]

        if False:
            output = []
            goal_x_kernel = goal_x_kernel.reshape(self.n_rotations, B, *goal_x_kernel.shape[1:])
            for b in range(B):
                output.append(self.correlate(goal_x_in_logits[b].unsqueeze(0), goal_x_kernel[:, b], softmax))
            output = torch.cat(output, dim=0)

            return output  # [B R W H]
        else:
            output = self.batch_correlate(goal_x_in_logits, goal_x_kernel, softmax)

