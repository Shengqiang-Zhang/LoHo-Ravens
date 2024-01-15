import numpy as np
import cliport.models as models
from cliport.utils import utils

import torch
import torch.nn as nn
import torch.nn.functional as F


class Transport(nn.Module):

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):
        """Transport (a.k.a Place) module."""
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

        # Crop before network (default from Transporters CoRL 2020).
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
        self.query_resnet = model(self.kernel_shape, self.kernel_dim, self.cfg, self.device)
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
    
    def batch_correlate(self, in0, in1, softmax):
        """Correlate two input tensors with grouped convolutions."""
        # in0: [B D W H]
        # in1: [R * B D W H]
        B = in0.shape[0]
        in1 = in1.view(self.n_rotations, B, *in1.shape[1:])  # [R B D W H]
        in1 = in1.permute(1, 0, 2, 3, 4)  # [B R D W H]
        in1 = in1.reshape(B * self.n_rotations, *in1.shape[2:])  # [B * R D W H]
        output = F.conv2d(
            in0.reshape(1, B * in0.shape[1], in0.shape[2], in0.shape[3]),  # [1 B*D W H]
            in1,  # [B * R D W H]
            padding=(self.pad_size, self.pad_size),
            groups=B,
        )  # [1 B * R W H]
        output = output.view(B, self.n_rotations, output.shape[-2], output.shape[-1])  # [B R W H]
        output = F.interpolate(output, size=(in0.shape[-2], in0.shape[-1]), mode='bilinear')
        output = output[:,:,self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]
        if softmax:
            output_shape = output.shape
            output = output.reshape(B, np.prod(output.shape[1:]))
            output = F.softmax(output, dim=-1)
            output = output.view(output_shape)
        return output

    def transport(self, in_tensor, crop):
        logits = self.key_resnet(in_tensor)
        kernel = self.query_resnet(crop)
        return logits, kernel

    def forward(self, inp_img, p, softmax=True):
        """Forward pass."""
        pytorch_padding = tuple([int(x) for x in self.padding[::-1].flatten()])
        in_tensor = F.pad(inp_img, pytorch_padding, mode='constant')  # [B W H 6]

        # Rotation pivot.
        pv = p + self.pad_size

        # Crop before network (default from Transporters CoRL 2020).
        hcrop = self.pad_size
        in_tensor = in_tensor.permute(0, 3, 1, 2) # [B D W H]

        crop = self.rotator(
            x_list=[in_tensor for _ in range(self.n_rotations)], 
            pivot_list=[pv for _ in range(self.n_rotations)]
        )
        B = in_tensor.shape[0]
        crop = [
            torch.stack(
                [
                    _crop[i, :, pv[i][0]-hcrop:pv[i][0]+hcrop, pv[i][1]-hcrop:pv[i][1]+hcrop]
                    for i in range(B)
                ]
            )
            for _crop in crop
        ]
        crop = torch.cat(crop, dim=0)

        logits, kernel = self.transport(in_tensor, crop)

        # TODO(Mohit): Crop after network. Broken for now.
        # in_tensor = in_tensor.permute(0, 3, 1, 2)
        # logits, crop = self.transport(in_tensor)
        # crop = crop.repeat(self.n_rotations, 1, 1, 1)
        # crop = self.rotator(crop, pivot=pv)
        # crop = torch.cat(crop, dim=0)

        # kernel = crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]
        # kernel = crop[:, :, p[0]:(p[0] + self.crop_size), p[1]:(p[1] + self.crop_size)]

        if False:
            output = []
            kernel = kernel.reshape(self.n_rotations, B, *kernel.shape[1:])
            for b in range(B):
                output.append(self.correlate(logits[b].unsqueeze(0), kernel[:, b], softmax))
            output = torch.cat(output, dim=0)

            return output  # [B R W H]
        else:
            return self.batch_correlate(logits, kernel, softmax)

