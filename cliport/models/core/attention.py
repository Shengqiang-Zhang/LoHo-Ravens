"""Attention module."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cliport.models as models
from cliport.utils import utils


class Attention(nn.Module):
    """Attention (a.k.a Pick) module."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        super().__init__()
        self.stream_fcn = stream_fcn
        self.n_rotations = n_rotations
        self.preprocess = preprocess
        self.cfg = cfg
        self.device = device
        self.batchnorm = self.cfg['train']['batchnorm']

        self.padding = np.zeros((3, 2), dtype=int)
        max_dim = np.max(in_shape[:2])
        pad = (max_dim - np.array(in_shape[:2])) / 2
        self.padding[:2] = pad.reshape(2, 1)

        in_shape = np.array(in_shape)
        in_shape += np.sum(self.padding, axis=1)
        in_shape = tuple(in_shape)
        self.in_shape = in_shape

        self.rotator = utils.ImageRotator(self.n_rotations)

        self._build_nets()

    def _build_nets(self):
        stream_one_fcn, _ = self.stream_fcn
        self.attn_stream = models.names[stream_one_fcn](self.in_shape, 1, self.cfg, self.device)
        print(f"Attn FCN: {stream_one_fcn}")

    def attend(self, x):
        return self.attn_stream(x)

    def forward(self, inp_img, softmax=True):
        """Forward pass."""
        pytorch_padding = tuple([int(x) for x in self.padding[::-1].flatten()])
        in_tens = F.pad(inp_img, pytorch_padding, mode='constant')  # [B W H 6]

        # Rotation pivot.
        pv = torch.from_numpy(np.array(in_tens.shape[1:3]) // 2).to(self.device)

        # Rotate input.
        in_tens = in_tens.permute(0, 3, 1, 2)  # [B 6 W H]
        rotated_in_tens = self.rotator(
            x_list=[in_tens for _ in range(self.n_rotations)], 
            pivot_list=[pv for _ in range(self.n_rotations)],
        )

        # Forward pass.
        rotated_in_tens = torch.cat(rotated_in_tens, dim=0)
        logits = self.attend(rotated_in_tens)  # [R * B 1 W H]
        logits = logits.view(self.n_rotations, -1, *logits.shape[1:])

        # Rotate back output.
        logits = self.rotator(
            x_list=logits, 
            reverse=True, 
            pivot_list=[pv for _ in range(self.n_rotations)],
        )
        logits = torch.cat(logits, dim=0)
        c0 = self.padding[:2, 0]
        c1 = c0 + inp_img[0].shape[:2]
        logits = logits[:, :, c0[0]:c1[0], c0[1]:c1[1]]

        logits = logits.squeeze(1)  # [R * B W H]
        B = in_tens.shape[0]
        logits = logits.reshape(self.n_rotations, B, *logits.shape[1:])  # [R B W H]
        logits = logits.permute(1, 0, 2, 3)  # [B R W H]
        output = logits.reshape(B, np.prod(logits.shape[1:]))  # [B R * W * H]
        if softmax:
            output = F.softmax(output, dim=-1)
            output = output.view(B, *logits.shape[1:])  # [B R W H]
        return output  # [B R * W * H]