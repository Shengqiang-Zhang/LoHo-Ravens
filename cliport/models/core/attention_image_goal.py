"""Attention module."""

import numpy as np
import torch
import torch.nn.functional as F


from cliport.models.core.attention import Attention


class AttentionImageGoal(Attention):
    """Attention (a.k.a Pick) with image-goals module."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)

    def forward(self, inp_img, goal_img, softmax=True):
        """Forward pass."""
        # Input image.
        pytorch_padding = tuple([int(x) for x in self.padding[::-1].flatten()])
        in_tens = F.pad(inp_img, pytorch_padding, mode='constant')  # [B W H 6]
        
        goal_tensor = F.pad(goal_img, pytorch_padding, mode='constant')  # [B W H 6]
        in_tens = in_tens * goal_tensor  # [B W H 6]

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
        logits = logits.view(self.n_rotations, -1, *logits.shape[1:])  # [R B 1 W H]

        # Rotate back output.
        logits = self.rotator(
            x_list=logits, 
            reverse=True, 
            pivot_list=[pv for _ in range(self.n_rotations)],
        )
        logits = torch.cat(logits, dim=0)  # [R * B 1 W H]
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