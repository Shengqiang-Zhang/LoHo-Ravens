import numpy as np
import torch
import torch.nn.functional as F

from cliport.models.core.attention import Attention
import cliport.models as models
import cliport.models.core.fusion as fusion


class TwoStreamAttentionLangFusion(Attention):
    """Two Stream Language-Conditioned Attention (a.k.a Pick) module."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        self.fusion_type = cfg['train']['attn_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)

    def _build_nets(self):
        stream_one_fcn, stream_two_fcn = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        stream_two_model = models.names[stream_two_fcn]

        self.attn_stream_one = stream_one_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.attn_stream_two = stream_two_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.fusion = fusion.names[self.fusion_type](input_dim=1)

        print(f"Attn FCN - Stream One: {stream_one_fcn}, Stream Two: {stream_two_fcn}, Stream Fusion: {self.fusion_type}")

    def attend(self, x, l):
        x1 = self.attn_stream_one(x)
        x2 = self.attn_stream_two(x, l)
        x = self.fusion(x1, x2)
        return x

    def forward(self, inp_img, lang_goal, softmax=True):
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
        logits = self.attend(rotated_in_tens, lang_goal)  # [R * B 1 W H]
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
            output = output.view(B, *logits.shape[1:])
        return output  # [B R * W * H]


class TwoStreamAttentionLangFusionLat(TwoStreamAttentionLangFusion):
    """Language-Conditioned Attention (a.k.a Pick) module with lateral connections."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        self.fusion_type = cfg['train']['attn_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)

    def attend(self, x, l):
        x1, lat = self.attn_stream_one(x)
        x2 = self.attn_stream_two(x, lat, l)
        x = self.fusion(x1, x2)
        return x
