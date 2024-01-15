import torch
import numpy as np
import torch.nn.functional as F

import cliport.models as models
import cliport.models.core.fusion as fusion
from cliport.models.core.transport import Transport


class TwoStreamTransportLangFusion(Transport):
    """Two Stream Transport (a.k.a Place) module"""

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):
        self.fusion_type = cfg['train']['trans_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)

    def _build_nets(self):
        stream_one_fcn, stream_two_fcn = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        stream_two_model = models.names[stream_two_fcn]

        self.key_stream_one = stream_one_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.key_stream_two = stream_two_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_one = stream_one_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_two = stream_two_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)
        self.fusion_key = fusion.names[self.fusion_type](input_dim=self.kernel_dim)
        self.fusion_query = fusion.names[self.fusion_type](input_dim=self.kernel_dim)

        print(f"Transport FCN - Stream One: {stream_one_fcn}, Stream Two: {stream_two_fcn}, Stream Fusion: {self.fusion_type}")

    def transport(self, in_tensor, crop, l):
        logits = self.fusion_key(self.key_stream_one(in_tensor), self.key_stream_two(in_tensor, l))
        kernel = self.fusion_query(self.query_stream_one(crop), self.query_stream_two(crop, l))
        return logits, kernel

    def forward(self, inp_img, p, lang_goal, softmax=True):
        """Forward pass."""
        pytorch_padding = tuple([int(x) for x in self.padding[::-1].flatten()])
        in_tensor = F.pad(inp_img, pytorch_padding, mode='constant')  # [B W H 6]

        # Rotation pivot.
        pv = p + self.pad_size

        # Crop before network (default for Transporters CoRL 2020).
        hcrop = self.pad_size
        in_tensor = in_tensor.permute(0, 3, 1, 2)

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
        crop = torch.cat(crop, dim=0)  # [R * B 6 2h 2h]

        logits, kernel = self.transport(in_tensor, crop, lang_goal)

        # TODO(Mohit): Crop after network. Broken for now.
        # # Crop after network (for receptive field, and more elegant).
        # in_tensor = in_tensor.permute(0, 3, 1, 2)
        # logits, crop = self.transport(in_tensor, lang_goal)
        # crop = crop.repeat(self.n_rotations, 1, 1, 1)
        # crop = self.rotator(crop, pivot=pv)
        # crop = torch.cat(crop, dim=0)
        # hcrop = self.pad_size
        # kernel = crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]

        if False:
            output1 = []
            kernel = kernel.reshape(self.n_rotations, B, *kernel.shape[1:])
            for b in range(B):
                output1.append(self.correlate(logits[b].unsqueeze(0), kernel[:, b], softmax))
            output1 = torch.cat(output1, dim=0)
        else:
            output = self.batch_correlate(logits, kernel, softmax)

        # print("consistency check")
        # print("output.mean()", (output * 10000).mean())
        # print("output1.mean()", (output1 * 10000).mean())
        # print("diff", (output * 10000 - output1 * 10000).mean())
        return output  # [B R W H]

class TwoStreamTransportLangFusionLat(TwoStreamTransportLangFusion):
    """Two Stream Transport (a.k.a Place) module with lateral connections"""

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):

        self.fusion_type = cfg['train']['trans_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)

    def transport(self, in_tensor, crop, l):
        key_out_one, key_lat_one = self.key_stream_one(in_tensor)
        key_out_two = self.key_stream_two(in_tensor, key_lat_one, l)
        logits = self.fusion_key(key_out_one, key_out_two)

        query_out_one, query_lat_one = self.query_stream_one(crop)
        query_out_two = self.query_stream_two(crop, query_lat_one, l)
        kernel = self.fusion_query(query_out_one, query_out_two)

        return logits, kernel
