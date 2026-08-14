import torch
import torch.nn as nn
import numpy as np


class AnchorGenerator(nn.Module):

    def __init__(
            self,
            feature_size=32,
            base_size=16,
            stride=16,
            anchor_params=None,
            device='cuda'
    ):
        super(AnchorGenerator, self).__init__()
        self.feature_size = feature_size
        self.base_size = base_size
        self.stride = stride
        self.device = device
        if anchor_params is None:
            self.anchor_params = {
                'small': {'scales': [0.5, 1], 'ratios': [0.5, 1, 2]},
                'medium': {'scales': [1, 2], 'ratios': [0.5, 1, 2]},
                'large': {'scales': [2, 4], 'ratios': [0.5, 1, 2]}
            }
        else:
            self.anchor_params = anchor_params
        self.register_buffer(
            'anchors',
            self._generate_anchors()
        )

    def _generate_anchors(self):
        all_anchors = {}
        for size_name, params in self.anchor_params.items():
            scales = params['scales']
            ratios = params['ratios']
            xs = torch.arange(self.feature_size, device=self.device)
            ys = torch.arange(self.feature_size, device=self.device)
            x_grid, y_grid = torch.meshgrid(xs, ys, indexing='ij')
            anchors_list = []
            for y in range(self.feature_size):
                for x in range(self.feature_size):
                    for scale in scales:
                        for ratio in ratios:
                            w = self.base_size * scale * np.sqrt(ratio)
                            h = self.base_size * scale / np.sqrt(ratio)

                            cx = (x + 0.5) * self.stride
                            cy = (y + 0.5) * self.stride

                            x1 = cx - w / 2
                            y1 = cy - h / 2
                            x2 = cx + w / 2
                            y2 = cy + h / 2

                            anchors_list.append([x1, y1, x2, y2])

            all_anchors[size_name] = torch.tensor(
                anchors_list,
                dtype=torch.float32,
                device=self.device
            )

        all_anchors_tensor = torch.cat(list(all_anchors.values()), dim=0)

        return all_anchors_tensor

    def forward(self, feature_map=None):
        if feature_map is not None:
            h, w = feature_map.shape[-2:]
            expected_h = self.feature_size
            expected_w = self.feature_size

            if h != expected_h or w != expected_w:
                raise ValueError(
                    f"Feature map size ({h}x{w}) doesn't match "
                    f"expected size ({expected_h}x{expected_w})"
                )

        return self.anchors


if __name__ == "__main__":
    anchor_generator = AnchorGenerator()
    anchor_generator.forward()
