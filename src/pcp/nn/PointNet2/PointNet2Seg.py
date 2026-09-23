import torch
import torch.nn.functional as F
from torch import nn

from pcp.core.PointCloud import PointCloud
from pcp.nn.PointNet2.FeaturePropagation import FeaturePropagation
from pcp.nn.PointNet2.PointNet2 import PointNet2


class PointNet2Seg(nn.Module):
    def __init__(
        self,
        num_classes: int = 13,
        backbone: PointNet2 | None = None,
        fp_channels: list[list[int]] | tuple[tuple[int, ...], ...] = (
            (256, 256),
            (256, 128),
            (128, 128, 128),
        ),
        dropout: float = 0.5,
    ):
        super().__init__()
        self.pointnet = backbone if backbone is not None else PointNet2()

        skip_channels = [
            self.pointnet.input_feature_channels,
            *(layer.out_channels for layer in self.pointnet.sa_layers[:-1]),
        ]
        source_channels = self.pointnet.out_channels
        self.fp_layers = nn.ModuleList()

        for skip, channels in zip(reversed(skip_channels), fp_channels):
            layer = FeaturePropagation(source_channels + skip, channels)
            self.fp_layers.append(layer)
            source_channels = layer.out_channels

        self.classifier = nn.Sequential(
            nn.Conv1d(source_channels, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(128, num_classes, kernel_size=1),
        )

    def forward(self, points: PointCloud | torch.Tensor):
        points = points.points if isinstance(points, PointCloud) else points

        xyz = points[..., :3]
        feats = points[..., 3:] if self.pointnet.input_feature_channels else None
        point_levels = [xyz]
        feature_levels = [feats]

        for layer in self.pointnet.sa_layers:
            xyz, feats = layer(xyz, feats)
            point_levels.append(xyz)
            feature_levels.append(feats)

        global_feats = feature_levels[-1]
        for index, layer in enumerate(self.fp_layers):
            source_level = len(point_levels) - 1 - index
            target_level = source_level - 1
            feats = layer(
                point_levels[target_level],
                point_levels[source_level],
                feature_levels[target_level],
                feats,
            )

        logits = self.classifier(feats.transpose(1, 2)).transpose(1, 2)
        return F.log_softmax(logits, dim=-1), global_feats


def main():
    model = PointNet2Seg(num_classes=13)
    points = torch.rand(2, 1024, 3)
    log_probs, global_feats = model(points)

    print("Log probabilities:", log_probs.shape)
    print("Global features:", global_feats.shape)


if __name__ == "__main__":
    main()
