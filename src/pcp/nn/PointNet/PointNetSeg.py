import torch
from torch import nn

from pcp.core.PointCloud import PointCloud
from pcp.nn.PointNet.PointNet import PointNetBackbone

class PointNetSeg(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        use_tnet: bool = False,
        backbone_channels: tuple[tuple[int, ...], tuple[int, ...]] = (
            (64, 64), (64, 128, 1024)
        ),
        segmentation_channels: list[int] | tuple[int, ...] = (512, 256, 128),
    ):
        super().__init__()
        self.use_tnet = use_tnet

        self.pointnet = PointNetBackbone(
            in_channels = in_channels,
            out_channels = backbone_channels,
            global_feats = False,
            use_tnet = use_tnet
        )
        layers = []
        last_channels = self.pointnet.feature_channels + self.pointnet.out_channels
        for channels in segmentation_channels:
            layers.extend((
                nn.Conv1d(last_channels, channels, kernel_size=1),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
            ))
            last_channels = channels
        layers.append(nn.Conv1d(last_channels, num_classes, kernel_size=1))
        self.segmentation_head = nn.Sequential(*layers)

    def forward(self, points: PointCloud | torch.Tensor):
        x = points.points if isinstance(points, PointCloud) else points

        concat_feats, _, _ = self.pointnet(x)
        logits = self.segmentation_head(concat_feats)
        return logits
