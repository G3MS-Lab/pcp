import torch
from torch import nn

from pcp.core.PointCloud import PointCloud
from pcp.nn.PointNet.PointNet import PointNetBackbone

class PointNetCls(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        use_tnet: bool = False,
        out_channels: list[list[int]] | tuple[tuple[int, ...], tuple[int, ...]] = (
            (64, 64), (64, 128, 1024)
        ),
        classifier_channels: list[int] | tuple[int, ...] = (512, 256),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.use_tnet = use_tnet

        self.pointnet = PointNetBackbone(
            in_channels=in_channels,
            out_channels=out_channels,
            global_feats=True,
            use_tnet=use_tnet,
        )
        layers = []
        last_channels = self.pointnet.out_channels
        for channels in classifier_channels:
            layers.extend((
                nn.Linear(last_channels, channels, bias=False),
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ))
            last_channels = channels
        layers.append(nn.Linear(last_channels, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, points: PointCloud | torch.Tensor):
        x = points.points if isinstance(points, PointCloud) else points

        global_feats, _ = self.pointnet(x)
        logits = self.classifier(global_feats)
        return logits
