import torch
import torch.nn.functional as F
from torch import nn

from pcp.core.PointCloud import PointCloud
from pcp.nn.PointNet2.PointNet2 import PointNet2


class PointNet2Cls(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: PointNet2 | None = None,
        classifier_channels: list[int] | tuple[int, ...] = (512, 256),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.pointnet = backbone if backbone is not None else PointNet2()

        layers = []
        last_channels = self.pointnet.out_channels
        for channels in classifier_channels:
            layers.extend(
                (
                    nn.Linear(last_channels, channels),
                    nn.BatchNorm1d(channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                )
            )
            last_channels = channels
        layers.append(nn.Linear(last_channels, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, points: PointCloud | torch.Tensor):
        x = points.points if isinstance(points, PointCloud) else points
        _, point_feats = self.pointnet(x)
        global_feats = point_feats.max(dim=1).values
        logits = self.classifier(global_feats)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, global_feats.unsqueeze(1)


def main():
    model = PointNet2Cls(num_classes=40)
    points = torch.rand(2, 1024, 3)
    log_probs, features = model(points)

    print("Log probabilities:", log_probs.shape)
    print("Features:", features.shape)


if __name__ == "__main__":
    main()
