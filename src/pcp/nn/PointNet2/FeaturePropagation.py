import torch
from torch import nn

from pcp.core.PointCloud import PointCloud
from pcp.interpolation import IDW


class FeaturePropagation(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: list[int] | tuple[int, ...] = (128, 128, 128),
    ):
        super().__init__()

        layers = []
        for channels in out_channels:
            layers.extend(
                (
                    nn.Conv1d(in_channels, channels, kernel_size=1),
                    nn.BatchNorm1d(channels),
                    nn.ReLU(inplace=True),
                )
            )
            in_channels = channels

        self.mlp = nn.Sequential(*layers)
        self.out_channels = in_channels

    def forward(
        self,
        target_points: PointCloud | torch.Tensor,
        source_points: PointCloud | torch.Tensor,
        target_feats: torch.Tensor | None,
        source_feats: torch.Tensor,
    ):
        target_points = (
            target_points.points if isinstance(target_points, PointCloud) else target_points
        )
        source_points = (
            source_points.points if isinstance(source_points, PointCloud) else source_points
        )
        interpolated_feats = IDW(
            points=source_points,
            values=source_feats,
            queries=target_points,
            alpha=2.0,
            n_neighbors=min(3, source_points.shape[1]),
        )

        if target_feats is not None:
            interpolated_feats = torch.cat(
                [target_feats, interpolated_feats], dim=-1
            )

        return self.mlp(interpolated_feats.transpose(1, 2)).transpose(1, 2)


def main():
    fp = FeaturePropagation(in_channels=96, out_channels=(64, 32))
    target_points = torch.rand(2, 128, 3)
    source_points = torch.rand(2, 32, 3)
    target_feats = torch.rand(2, 128, 32)
    source_feats = torch.rand(2, 32, 64)

    features = fp(target_points, source_points, target_feats, source_feats)
    print("Features:", features.shape)


if __name__ == "__main__":
    main()
