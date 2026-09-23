from dataclasses import dataclass

import torch
from torch import nn

from pcp.core.PointCloud import PointCloud
from pcp.nn.PointNet2.SetAbstraction import SetAbstraction


@dataclass(frozen=True)
class SALayerConfig:
    n_points: int | None
    radius: float | list[float] | None
    n_samples: int | list[int] | None
    out_channels: list[int] | list[list[int]]
    group_all: bool = False


class PointNet2(nn.Module):
    def __init__(
        self,
        layers: list[SALayerConfig] | None = None,
        input_feature_channels: int = 0,
    ):
        super().__init__()
        if layers is None:
            layers = [
                SALayerConfig(512, 0.2, 32, [64, 128]),
                SALayerConfig(128, 0.4, 64, [128, 256]),
                SALayerConfig(None, None, None, [512, 1024], group_all=True),
            ]

        self.input_feature_channels = input_feature_channels
        self.sa_layers = nn.ModuleList()
        feature_channels = input_feature_channels

        for index, layer_config in enumerate(layers):
            layer = SetAbstraction(
                n_points=layer_config.n_points,
                radius=layer_config.radius,
                n_samples=layer_config.n_samples,
                in_channels=3 + feature_channels,
                out_channels=layer_config.out_channels,
                group_all=layer_config.group_all,
            )
            self.sa_layers.append(layer)
            feature_channels = layer.out_channels

        self.out_channels = feature_channels

    def forward(self, points: PointCloud | torch.Tensor):
        points = points.points if isinstance(points, PointCloud) else points

        expected_channels = 3 + self.input_feature_channels

        feats = points[..., 3:] if self.input_feature_channels else None
        points = points[..., :3]
        for layer in self.sa_layers:
            points, feats = layer(points, feats)
        return points, feats


def main():
    model = PointNet2()
    points = torch.rand(2, 1024, 3)
    sampled_points, features = model(points)

    print("Sampled points:", sampled_points.shape)
    print("Features:", features.shape)


if __name__ == "__main__":
    main()
