import torch
from torch import nn

from pcp.core.PointCloud import PointCloud
from pcp.grouping import ball_query
from pcp.nn.PointNet.PointNet import PointNetBackbone
from pcp.sampling import fps


class SetAbstraction(nn.Module):
    def __init__(
        self,
        n_points: int | None = 512,
        radius: float | list[float] | tuple[float, ...] | None = 0.2,
        n_samples: int | list[int] | tuple[int, ...] | None = 32,
        in_channels: int = 3,
        out_channels: list[int] | list[list[int]] | tuple[int, ...] = (64, 128),
        group_all: bool = False,
    ):
        super().__init__()
        self.n_points = n_points
        self.in_channels = in_channels
        self.group_all = group_all
        self.multi_scale = isinstance(radius, (list, tuple))

        if self.multi_scale:
            self.radius_list = list(radius)
            self.n_samples_list = list(n_samples)
            out_channels_list = [list(channels) for channels in out_channels]
        else:
            self.radius_list = [radius]
            self.n_samples_list = [n_samples]
            out_channels_list = [list(out_channels)]

        self.pointnets = nn.ModuleList()
        for channels in out_channels_list:
            if isinstance(channels[0], int):
                channels = [[64, 64], channels]
            self.pointnets.append(
                PointNetBackbone(
                    in_channels=in_channels,
                    out_channels=channels,
                    global_feats=True,
                )
            )
        self.out_channels = sum(pointnet.out_channels for pointnet in self.pointnets)

    def forward(
        self,
        points: PointCloud | torch.Tensor,
        feats: PointCloud | torch.Tensor | None = None,
    ):
        points = points.points if isinstance(points, PointCloud) else points
        feats = feats.points if isinstance(feats, PointCloud) else feats

        channels = points.shape[-1] + (feats.shape[-1] if feats is not None else 0)

        # group_all for last layer of PointNet++ (global feature)
        if self.group_all:
            batch_size, _, channels = points.shape
            sampled_points = torch.zeros(
                batch_size, 1, channels, device=points.device, dtype=points.dtype
            )
            grouped_points = points.unsqueeze(1)
            if feats is not None:
                grouped_feats = torch.cat(
                    [grouped_points, feats.unsqueeze(1)], dim=-1
                )
            else:
                grouped_feats = grouped_points
            grouped_feats_list = [grouped_feats]
        else:
            # Sampling and grouping for both SSG and MSG
            sampled_points = fps(points, self.n_points).points
            grouped_feats_list = []
            for radius, n_samples in zip(self.radius_list, self.n_samples_list):
                group_idx, grouped_points = ball_query(
                    radius, n_samples, points, sampled_points
                )
                grouped_points = grouped_points - sampled_points.unsqueeze(2)
                if feats is not None:
                    batch_size = feats.shape[0]
                    batch_idx = torch.arange(batch_size, device=feats.device)
                    batch_idx = batch_idx.view(batch_size, 1, 1)
                    grouped_feats = feats[batch_idx, group_idx]
                    grouped_feats = torch.cat(
                        [grouped_points, grouped_feats], dim=-1
                    )
                else:
                    grouped_feats = grouped_points
                grouped_feats_list.append(grouped_feats)

        # Apply PointNet to each group and concatenate the features
        sampled_feats_list = []
        for pointnet, grouped_feats in zip(self.pointnets, grouped_feats_list):
            batch_size, n_points, n_samples, channels = grouped_feats.shape
            grouped_feats = grouped_feats.reshape(
                batch_size * n_points, n_samples, channels
            )
            sampled_feats, _ = pointnet(grouped_feats)
            sampled_feats_list.append(
                sampled_feats.reshape(batch_size, n_points, -1)
            )

        sampled_feats = torch.cat(sampled_feats_list, dim=-1)
        return sampled_points, sampled_feats


def main():
    points = torch.rand(2, 1024, 3)
    sa_module = SetAbstraction()

    sampled_points, sampled_feats = sa_module(points)
    print("Sampled points:", sampled_points.shape)
    print("Features:", sampled_feats.shape)


if __name__ == "__main__":
    main()
