import torch
from torch import nn

from pcp.core.PointCloud import PointCloud
from pcp.grouping import ball_query
from pcp.nn.PointNet.PointNet import PointNetBackbone
from pcp.sampling import fps


class SetAbstraction(nn.Module):
    def __init__(
        self,
        n_points: int,
        radius: float | list[float] | tuple[float, ...],
        n_samples: int | list[int] | tuple[int, ...],
        in_channels: int,
        out_channels: list[int] | list[list[int]] | list[list[list[int]]],
    ):
        super().__init__()
        self.n_points = n_points
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

    def forward(
        self,
        points: PointCloud | torch.Tensor,
        feats: PointCloud | torch.Tensor | None = None,
    ):
        points = points.points if isinstance(points, PointCloud) else points
        feats = feats.points if isinstance(feats, PointCloud) else feats

        # Sampling
        sampled_points = fps(points, self.n_points).points
        grouped_feats_list = []

        # Grouping for both SSG and MSG
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
                grouped_feats = torch.cat([grouped_points, grouped_feats], dim=-1)
            else:
                grouped_feats = grouped_points
            grouped_feats_list.append(grouped_feats)

        # Apply PointNet to each group and concatenate the features
        sampled_feats_list = []
        for grouped_feats, pointnet in zip(grouped_feats_list, self.pointnets):
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
