import torch

from pcp.core.PointCloud import PointCloud
from pcp.grouping._query import index_points, nearest_indices


def ball_query(
    radius: float,
    n_samples: int,
    points: PointCloud | torch.Tensor,
    centroids: PointCloud | torch.Tensor,
    return_mask: bool = False,
):
    points = points.points if isinstance(points, PointCloud) else points
    centroids = centroids.points if isinstance(centroids, PointCloud) else centroids

    unbatched = points.ndim == 2 and centroids.ndim == 2
    if points.ndim == 2:
        points = points.unsqueeze(0)
    if centroids.ndim == 2:
        centroids = centroids.unsqueeze(0)

    _, num_points, _ = points.shape
    k = min(n_samples, num_points)
    group_idx, distances = nearest_indices(points, centroids, k)
    valid = distances <= radius

    first_idx = group_idx[:, :, :1]
    first_idx = first_idx.repeat(1, 1, k)
    group_idx = torch.where(valid, group_idx, first_idx)

    if k < n_samples:
        padding = group_idx[:, :, :1].repeat(1, 1, n_samples - k)
        group_idx = torch.cat([group_idx, padding], dim=-1)
        valid = torch.cat([valid, torch.zeros_like(padding, dtype=torch.bool)], dim=-1)

    group_points = index_points(points, group_idx)

    if unbatched:
        group_idx = group_idx.squeeze(0)
        group_points = group_points.squeeze(0)
        valid = valid.squeeze(0)
    result = (group_idx, group_points)
    return (*result, valid) if return_mask else result
