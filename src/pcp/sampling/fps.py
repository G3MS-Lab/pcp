import torch

from pcp.core.PointCloud import PointCloud


def fps(
    points: PointCloud | torch.Tensor,
    k: int,
    return_indices: bool = False,
):
    points = points.points if isinstance(points, PointCloud) else points
    unbatched = points.ndim == 2
    if unbatched:
        points = points.unsqueeze(0)

    batch_size, num_points, _ = points.shape
    indices = torch.zeros(batch_size, k, dtype=torch.long, device=points.device)
    distances = torch.full(
        (batch_size, num_points),
        float("inf"),
        device=points.device,
        dtype=points.dtype,
    )
    current_idx = torch.zeros(batch_size, dtype=torch.long, device=points.device)
    batch_idx = torch.arange(batch_size, device=points.device)

    for i in range(k):
        indices[:, i] = current_idx
        current_point = points[batch_idx, current_idx].unsqueeze(1)
        dist = torch.sum((points - current_point) ** 2, dim=-1)
        distances = torch.minimum(distances, dist)
        current_idx = torch.argmax(distances, dim=-1)

    sampled_points = points[batch_idx.unsqueeze(1), indices]
    if unbatched:
        sampled_points = sampled_points.squeeze(0)
        indices = indices.squeeze(0)

    sampled_points = PointCloud(sampled_points)
    if return_indices:
        return sampled_points, indices
    return sampled_points
