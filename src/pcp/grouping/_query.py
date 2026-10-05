"""Shared tiled nearest-neighbor primitives for point grouping utilities."""

import torch


def nearest_indices(
    points: torch.Tensor,
    centroids: torch.Tensor,
    k: int,
    chunk_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return k nearest source indices and distances for batched coordinates."""
    k = min(k, points.shape[1])
    index_parts = []
    distance_parts = []
    for start in range(0, centroids.shape[1], chunk_size):
        query = centroids[:, start : start + chunk_size]
        distances = torch.cdist(query, points)
        distances, indices = distances.topk(k, dim=-1, largest=False)
        index_parts.append(indices)
        distance_parts.append(distances)
    return torch.cat(index_parts, dim=1), torch.cat(distance_parts, dim=1)


def index_points(
    points: torch.Tensor,
    indices: torch.Tensor,
    fill: float = 0.0,
) -> torch.Tensor:
    """Gather ``(B,N,C)`` points by ``(B,S[,K])`` indices; ``N`` means padding."""
    batch, count, channels = points.shape
    valid = indices < count
    safe = indices.clamp(max=count - 1)
    gathered = points.gather(
        1,
        safe.reshape(batch, -1).unsqueeze(-1).expand(-1, -1, channels),
    ).reshape(*indices.shape, channels)
    return gathered.masked_fill(~valid.unsqueeze(-1), fill)


def index_features(
    features: torch.Tensor,
    indices: torch.Tensor,
    fill: float = 0.0,
) -> torch.Tensor:
    """Gather ``(B,C,N)`` features into ``(B,C,S,K)`` with sentinel padding."""
    batch, channels, count = features.shape
    valid = indices < count
    safe = indices.clamp(max=count - 1)
    gathered = (
        features.transpose(1, 2)
        .gather(
            1,
            safe.reshape(batch, -1).unsqueeze(-1).expand(-1, -1, channels),
        )
        .reshape(*indices.shape, channels)
    )
    return gathered.permute(0, 3, 1, 2).masked_fill(~valid.unsqueeze(1), fill)
