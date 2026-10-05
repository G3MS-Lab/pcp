import torch

from pcp.core.PointCloud import PointCloud
from pcp.grouping._query import index_points, nearest_indices


def knn(
    points: PointCloud | torch.Tensor,
    centroids: PointCloud | torch.Tensor,
    k: int,
    return_indices: bool = False,
):
    points = points.points if isinstance(points, PointCloud) else points
    centroids = centroids.points if isinstance(centroids, PointCloud) else centroids

    unbatched = points.ndim == 2 and centroids.ndim <= 2
    if centroids.ndim == 1:
        centroids = centroids.unsqueeze(0)

    if points.ndim == 2:
        points = points.unsqueeze(0)
    if centroids.ndim == 2:
        centroids = centroids.unsqueeze(0)
    indices, _ = nearest_indices(points, centroids, k)
    result = index_points(points, indices)

    if unbatched:
        result = result.squeeze(0)
        indices = indices.squeeze(0)

    result = PointCloud(result)
    return (result, indices) if return_indices else result
