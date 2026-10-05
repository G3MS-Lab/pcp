from pathlib import Path
from typing import Sequence
import numpy as np
import torch
import laspy

from pcp.core.PointCloud import PointCloud


def read_laz(
    path: str | Path,
    extra_dims: Sequence[str] | None = None,
    label_dim: str | None = "classification",
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> PointCloud:
    
    with laspy.open(path) as reader:
        las = reader.read()

    xyz = np.stack([las.x, las.y, las.z], axis=-1)

    if extra_dims:
        extra_arrays = [np.asarray(las[dim]) for dim in extra_dims]
        data = np.column_stack([xyz, *extra_arrays])
    else:
        data = xyz

    tensor_data = torch.as_tensor(data, dtype=dtype, device=device)

    labels_tensor = None
    if label_dim and label_dim in las.point_format.dimension_names:
        labels_tensor = torch.as_tensor(
            np.asarray(las[label_dim]), dtype=torch.long, device=device
        )

    return PointCloud(tensor_data, labels=labels_tensor)


def write_laz(
    path: str | Path,
    points: PointCloud | torch.Tensor,
    labels: torch.Tensor | np.ndarray | None = None,
    point_format: int = 0,
) -> None:
    
    tensor = points.points if isinstance(points, PointCloud) else points
    if labels is None and isinstance(points, PointCloud):
        labels = points.labels

    arr = tensor.detach().cpu().numpy()

    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])

    xyz = arr[:, :3]

    fmt = 1 if (labels is not None or point_format > 0) else 0
    header = laspy.LasHeader(point_format=fmt, version="1.2")
    header.offsets = np.min(xyz, axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])

    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]

    if labels is not None:
        if isinstance(labels, torch.Tensor):
            labels_arr = labels.detach().cpu().numpy()
        else:
            labels_arr = np.asarray(labels)
        las.classification = labels_arr.reshape(-1).astype(np.uint8)

    las.write(path)
	