from pathlib import Path
from typing import Sequence
import numpy as np
import torch
import laspy
from pcp.core.PointCloud import PointCloud


def read_laz(
    path: str | Path,
    extra_dims: Sequence[str] | None = None,
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
    return PointCloud(tensor_data)
	