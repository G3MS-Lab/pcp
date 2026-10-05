from dataclasses import dataclass
from typing import Any
import torch


@dataclass
class PointCloud:
    points: torch.Tensor
    labels: torch.Tensor | None = None

    def __init__(self, points: torch.Tensor, labels: torch.Tensor | None = None):
        self.points = points
        self.labels = labels

    def __getitem__(self, index: Any):
        if isinstance(index, tuple):
            return self.points[index]

        new_pos = self.points[index]
        if new_pos.ndim == 2:
            new_pos = new_pos.unsqueeze(0)

        return new_pos

    @property
    def shape(self):
        return self.points.shape

    def to_numpy(self):
        return self.points.numpy()

