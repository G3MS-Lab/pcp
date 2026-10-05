# pcp

Small point-cloud utilities built with PyTorch.

The package currently includes:

- a `PointCloud` tensor wrapper
- CSV loading
- farthest-point and random sampling
- k-nearest-neighbor search
- Chamfer distance
- inverse-distance weighted interpolation
- PointNet classification and segmentation models

## Installation

Install the project in editable mode:

```bash
pip install -e .
```

Using `uv`:

```bash
uv pip install -e .
```

Python 3.10 or newer is required.

## Basic usage

```python
import torch

from pcp import PointCloud
from pcp.distance import chamfer
from pcp.grouping import knn
from pcp.sampling import fps, random

cloud = PointCloud(torch.rand(1024, 3))  # (N, C)

sampled = fps(cloud, k=256)
random_sampled = random(cloud, k=256)

center = torch.tensor([0.5, 0.5, 0.5])
neighbors = knn(cloud, center, k=16)

distance = chamfer(sampled, random_sampled)
```

Interpolate scalar or vector values at new coordinates with IDW:

```python
from pcp.interpolation import IDW

known_points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
known_values = torch.tensor([10.0, 20.0, 30.0])
query_points = torch.tensor([[0.25, 0.25]])

interpolated = IDW(
    known_points,
    known_values,
    query_points,
    alpha=2.0,
    n_neighbors=3,
)
```

Set `n_neighbors=None` to use every known point. IDW also supports batched
coordinates `(B, N, C)` and vector values `(B, N, F)`.

Load a point cloud from CSV:

```python
from pcp.io import read_csv

cloud = read_csv("points.csv")
```

## PointNet

PointNet models accept batched tensors in `(B, N, C)` format:

- `B`: batch size
- `N`: number of points
- `C`: number of input channels

```python
import torch

from pcp.nn.PointNet import PointNetCls, PointNetSeg

points = torch.randn(4, 1024, 3)  # (B, N, C)

classifier = PointNetCls(
    in_channels=3,
    num_classes=10,
    use_tnet=True,
)
class_logits = classifier(points)  # (4, 10)

segmenter = PointNetSeg(
    in_channels=3,
    num_classes=4,
    use_tnet=True,
)
point_logits = segmenter(points)  # (4, 4, 1024)
```

An unbatched `PointCloud` with shape `(N, C)` can be prepared for PointNet with:

```python
batched_points = cloud.points.unsqueeze(0)  # (1, N, C)
```

Segmentation logits use `(B, num_classes, N)`, which can be passed directly to PyTorch's `CrossEntropyLoss` with targets shaped `(B, N)`.

## Source layout

```text
src/pcp/
├── core/       # PointCloud
├── distance/   # Chamfer distance
├── grouping/   # k-nearest neighbors
├── interpolation/ # inverse-distance weighting
├── io/         # CSV loading
├── nn/         # PointNet models
└── sampling/   # FPS and random sampling
```

## RepKPU point-cloud upsampling

`RepKPU` follows the official architecture: a three-stage PointTransformer
encoder, deformable RepKPoints Extraction Modules (REM), a KP-Queries
Generation Module (KGM), and cross-attention that turns queries into
displacements. The full decoder uses three REM blocks; `simple=True` uses one
REM block and the simple attention/skip path. The forward result is a tuple
`(upsampled_points, regularization_loss)`.

This library implementation uses tiled PyTorch neighborhood lookup rather
than the reference repository's compiled `pointops` CUDA kernels. Its distance
work remains quadratic in point count, though temporary distance memory is
bounded by `B * chunk_size * N`; the portable path is intended for patch-sized
inputs. The encoder and decoder reuse `pcp.grouping.knn` and
`pcp.grouping.ball_query`; both utilities retain their original return values
and add optional neighbor-index/validity-mask returns. It does not require
`pointops`, `einops`, or compiled Chamfer3D.

Inputs are batched coordinates with shape `(B, N, 3)`. The model returns
coordinates of shape `(B, N * up_rate, 3)`, with each input point followed by
its generated points. For example:

```python
import torch

from pcp.nn.RepKPU import RepKPU, RepKPUConfig

config = RepKPUConfig(
    up_rate=4,
    encoder_dim=32,
    out_dim=64,
    k=16,
    in_dim=64,
    kp_dim=64,
    num_kernel_points=15,
    trans_dim=128,
    head_num=4,
    trans_num=3,
    simple=False,
)
model = RepKPU(config)
points = torch.rand(2, 256, 3)
upsampled, reg_loss = model(points)  # (2, 1024, 3), scalar loss
```

Configuration fields mirror the official PU1K options: encoder width and
neighbor count, kernel radii and limits, KP dimensions, upsampling rate,
attention width/depth, and the simple/full decoder switch. The scalar loss is
the sum of the official deformable-kernel fitting and repulsion terms; include
it with the task loss during training. Input and output coordinates are
`(B, N, 3)` and `(B, N * up_rate, 3)`.

The module structure and forward mechanisms follow the official source, but
this is not a bit-for-bit port: neighborhood search is tiled PyTorch instead
of `pointops`, and the fixed kernel layouts use a deterministic centered
Fibonacci sphere instead of `kernel_utils.load_kernels`. Consequently model
weights and reported results are not checkpoint-compatible with official
RepKPU. The reference training/evaluation scripts also require their datasets
and custom Chamfer3D extension. The official code is MIT licensed; its notice
is included with the RepKPU package code.
