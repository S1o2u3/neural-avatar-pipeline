# 📚 Neural Avatar Pipeline — API Documentation

## MeshGenerator
```python
from reconstruction.mesh_generator import MeshGenerator

model = MeshGenerator(resolution=128, device="cuda")
output = model(pts)        # pts: (N, 3) tensor
verts, faces = model.extract_mesh()
```

## SDFRenderer
```python
from rendering.sdf_renderer import SDFRenderer

renderer = SDFRenderer(n_samples=64, near=0.1, far=10.0)
result = renderer(rays_o, rays_d, sdf_fn, color_fn)
# result = {"rgb": ..., "depth": ..., "acc": ...}
```

## SpatialHashGrid
```python
from utils.spatial_hash import SpatialHashGrid

grid = SpatialHashGrid(voxel_size=0.01)
grid.insert(points, values)
results = grid.query(query_points)
print(grid.memory_usage())
```

## BatchedRaySampler
```python
from utils.ray_sampler import BatchedRaySampler

sampler = BatchedRaySampler(near=0.1, far=10.0, n_samples=64)
pts, z_vals = sampler.sample_rays(rays_o, rays_d)
rays_o, rays_d = sampler.get_rays(H, W, focal, c2w)
```

## DatasetLoader
```python
from ml_platform.dataset_loader import DatasetLoader

loader = DatasetLoader("neuman", "data/", split="train")
dataloader = loader.get_loader(batch_size=4)
loader.summary()
```

## Visualizer
```python
from utils.visualizer import Visualizer

viz = Visualizer(output_dir="outputs/")
viz.save_render(rgb, epoch=1)
viz.save_depth_map(depth, epoch=1)
viz.save_loss_curve(losses)
viz.save_mesh_visualization(verts, faces, epoch=1)
```
