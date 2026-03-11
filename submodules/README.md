# CUDA Extensions for Auto3R

This directory contains three CUDA extensions required for Auto3R to function.

## Modules

### 1. diff-gaussian-rasterization
**Source**: [3D Gaussian Splatting](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
**License**: See `diff-gaussian-rasterization/LICENSE.md`
**Description**: Standard differentiable Gaussian rasterizer from the original 3DGS paper.

**Outputs**:
- `rendered_image`: RGB image
- `radii`: Screen-space radii of Gaussians

### 2. modified-diff-gaussian-rasterization
**Source**: [FisherRF](https://github.com/JiangWenPL/FisherRF)
**License**: See `modified-diff-gaussian-rasterization/LICENSE.md`
**Description**: Modified version of the Gaussian rasterizer that outputs additional information for uncertainty quantification.

**Outputs**:
- `rendered_image`: RGB image
- `depth`: Depth map
- `radii`: Screen-space radii of Gaussians
- `pixel_gaussian_counter`: Number of Gaussians contributing to each pixel

**Key modifications**:
- Added depth accumulation in CUDA kernel
- Added per-pixel Gaussian counter
- Modified return signature in Python bindings

### 3. simple-knn
**Source**: [3D Gaussian Splatting](https://gitlab.inria.fr/bkerbl/simple-knn)
**License**: See `simple-knn/LICENSE.md` (if available)
**Description**: Fast CUDA-based K-nearest neighbors for point cloud initialization.

## Installation

Install all three modules:

```bash
pip install submodules/diff-gaussian-rasterization
pip install submodules/modified-diff-gaussian-rasterization
pip install submodules/simple-knn
```

## Requirements

- CUDA 11.8+ (must match PyTorch CUDA version)
- `nvcc` compiler in PATH
- PyTorch 2.0.1+

## Acknowledgements

These modules are based on the excellent work from:
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) by Inria GRAPHDECO
- [FisherRF](https://github.com/JiangWenPL/FisherRF) for the modified rasterizer

Please cite the original papers if you use these modules in your research.
