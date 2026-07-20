import functools
import numpy as np
from pathlib import Path
import pytest
import trimesh

import utils

import dvx_ext

TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_DIMENSIONS = [2, 3]
TEST_PRECISIONS = [32, 64]
TEST_RESOLUTIONS = [4, 16, 32]

def load_test_shape(dim: int):
    if dim == 3:
        mesh = trimesh.load_mesh(TEST_DATA_DIR / "hand.obj")
        v, f = np.array(mesh.vertices, dtype=np.float32), np.array(mesh.faces, dtype=np.int32)
        winding_number_range = [0, 1]
    elif dim == 2:
        v, f = utils.load_svg_as_linear_path(TEST_DATA_DIR / "twist.svg", path_idx=0)
        winding_number_range = [-1, 3]
    else:
        raise ValueError("Only 2D/3D shapes are available for testing.")
    
    return v, f, winding_number_range

def _test_primal(voxelize_func, dim: int, precision: int, resolution: int):
    dtype = getattr(np, f"float{precision}")

    v, f, winding_number_range = load_test_shape(dim)
    v = v.astype(dtype)

    output = np.zeros([resolution]*dim, dtype=dtype)
    voxelize_func(v, f, output)

    # Check if the result is within the expected winding number range (with a little bit of tolerance)
    eps = 1e-5
    assert np.all(output >= winding_number_range[0] - eps) and np.all(output <= winding_number_range[1] + eps)

@pytest.mark.parametrize("resolution", TEST_RESOLUTIONS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("dim", TEST_DIMENSIONS)
def test_primal_closed_form(dim: int, precision: int, resolution: int):
    voxelize_func = getattr(dvx_ext, f"voxelize_cf_f{precision}")
    _test_primal(voxelize_func, dim, precision, resolution)

@pytest.mark.parametrize("num_samples", [8, 16])
@pytest.mark.parametrize("resolution", TEST_RESOLUTIONS)
@pytest.mark.parametrize("precision", TEST_PRECISIONS)
@pytest.mark.parametrize("dim", TEST_DIMENSIONS)
def test_primal_monte_carlo(dim: int, precision: int, resolution: int, num_samples: int):
    voxelize_func = getattr(dvx_ext, f"voxelize_mc_f{precision}")
    voxelize_extra_args = {}
    voxelize_extra_args["num_samples_per_voxel"] = num_samples
    voxelize_extra_args["filter_radius"] = 0.5 * (2 / resolution) # half a voxel size
    _test_primal(functools.partial(voxelize_func, **voxelize_extra_args), dim, precision, resolution)

# Minimal tests for framework frontends (primal + gradient)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("framework", ["torch"])
def test_framework_minimal(framework: str, device: str):
    pytest.importorskip(framework)

    if framework == "torch":
        import torch
        import dvx.torch as dvx
        v, f, _ = load_test_shape(dim=3)
        v = torch.from_numpy(v).to(device=device)
        f = torch.from_numpy(f).to(device=device)
        v.requires_grad_(True)
    elif framework == "drjit":
        pass
    else:
        raise ValueError(f"Unknown framework '{framework}'.")
        
    n = 32
    voxels = dvx.voxelize(n, v, f)

    if framework == "torch":
        assert voxels.shape == (n, n, n)
        assert voxels.device.type == device
        voxels.sum().backward()
        assert v.grad is not None

@pytest.mark.parametrize("dim", TEST_DIMENSIONS)
@pytest.mark.parametrize("method", ["cf", "mc"])
@pytest.mark.parametrize("framework", ["torch"])
def test_coordinate_conventions(framework: str, method: str, dim: int):
    # Test coordinate conventions on the *framework-level*
    # (the internal conventions in low-level `dvx_ext` are currently inconsistent)

    pytest.importorskip(framework)

    n = 8 # Voxel grid resolution

    # Create test shapes (spanning only certain parts of the voxel grid)
    # TODO: Maybe avoid symmetry for the tests
    if dim == 2:
        v = np.array([[-1, -1], [0, 0], [1, -1]], dtype=np.float32)
        f = np.array([[0, 2], [2, 1], [1, 0]], dtype=np.int64)
    elif dim == 3:
        # Pyramid with the base in the xy-plane at z=-1 and the tip at (0, 0, 0)
        z_min = -1
        v = np.array([[-1, -1, z_min], [1, -1, z_min], [1, 1, z_min], [-1, 1, z_min], [0, 0, 0]], dtype=np.float32)
        f = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [2, 1, 0], [0, 3, 2]], dtype=np.int64)

    if framework == "torch":
        import torch
        import dvx.torch as dvx
        v = torch.from_numpy(v)
        f = torch.from_numpy(f)
    elif framework == "drjit":
        pass
    else:
        raise ValueError(f"Unknown framework '{framework}'.")

    voxels = dvx.voxelize(n, v, f, method=method)

    if framework == "torch":
        voxels = voxels.cpu().numpy()
    elif framework == "drjit":
        pass
    else:
        raise ValueError(f"Unknown framework '{framework}'.")

    if dim == 2:
        assert voxels[0, 0] > 0 # Bottom-left corner
        assert voxels[0, n - 1] > 0 # Bottom-right corner
        assert np.isclose(voxels[n - 1, 0], 0) # Top-left corner
        assert np.isclose(voxels[n - 1, n - 1], 0) # Top-right corner
        assert voxels[n//2 - 1, n//2 - 1] > 0 # Center
    if dim == 3:
        # Check if the corners in the xy-plane are occupied
        assert voxels[0, 0, 0] > 0 # Bottom-left corner
        assert voxels[0, 0, n - 1] > 0 # Bottom-right corner
        assert voxels[0, n - 1, 0] > 0 # Top-left corner
        assert voxels[0, n - 1, n - 1] > 0 # Top-right corner
        # Check if the top corners are unoccupied
        assert np.isclose(voxels[n - 1, n - 1, 0], 0) # Bottom-left corner
        assert np.isclose(voxels[n - 1, n - 1, n - 1], 0) # Bottom-right corner
        assert np.isclose(voxels[n - 1, 0, 0], 0) # Top-left corner
        assert np.isclose(voxels[n - 1, 0, n -1], 0) # Top-right corner
        # Check if the center is occupied
        assert voxels[n//2 - 1, n//2 - 1, n//2 - 1] > 0 # Center