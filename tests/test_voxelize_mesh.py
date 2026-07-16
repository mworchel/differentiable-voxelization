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
        v, f = utils.load_svg_as_linear_path(TEST_DATA_DIR / "twist.svg", 0)
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
    assert np.all(output >= winding_number_range[0] - 1e-6) and np.all(output <= winding_number_range[1] + 1e-6)

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
    voxelize_extra_args["filter_radius"] = 0.5 * (2 / resolution) # half a voxel filter size
    _test_primal(functools.partial(voxelize_func, **voxelize_extra_args), dim, precision, resolution)
