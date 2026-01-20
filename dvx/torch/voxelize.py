import functools
import math
import torch
from typing import Any, Optional, Tuple

import dvx_ext

class VoxelizeFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, grid_shape: Tuple[int, int] | Tuple[int, int, int], vertices: torch.Tensor, indices: torch.Tensor, method: str, primal_params: dict, backward_params: dict):
        dtype  = vertices.dtype
        device = vertices.device

        primal_func = None
        if method == 'mc':            
            primal_func = dvx_ext.voxelize_mc_f32 if dtype == torch.float32 else dvx_ext.voxelize_mc_f64
        elif method == 'cf':
            primal_func = dvx_ext.voxelize_cf_f32 if dtype == torch.float32 else dvx_ext.voxelize_cf_f64
        else:
            raise RuntimeError(f"Invalid method '{method}'.")

        primal_func = functools.partial(primal_func, **primal_params)

        occupancy = torch.zeros(grid_shape, dtype=vertices.dtype, device=device)
        primal_func(vertices, indices, occupancy)

        ctx.save_for_backward(vertices, indices, occupancy)

        ctx.method          = method
        ctx.primal_params   = primal_params
        ctx.backward_params = backward_params

        return occupancy
    
    @staticmethod
    def backward(ctx: Any, δoccupancy: torch.Tensor):
        vertices, indices, occupancy = ctx.saved_tensors # Unused
        method = ctx.method

        dtype  = vertices.dtype
        device = vertices.device

        backward_func = None
        if method == 'mc':            
            backward_func = dvx_ext.voxelize_backward_mc_f32 if dtype == torch.float32 else dvx_ext.voxelize_backward_mc_f64
        elif method == 'cf':
            backward_func = dvx_ext.voxelize_backward_cf_f32 if dtype == torch.float32 else dvx_ext.voxelize_backward_cf_f64
        else:
            raise RuntimeError(f"Invalid method '{method}'.")

        backward_func = functools.partial(backward_func, **ctx.backward_params)

        δvertices = torch.zeros_like(vertices, dtype=dtype, device=device)
        backward_func(vertices, indices, occupancy, δvertices, δoccupancy)

        return None, δvertices, None, None, None, None

def voxelize(n: int, vertices: torch.Tensor, indices: torch.Tensor, method: str = 'auto', filter_radius: Optional[float] = None, **kwargs):
    """
    Voxelize a polygon or triangle mesh inside the [-1,1]^3 cube.

    Parameters
    ----------
    n : int
        Shape of the voxel grid, which is (n,n) for 2D and (n,n,n) for 3D.
    vertices : torch.Tensor
        Vertices of the 2D polygon as (V,2) or the 3D triangle mesh as (V,3).
    indices : torch.Tensor
        Index array, in 2D with shape (F,2) and in 3D with shape (F,3)
    method : str
        The integration method, one of ['mc', 'cf', 'auto']. Default: 'auto'.
    filter_radius : Optional[float]
        Radius of the filter, by default the size of half a voxel.

    Returns
    -------
    torch.Tensor
        The occupancy as an array of shape (H,W) or (D,H,W).
    """

    dim = vertices.shape[1]
    grid_shape = [n]*dim

    half_voxel_size = 0.5 * (2 / n)
    if filter_radius is None:
        filter_radius = half_voxel_size
    elif method == 'cf' and not math.isclose(half_voxel_size, filter_radius):
        print(f"Warning: Integration mode 'cf' implicitly assumes a box filter the size of a voxel (={half_voxel_size}). The given filter radius (={filter_radius}) has no effect.")

    if method == 'auto':
        method = 'cf' if math.isclose(half_voxel_size, filter_radius) else 'mc'    

    primal_params = {}
    backward_params = {}
    if method == 'mc':
        primal_params['num_samples_per_voxel'] = kwargs.get('num_samples_per_voxel', 16)
        primal_params['filter_radius']         = filter_radius
        backward_params['num_samples_per_simplex'] = kwargs.get('num_samples_per_simplex', 64)
        backward_params['filter_radius']           = filter_radius

    occupancy = VoxelizeFunc.apply(grid_shape, vertices, indices, method, primal_params, backward_params)

    # Convert from [depth,height,width] -> [width,height,depth]
    # TODO: Use [depth,height,width] as canonical result
    if method == 'mc' and dim == 3:
        occupancy = occupancy.T 

    return occupancy