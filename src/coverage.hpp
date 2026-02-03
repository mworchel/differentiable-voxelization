#pragma once

#include <cmath>
#include <cstdint>

#include "aabb.hpp"
#include "bitset.hpp"
#include "grid.hpp"
#include "math.hpp"

namespace dvx
{

template<typename Float, unsigned int N>
struct MarkBoundaryVoxelsOp
{
    void operator()(uint32_t const primitive_index)
    {
        AABB<Float, N> aabb;

        Point<uint32_t, N> const simplex(&simplices[N * primitive_index]);
        for (unsigned int i = 0; i < N; ++i)
        {
            Point<Float, N> const vi(&vertices[N * simplex[i]]);
            aabb.extend(vi);
        }

        //
        aabb.min -= Vector<Float, N>(shell_radius);
        aabb.max += Vector<Float, N>(shell_radius);

        Point<Float, N> const min_grid_coord = point_to_grid_coord(aabb.min, voxel_size);
        Point<Float, N> const max_grid_coord = point_to_grid_coord(aabb.max, voxel_size);

        int32_t min_coord[N];
        int32_t max_coord[N];
        for (unsigned int i = 0; i < N; ++i)
        {
            min_coord[i] = floor(min_grid_coord[i]);
            max_coord[i] = floor(max_grid_coord[i]);
        }

        // TODO: Use grid iterator
        if constexpr (N == 2)
        {
            for (int32_t ny = std::max(min_coord[1], 0); ny < std::min(static_cast<int32_t const>(grid.height()), max_coord[1] + 1); ++ny)
            {
                for (int32_t nx = std::max(min_coord[0], 0); nx < std::min(static_cast<int32_t const>(grid.width()), max_coord[0] + 1); ++nx)
                {
                    uint64_t const n_voxel_index = coords_to_linear_index<2>({nx, ny}, grid);
                    mask.set(n_voxel_index);
                }
            }
        }
        if constexpr (N == 3)
        {
            for (int32_t nz = std::max(min_coord[2], 0); nz < std::min(static_cast<int32_t const>(grid.depth()), max_coord[2] + 1); ++nz)
            {
                for (int32_t ny = std::max(min_coord[1], 0); ny < std::min(static_cast<int32_t const>(grid.height()), max_coord[1] + 1); ++ny)
                {
                    for (int32_t nx = std::max(min_coord[0], 0); nx < std::min(static_cast<int32_t const>(grid.width()), max_coord[0] + 1); ++nx)
                    {
                        uint64_t const n_voxel_index = coords_to_linear_index<3>({nx, ny, nz}, grid);
                        mask.set(n_voxel_index);
                    }
                }
            }
        }
    }

    Float const*           vertices;
    uint32_t const*        simplices;
    uint32_t const         num_simplices;
    Extent<N> const        grid;
    Vector<Float, N> const voxel_size;
    Float const            shell_radius;
    Bitset                 mask;
};

template<typename Float, unsigned int N>
void mark_boundary_voxels(Float const* vertices, uint32_t const* simplices, uint32_t const num_simplices,
                          Extent<N> const& grid, Float const shell_radius, Bitset* mask)
{
    static_assert(N == 2 || N == 3, "Dimension `N` must be 2 or 3.");

    MarkBoundaryVoxelsOp<Float, N> op{
        .vertices      = vertices,
        .simplices     = simplices,
        .num_simplices = num_simplices,
        .grid          = grid,
        .voxel_size    = get_voxel_size<Float>(grid),
        .shell_radius  = shell_radius,
        .mask          = *mask};

    uint32_t const num_primitives = num_simplices;
    for (uint32_t primitive_index = 0; primitive_index < num_primitives; ++primitive_index)
        op(primitive_index);
}

} // namespace dvx