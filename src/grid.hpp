#pragma once

#include <cmath>

#include "math.hpp"

namespace dvx
{

template<typename Float, unsigned int N>
inline void get_grid_support(Point<Float, N> const& query_coord, Vector<Float, N> const& extent, int32_t min_coord[N], int32_t max_coord[N])
{
    for (unsigned int i = 0; i < N; ++i)
    {
        min_coord[i] = query_coord[i] - extent[i];
        max_coord[i] = query_coord[i] + extent[i];
    }
}

template<typename Float, unsigned int N>
inline Point<Float, N> point_to_grid_coord(Point<Float, N> const& x, Vector<Float, N> const& voxel_size)
{
    Point<Float, N> const grid_origin(Float(-1));
    return (x - grid_origin) / voxel_size + Point<Float,N>(Float(0));
}

template<unsigned int N>
inline Point<uint32_t, N> linear_index_to_coords(uint64_t const index, Vector<uint32_t, N> const& grid_size)
{
    if constexpr (N == 2)
        return Point<uint32_t, N>(
            /*x=*/index % grid_size[0], 
            /*y=*/index / grid_size[0]);
    else if constexpr (N == 2)
        return Point<uint32_t, N>(
            /*x=*/(index % (grid_size[0] * grid_size[1])) % grid_size[0],
            /*y=*/(index % (grid_size[0] * grid_size[1])) / grid_size[0],
            /*z=*/index / (grid_size[0] * grid_size[1])
        );
    else 
        return Point<uint32_t, N>();
}

}