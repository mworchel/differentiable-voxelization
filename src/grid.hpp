#pragma once

#include <cmath>
#include <type_traits>

#include "math.hpp"

namespace dvx
{

template<unsigned int N>
struct Extent
{
    template<typename = std::enable_if_t<(N > 0u)>>
    inline uint32_t width() const
    {
        return extents[0];
    }

    template<typename = std::enable_if_t<(N > 1u)>>
    inline uint32_t height() const
    {
        return extents[1];
    }

    template<typename = std::enable_if_t<(N > 2u)>>
    inline uint32_t depth() const
    {
        return extents[2];
    }

    inline uint32_t operator[](uint32_t i) const
    {
        return extents[i];
    }

    inline uint64_t num_elements() const
    {
        uint64_t count = 1;
        for (unsigned int i = 0; i < N; ++i)
            count *= extents[i];
        return count;        
    }

    uint32_t extents[N];
};

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
inline Vector<Float, N> get_voxel_size(Extent<N> const& grid)
{
    if constexpr (N == 2)
        return Vector2<Float>(
            Float(2) / grid.width(),
            Float(2) / grid.height());
    else if constexpr (N == 3)
        return Vector3<Float>(
            Float(2) / grid.width(),
            Float(2) / grid.height(),
            Float(2) / grid.depth());
    else
        return Vector<Float, N>(Float(0));
}

template<typename Float, unsigned int N>
inline Point<Float, N> point_to_grid_coord(Point<Float, N> const& x, Vector<Float, N> const& voxel_size)
{
    Point<Float, N> const grid_origin(Float(-1));
    return (x - grid_origin) / voxel_size + Point<Float, N>(Float(0));
}

template<unsigned int N>
inline Point<uint32_t, N> linear_index_to_coords(uint64_t const index, Extent<N> const& grid)
{
    if constexpr (N == 2)
        return Point<uint32_t, N>(
            /*x=*/index % grid.width(),
            /*y=*/index / grid.width());
    else if constexpr (N == 2)
        return Point<uint32_t, N>(
            /*x=*/(index % (grid.width() * grid.height())) % grid.width(),
            /*y=*/(index % (grid.width() * grid.height())) / grid.width(),
            /*z=*/index / (grid.width() * grid.height()));
    else
        return Point<uint32_t, N>();
}

template<unsigned int N>
inline uint64_t coords_to_linear_index(Point<int32_t, N> const& coords, Extent<N> const& grid)
{
    // NOTE: No out-of-bounds check

    if constexpr (N == 2)
        return grid.width() * coords.y() + coords.x(); // width * y + x
    else if constexpr (N == 3)
        return grid.width() * (grid.height() * coords[2] + coords[1]) + coords[0]; // width * (height * z +  y) + x
    else
        return 0u;
}

}