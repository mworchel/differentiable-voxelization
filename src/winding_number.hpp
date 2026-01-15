#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdint>

#include "math.hpp"

namespace dvx
{

template<typename Float, unsigned int N>
Float generalized_winding_number(Float const* vertices, uint32_t const num_vertices,
                                 uint32_t const* simplices, uint32_t const num_simplices,
                                 Point<Float, N> const& x)
{
    Float occupancy = 0.f;
    for (uint32_t s = 0; s < num_simplices; ++s)
    {
        // Load the vertices and compute `e` vectors
        Vector<Float, N> e[N];
        for (int i = 0; i < N; ++i)
        {
            Point<Float, N> vi;
            for (int d = 0; d < N; ++d)
                vi[d] = vertices[N * simplices[N * s + i] + d];

            e[i] = normalize(vi - x);
        }

        Float const alpha = [&e]()
        {
            if constexpr (N == 2)
                return e[0].x() * e[1].y() - e[0].y() * e[1].x();
            else // N == 3
                return dot(cross(e[0], e[1]), e[2]);
        }();
        Float const beta = ((N == 2) ? dot(e[0], e[1]) : 1 + dot(e[0], e[1]) + dot(e[1], e[2]) + dot(e[2], e[0]));
        occupancy += atan2(alpha, beta);
    }

    Float normalization = (N == 2) ? 2 * M_PI : 0.5 * 4 * M_PI;

    return occupancy / normalization;
}

} // namespace dvx