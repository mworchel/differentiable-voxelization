#pragma once

#include <algorithm>
#include <limits>

#include "math.hpp"

namespace dvx
{

template<typename Float, unsigned int N>
struct AABB
{
    void extend(Point<Float, N> const& p)
    {
        for (unsigned int i = 0; i < N; ++i)
        {
            min[i] = std::min(min[i], p[i]);
            max[i] = std::max(max[i], p[i]);
        }
    }

    Point<Float, N> min{std::numeric_limits<Float>::infinity()};
    Point<Float, N> max{-std::numeric_limits<Float>::infinity()};
};

DECLARE_FLOAT_DEFINES(AABB, 2);
DECLARE_FLOAT_DEFINES(AABB, 3);

}