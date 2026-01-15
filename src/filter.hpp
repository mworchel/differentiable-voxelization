#pragma once

#define _USE_MATH_DEFINES
#include <cmath>

#include "common.hpp"
#include "math.hpp"

namespace dvx
{

enum class FilterType
{
    Box = 0,
    Gaussian
};

template<typename Float>
struct Filter
{
    FilterType type;
    Float      radius; // Filter radius in world space

    template<unsigned int N>
    DEVICE Float eval(Vector<Float, N> const& difference) const
    {
        switch (type)
        {
        case FilterType::Box:
        {
            bool is_inside = true;
            for (unsigned int i = 0; i < N; ++i)
                is_inside &= MAYBE_STD(abs)(difference[i]) < radius;
            return Float(is_inside) / volume<N>();
        }
        case FilterType::Gaussian:
        {
            Float const           distance = norm(difference);
            constexpr Float const norm     = Float(0.5 * M_SQRT1_2 * M_2_SQRTPI);
            return Float(distance < radius) * norm * MAYBE_STD(exp)(-Float(0.5) * (distance * distance) / (2 * gaussian.stddev * gaussian.stddev));
        }
        default:
            return 0;
        }
    }

    template<unsigned int N>
    DEVICE Float volume() const
    {
        switch (type)
        {
        case FilterType::Box:
        {
            if constexpr (N == 2)
                return 4 * radius * radius;
            if constexpr (N == 3)
                return 8 * radius * radius * radius;
            return 0;
        }
        case FilterType::Gaussian:
        {
            return Float(1);
        }
        default:
            return 0;
        }
    }

    union
    {
        struct
        {
        } box;
        struct
        {
            Float stddev;
        } gaussian;
    };
};

} // namespace dvx