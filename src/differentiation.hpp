#pragma once

#include <type_traits>

namespace dvx
{

enum class DifferentiationMode
{
    Forward,
    Backward
};

template<typename T, DifferentiationMode Mode>
using dIn = typename std::conditional_t<Mode == DifferentiationMode::Backward, T, T const>;

template<typename T, DifferentiationMode Mode>
using dOut = typename std::conditional_t<Mode == DifferentiationMode::Backward, T const, T>;

}