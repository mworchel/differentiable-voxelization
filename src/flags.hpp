#pragma once

#include "common.hpp"

#define DVX_DECLARE_FLAG_OPERATORS(FlagsType, BitsType)                 \
                                                                        \
    HOST DEVICE constexpr FlagsType operator+(BitsType bit)                  \
    {                                                                   \
        return static_cast<FlagsType>(bit);                             \
    }                                                                   \
    HOST DEVICE constexpr FlagsType operator|(BitsType bit1, BitsType bit2)  \
    {                                                                   \
        return +bit1 | +bit2;                                           \
    }                                                                   \
    HOST DEVICE constexpr FlagsType operator|(FlagsType flags, BitsType bit) \
    {                                                                   \
        return flags | +bit;                                            \
    }                                                                   \
    HOST DEVICE constexpr FlagsType operator&(BitsType bit1, BitsType bit2)  \
    {                                                                   \
        return +bit1 & +bit2;                                           \
    }                                                                   \
    HOST DEVICE constexpr FlagsType operator&(FlagsType flags, BitsType bit) \
    {                                                                   \
        return flags & +bit;                                            \
    }                                                                   \
    HOST DEVICE constexpr FlagsType operator~(BitsType bit)                  \
    {                                                                   \
        return ~static_cast<FlagsType>(bit);                            \
    }                                                                   \
    HOST DEVICE constexpr bool has_flag(FlagsType flags, BitsType bit)       \
    {                                                                   \
        return static_cast<bool>(flags & bit);                          \
    }
