#pragma once

#define DVX_DECLARE_FLAG_OPERATORS(FlagsType, BitsType)          \
                                                                 \
    constexpr FlagsType operator+(BitsType bit)                  \
    {                                                            \
        return static_cast<FlagsType>(bit);                      \
    }                                                            \
    constexpr FlagsType operator|(BitsType bit1, BitsType bit2)  \
    {                                                            \
        return +bit1 | +bit2;                                    \
    }                                                            \
    constexpr FlagsType operator|(FlagsType flags, BitsType bit) \
    {                                                            \
        return flags | +bit;                                     \
    }                                                            \
    constexpr FlagsType operator&(BitsType bit1, BitsType bit2)  \
    {                                                            \
        return +bit1 & +bit2;                                    \
    }                                                            \
    constexpr FlagsType operator&(FlagsType flags, BitsType bit) \
    {                                                            \
        return flags & +bit;                                     \
    }                                                            \
    constexpr FlagsType operator~(BitsType bit)                  \
    {                                                            \
        return ~static_cast<FlagsType>(bit);                     \
    }                                                            \
    constexpr bool has_flag(FlagsType flags, BitsType bit)       \
    {                                                            \
        return static_cast<bool>(flags & bit);                   \
    }
