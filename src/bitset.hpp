#pragma once

#include <cstdint>

#include "allocator.hpp"
#include "common.hpp"

namespace dvx
{

struct Bitset
{
    using Integer = int64_t;

    static constexpr uint8_t const NumBitsPerElement = 8 * sizeof(Integer);

    void set(uint64_t index);

    bool is_set(uint64_t index);

    uint32_t num_elements;
    Integer* elements;
};

// NOTE: Using create/free makes the bitset trivially copyable

bool create_bitset(Bitset* bitset, uint64_t const size, Allocator& allocator = DefaultAllocator::instance());

void free_bitset(Bitset* bitset, Allocator& allocator = DefaultAllocator::instance());

} // namespace dvx