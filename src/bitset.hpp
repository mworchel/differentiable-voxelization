#pragma once

#include <cstdint>

#include "allocator.hpp"
#include "common.hpp"

#if !IS_CUDA
#include <cassert>
#endif

namespace dvx
{

struct Bitset
{
    using Integer = int64_t;

    static constexpr uint8_t const NumBitsPerElement = 8 * sizeof(Integer);

    HOST DEVICE inline void set(uint64_t index)
    {
        uint32_t const element_index = index / NumBitsPerElement;
        uint8_t const  bit_index     = index % NumBitsPerElement;

#if !IS_CUDA
        // Check oob
        assert(element_index < num_elements);
        assert(bit_index < NumBitsPerElement);
#endif

        elements[element_index] |= (1 << bit_index);
    }

    HOST DEVICE inline bool is_set(uint64_t index)
    {
        uint32_t const element_index = index / NumBitsPerElement;
        uint8_t const  bit_index     = index % NumBitsPerElement;

#if !IS_CUDA
        // Check oob
        assert(element_index < num_elements);
        assert(bit_index < NumBitsPerElement);
#endif

        return elements[element_index] & (1 << bit_index);
    }

    uint32_t num_elements;
    Integer* elements;
};

// NOTE: Using create/free makes the bitset trivially copyable

bool create_bitset(Bitset* bitset, uint64_t const size, Allocator& allocator = DefaultAllocator::instance());

void free_bitset(Bitset* bitset, Allocator& allocator = DefaultAllocator::instance());

} // namespace dvx