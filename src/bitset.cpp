#include "bitset.hpp"

#include <cassert>
#include <cstring>

namespace dvx
{

void Bitset::set(uint64_t index)
{
    uint32_t const element_index = index / NumBitsPerElement;
    uint8_t const  bit_index     = index % NumBitsPerElement;

    // Check oob
    assert(element_index < num_elements);
    assert(bit_index < NumBitsPerElement);

    elements[element_index] |= (1 << bit_index);
}

bool Bitset::is_set(uint64_t index)
{
    uint32_t const element_index = index / NumBitsPerElement;
    uint8_t const  bit_index     = index % NumBitsPerElement;

    // Check oob
    assert(element_index < num_elements);
    assert(bit_index < NumBitsPerElement);

    return elements[element_index] & (1 << bit_index);
}

bool create_bitset(Bitset* bitset, uint64_t const size, Allocator& allocator)
{
    *bitset = Bitset{
        .num_elements = static_cast<uint32_t>((size + Bitset::NumBitsPerElement - 1) / Bitset::NumBitsPerElement),
    };
    bitset->elements = allocator.allocate<Bitset::Integer>(bitset->num_elements);
    MAYBE_STD(memset)(bitset->elements, 0, sizeof(Bitset::Integer) * bitset->num_elements);
    return true;
}

void free_bitset(Bitset* bitset, Allocator& allocator)
{
    allocator.deallocate(bitset->elements);
    bitset->elements     = nullptr;
    bitset->num_elements = 0;
}

} // namespace dvx