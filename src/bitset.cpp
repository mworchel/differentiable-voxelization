#include "bitset.hpp"

#include <cstring>

namespace dvx
{

bool create_bitset(Bitset* bitset, uint64_t const size, Allocator& allocator)
{
    *bitset = Bitset{
        .num_elements = static_cast<uint32_t>((size + Bitset::NumBitsPerElement - 1) / Bitset::NumBitsPerElement),
    };
    bitset->elements = allocator.allocate<Bitset::Integer>(bitset->num_elements);
    allocator.memset(bitset->elements, 0, sizeof(Bitset::Integer) * bitset->num_elements);
    return true;
}

void free_bitset(Bitset* bitset, Allocator& allocator)
{
    allocator.deallocate(bitset->elements);
    bitset->elements     = nullptr;
    bitset->num_elements = 0;
}

} // namespace dvx