#pragma once

#include <cassert>
#include <cstdint>

class Bitset
{
public:
    using Integer = int64_t;

    static constexpr uint8_t const NumBitsPerElement = 8 * sizeof(Integer);

    Bitset(uint64_t const size)
    {
        m_num_elements = (size + NumBitsPerElement - 1) / NumBitsPerElement;
        m_elements     = new Integer[m_num_elements];
    }

    virtual ~Bitset()
    {
        delete[] m_elements;
    }

    void set(uint64_t index)
    {
        uint32_t const element_index = index / NumBitsPerElement;
        uint32_t const bit_index     = index % NumBitsPerElement;
        
        // Check oob
        assert(element_index < m_num_elements);
        assert(bit_index < NumBitsPerElement);

        m_elements[element_index] |= (1 << bit_index);
    }

    bool is_set(uint64_t index)
    {
        uint32_t const element_index = index / NumBitsPerElement;
        uint32_t const bit_index     = index % NumBitsPerElement;
        
        // Check oob
        assert(element_index < m_num_elements);
        assert(bit_index < NumBitsPerElement);

        return m_elements[element_index] & (1 << bit_index);
    }

private:
    uint32_t m_num_elements;
    Integer* m_elements;
};