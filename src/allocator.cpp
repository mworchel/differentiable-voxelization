#include "allocator.hpp"

#include <cstring>

namespace dvx
{

void* DefaultAllocator::allocate(uint64_t byte_size)
{
    return new unsigned char[byte_size];
}

void* DefaultAllocator::allocate(uint64_t byte_size, uint64_t alignment)
{
    // TODO
    return nullptr;
}

void DefaultAllocator::deallocate(void* ptr)
{
    delete[] ptr;
}

void DefaultAllocator::memset(void* ptr, int value, uint64_t byte_size)
{
    std::memset(ptr, value, byte_size);
}

DefaultAllocator& DefaultAllocator::instance()
{
    static DefaultAllocator _instance;
    return _instance;
}

}