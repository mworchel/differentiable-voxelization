#include "allocator.hpp"

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

DefaultAllocator& DefaultAllocator::instance()
{
    static DefaultAllocator _instance;
    return _instance;
}

}