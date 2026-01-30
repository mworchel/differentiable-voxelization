#pragma once

#include <cstdint>

namespace dvx
{

class Allocator
{
public:
    virtual void* allocate(uint64_t byte_size) = 0;

    virtual void* allocate(uint64_t byte_size, uint64_t alignment) = 0;

    virtual void deallocate(void* ptr) = 0;

    template<typename T>
    T* allocate(uint64_t size)
    {
        return reinterpret_cast<T*>(allocate(sizeof(T) * size));
    }
};

class DefaultAllocator : public Allocator
{
public:
    void* allocate(uint64_t byte_size) override;

    void* allocate(uint64_t byte_size, uint64_t alignment) override;
    
    void  deallocate(void* ptr) override;

    static DefaultAllocator& instance();
};

} // namespace dvx
    