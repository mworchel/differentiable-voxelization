#pragma once

/**
 * The same code is compiled for the CPU and the GPU,
 * so provide some defines to distinguish between
 * the versions during compilation.
 */
#ifdef __CUDACC__
#define IS_CUDA 1
#define HOST __host__
#define DEVICE __device__
#define CONSTANT __constant__
#define MAYBE_STD(func) ::func
#define MAYBE_CUDA(func) ::func
#define IF_NOT_CUDA(expr)
#define FORCE_INLINE __forceinline__ 
#else
#define IS_CUDA 0
#define HOST
#define DEVICE
#define CONSTANT
// Mark a function that, on the CPU, lives in the `std` namespace
#define MAYBE_STD(func) std::func
// Mark a function that is available only in CUDA
#define MAYBE_CUDA(func) ::cuda_replacement::func
#define IF_NOT_CUDA(expr) expr
#define FORCE_INLINE

#include <cmath>

namespace cuda_replacement
{
	template<typename Float>
	void sincos(Float v, Float* s, Float* c)
	{
		*s = ::std::sin(v);
		*c = ::std::cos(v);
	}
}
#endif