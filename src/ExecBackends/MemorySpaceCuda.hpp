#ifndef HIOP_MEM_SPACE_CUDA
#define HIOP_MEM_SPACE_CUDA

#include <ExecSpace.hpp>

#ifdef HIOP_USE_CUDA

#include <cuda_runtime.h>
#include <cassert>

namespace hiop
{
//
// Allocator
//
template<typename T>
struct AllocImpl<MemBackendCuda, T>
{
  inline static T* alloc(MemBackendCuda& mb, const size_t& n)
  {
    T* p;
    auto err = cudaMalloc((void**)&p, n*sizeof(T));
    assert(cudaSuccess==err);
    return p;
  }
  inline static void dealloc(MemBackendCuda& mb, T* p)
  {
    auto err = cudaFree((void*)p);
    assert(cudaSuccess==err);
  }  
};

//
// Transfers
//
template<class EXECPOLDEST, class EXECPOLSRC, typename T>
struct TransferImpl<MemBackendCuda, EXECPOLDEST, MemBackendCuda, EXECPOLSRC, T>
{
  inline static bool do_it(T* p_dest,
                           ExecSpace<MemBackendCuda, EXECPOLDEST>& hwb_dest,
                           const T* p_src,
                           const ExecSpace<MemBackendCuda, EXECPOLSRC>& hwb_src,
                           const size_t& n)
  {
    return cudaSuccess == cudaMemcpy(p_dest, p_src, n*sizeof(T), cudaMemcpyDeviceToDevice);
  }
};

template<class EXECPOLDEST, class EXECPOLSRC, typename T>
struct TransferImpl<MemBackendCuda, EXECPOLDEST, MemBackendCpp, EXECPOLSRC, T>
{
  inline static bool do_it(T* p_dest,
                           ExecSpace<MemBackendCuda, EXECPOLDEST>& hwb_dest,
                           const T* p_src,
                           const ExecSpace<MemBackendCpp, EXECPOLSRC>& hwb_src,
                           const size_t& n)
  {
    return cudaSuccess == cudaMemcpy(p_dest, p_src, n*sizeof(T), cudaMemcpyHostToDevice);
  }
};

template<class EXECPOLDEST, class EXECPOLSRC, typename T>
struct TransferImpl<MemBackendCpp, EXECPOLDEST, MemBackendCuda, EXECPOLSRC, T>
{
  inline static bool do_it(T* p_dest,
                           ExecSpace<MemBackendCpp, EXECPOLDEST>& hwb_dest,
                           const T* p_src,
                           const ExecSpace<MemBackendCuda, EXECPOLSRC>& hwb_src,
                           const size_t& n)
  {
    return cudaSuccess == cudaMemcpy(p_dest, p_src, n*sizeof(T), cudaMemcpyDeviceToHost);
  }
};

} // end namespace hiop
#endif //HIOP_USE_CUDA
#endif //HIOP_MEM_SPACE_CUDA


