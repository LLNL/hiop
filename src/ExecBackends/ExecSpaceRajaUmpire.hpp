#ifndef HIOP_MEM_SPACE_UMPIRE
#define HIOP_MEM_SPACE_UMPIRE

#include <ExecSpace.hpp>

#ifdef HIOP_USE_RAJA // can/should be HIOP_USE_UMPIRE

#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>

#include "ExecSpaceHost.hpp"

#ifdef HIOP_USE_HIP
#include "MemBackendHip.hpp"
#endif //HIOP_USE_HIP

#ifdef HIOP_USE_CUDA
#include "MemorySpaceCuda.hpp"
#endif //HIOP_USE_CUDA 

namespace hiop {

//
// Allocators
//
template<typename T>
struct AllocImpl<MemBackendUmpire, T>
{
  inline static T* alloc(MemBackendUmpire& mb, const size_t& n)
  {
    std::cout << "alloc_array umpire    loc " << mb.mem_space() << "\n";
    auto& resmgr = umpire::ResourceManager::getInstance();
    umpire::Allocator devalloc  = resmgr.getAllocator(mb.mem_space());
    return static_cast<T*>(devalloc.allocate(n*sizeof(T)));
  }
  inline static void dealloc(MemBackendUmpire& mb, T* p)
  {
    std::cout << "dealloc umpire    loc " << mb.mem_space() << "\n";
    auto& resmgr = umpire::ResourceManager::getInstance();
    umpire::Allocator devalloc  = resmgr.getAllocator(mb.mem_space());
    devalloc.deallocate(p);
  }  
};

//////////////////////////////////////////////////////////////////////////////////////////
// Transfers
//////////////////////////////////////////////////////////////////////////////////////////

template<class EXECPOLDEST, class EXECPOLSRC, typename T>
struct TransferImpl<MemBackendUmpire, EXECPOLDEST, MemBackendUmpire, EXECPOLSRC, T>
{
  inline static bool do_it(T* p_dest,
                           ExecSpace<MemBackendUmpire, EXECPOLDEST>& hwb_dest,
                           const T* p_src,
                           const ExecSpace<MemBackendUmpire, EXECPOLSRC>& hwb_src,
                           const size_t& n)
  {
    auto& rm = umpire::ResourceManager::getInstance();

    if(n>0) {
      double* src = const_cast<double*>(p_src);
      rm.copy(p_dest, src, n*sizeof(T));
    }
    return true;
  }
};

////////////////////////////////////////////////////////////////////////////////////////
// Transfers to/from Host C++ memory
////////////////////////////////////////////////////////////////////////////////////////
template<class EXECPOLDEST, class EXECPOLSRC, typename T>
struct TransferImpl<MemBackendCpp, EXECPOLDEST, MemBackendUmpire, EXECPOLSRC, T>
{
  inline static bool do_it(T* p_dest,
                           ExecSpace<MemBackendCpp, EXECPOLDEST>& hwb_dest,
                           const T* p_src,
                           const ExecSpace<MemBackendUmpire, MemBackendUmpire>& hwb_src,
                           const size_t& n)
  {
    if(hwb_src.mem_backend().is_host()) {
      std::memcpy(p_dest, p_src, n*sizeof(T));
    } else {
      assert(false && "Transfer BACKENDS(TO:Cpp-host,FROM:umpire) only supported with Umpire mem space host"); 
      return false;
    }
    return true;
  }
};

template<class EXECPOLDEST, class EXECPOLSRC, typename T>
struct TransferImpl<MemBackendUmpire, EXECPOLDEST, MemBackendCpp, EXECPOLSRC, T>
{
  inline static bool do_it(T* p_dest,
                           ExecSpace<MemBackendUmpire, EXECPOLDEST>& hwb_dest,
                           const T* p_src,
                           const ExecSpace<MemBackendCpp, EXECPOLSRC>& hwb_src,
                           const size_t& n)
  {
    if(hwb_dest.mem_backend().is_host()) {
      std::memcpy(p_dest, p_src, n*sizeof(T));
    } else {
      assert(false && "Transfer BACKENDS(TO:Umpire,FROM:Cpp-host) only supported with Umpire mem space host"); 
      return false;
    }
    return true;
  }
};

////////////////////////////////////////////////////////////////////////////////////////
// Transfers to/from CUDA memory
////////////////////////////////////////////////////////////////////////////////////////
#ifdef HIOP_USE_CUDA
template<class EXECPOLDEST, class EXECPOLSRC, typename T>
struct TransferImpl<MemBackendCuda, EXECPOLDEST, MemBackendUmpire, EXECPOLSRC, T>
{
  inline static bool do_it(T* p_dest,
                           ExecSpace<MemBackendCuda, EXECPOLDEST>& hwb_dest,
                           const T* p_src,
                           const ExecSpace<MemBackendUmpire, EXECPOLSRC>& hwb_src,
                           const size_t& n)
  {
    if(hwb_src.mem_backend().is_device()) {
      return cudaSuccess == cudaMemcpy(p_dest, p_src, n*sizeof(T), cudaMemcpyDeviceToDevice);
    } else {
      if(hwb_src.mem_backend().is_host()) {
        return cudaSuccess == cudaMemcpy(p_dest, p_src, n*sizeof(T), cudaMemcpyHostToDevice);
      } else {
        assert(false && "Transfer BACKENDS(TO:Cuda,FROM:umpire) not supported with Umpire mem space 'um'");
        return false;
      }
    }
  }
};

template<class EXECPOLDEST, class EXECPOLSRC, typename T>
struct TransferImpl<MemBackendUmpire, EXECPOLDEST, MemBackendCuda, EXECPOLSRC, T>
{
  inline static bool do_it(T* p_dest,
                           ExecSpace<MemBackendUmpire, EXECPOLDEST>& hwb_dest,
                           const T* p_src,
                           const ExecSpace<MemBackendCuda, EXECPOLSRC>& hwb_src,
                           const size_t& n)
  {
    if(hwb_dest.mem_backend().is_device()) {
      return cudaSuccess == cudaMemcpy(p_dest, p_src, n*sizeof(T), cudaMemcpyDeviceToDevice);
    } else {
      if(hwb_dest.mem_backend().is_host()) {
        return cudaSuccess == cudaMemcpy(p_dest, p_src, n*sizeof(T), cudaMemcpyDeviceToHost);
      } else {
        assert(false && "Transfer BACKENDS(TO:Umpire,FROM:Cuda) not supported with Umpire mem space 'um'");
        return false;
      }
    }
  }
};
#endif //HIOP_USE_CUDA

////////////////////////////////////////////////////////////////////////////////////////
// Transfers to/from HIP memory
////////////////////////////////////////////////////////////////////////////////////////
#ifdef HIOP_USE_HIP
// TODO
#endif
}  // end namespace hiop
#endif //HIOP_USE_RAJA
#endif //HIOP_EXEC_SPACE_RAJAUMP

