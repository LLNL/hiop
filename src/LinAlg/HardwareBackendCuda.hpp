#ifdef HIOP_USE_CUDA
namespace hiop
{

template<>
struct FeatureIsPresent<MemBackendCuda>
{
 static constexpr bool value = true;
};


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
template<typename T>
struct TransferImpl<MemBackendCuda, MemBackendCuda, T>
{
  inline static bool do_it(T* p_dest,
                           HWBackend<MemBackendCuda>& hwb_dest,
                           const T* p_src,
                           const HWBackend<MemBackendCuda>& hwb_src,
                           const size_t& n)
  {
    return cudaSuccess == cudaMemcpy(p_dest, p_src, n*sizeof(T), cudaMemcpyDeviceToDevice);
  }
};


template<typename T>
struct TransferImpl<MemBackendCuda, MemBackendCpp, T>
{
  inline static bool do_it(T* p_dest,
                           HWBackend<MemBackendCuda>& hwb_dest,
                           const T* p_src,
                           const HWBackend<MemBackendCpp>& hwb_src,
                           const size_t& n)
  {
    return cudaSuccess == cudaMemcpy(p_dest, p_src, n*sizeof(T), cudaMemcpyHostToDevice);
  }
};

template<typename T>
struct TransferImpl<MemBackendCpp, MemBackendCuda, T>
{
  inline static bool do_it(T* p_dest,
                           HWBackend<MemBackendCpp>& hwb_dest,
                           const T* p_src,
                           const HWBackend<MemBackendCuda>& hwb_src,
                           const size_t& n)
  {
    return cudaSuccess == cudaMemcpy(p_dest, p_src, n*sizeof(T), cudaMemcpyDeviceToHost);
  }
};

  
template<typename T>
struct TransferImpl<MemBackendCuda, MemBackendUmpire, T>
{
  inline static bool do_it(T* p_dest,
                           HWBackend<MemBackendCuda>& hwb_dest,
                           const T* p_src,
                           const HWBackend<MemBackendUmpire>& hwb_src,
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

template<typename T>
struct TransferImpl<MemBackendUmpire, MemBackendCuda, T>
{
  inline static bool do_it(T* p_dest,
                           HWBackend<MemBackendUmpire>& hwb_dest,
                           const T* p_src,
                           const HWBackend<MemBackendCuda>& hwb_src,
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


} // end namespace hiop
#endif
