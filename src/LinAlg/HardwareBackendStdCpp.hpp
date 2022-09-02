namespace hiop
{

template<>
struct FeatureIsPresent<MemBackendCpp>
{
 static constexpr bool value = true;
};

template<>
struct SupportsHostMemSpace<MemBackendCpp>
{
  static constexpr bool value = true;
};

template<typename T>
struct TransferImpl<MemBackendCpp, MemBackendCpp, T>
{
  inline static bool do_it(T* p_dest,
                           HWBackend<MemBackendCpp>& hwb_dest,
                           const T* p_src,
                           const HWBackend<MemBackendCpp>& hwb_src,
                           const size_t& n)
  {
    std::memcpy(p_dest, p_src, n*sizeof(T));
    return true;
  }
};

template<typename T>
struct TransferImpl<MemBackendCpp, MemBackendUmpire, T>
{
  inline static bool do_it(T* p_dest,
                           HWBackend<MemBackendCpp>& hwb_dest,
                           const T* p_src,
                           const HWBackend<MemBackendUmpire>& hwb_src,
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

template<typename T>
struct TransferImpl<MemBackendUmpire, MemBackendCpp, T>
{
  inline static bool do_it(T* p_dest,
                           HWBackend<MemBackendUmpire>& hwb_dest,
                           const T* p_src,
                           const HWBackend<MemBackendCpp>& hwb_src,
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


  
} // end namespace hiop
