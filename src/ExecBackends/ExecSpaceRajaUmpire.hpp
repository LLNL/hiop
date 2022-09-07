#ifdef HIOP_USE_RAJA

#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>
#include <RAJA/RAJA.hpp>

namespace hiop {

template<>
struct FeatureIsPresent<MemBackendUmpire>
{
  static constexpr bool value = true;
};

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

template<typename T>
struct TransferImpl<MemBackendUmpire, MemBackendUmpire, T>
{
  inline static bool do_it(T* p_dest,
                           HWBackend<MemBackendUmpire>& hwb_dest,
                           const T* p_src,
                           const HWBackend<MemBackendUmpire>& hwb_src,
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

}  // end namespace
#endif

