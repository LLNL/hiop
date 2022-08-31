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
    std::cout << "alloc_array umpire    loc " << mb.location() << "\n";

    auto& resmgr = umpire::ResourceManager::getInstance();
    umpire::Allocator devalloc  = resmgr.getAllocator("HOST");
    return static_cast<T*>(devalloc.allocate(n*sizeof(T)));
  }
  inline static void dealloc(MemBackendUmpire& mb, T* p)
  {
    std::cout << "dealloc umpire    loc " << mb.location() << "\n";
    auto& resmgr = umpire::ResourceManager::getInstance();
    umpire::Allocator devalloc  = resmgr.getAllocator("HOST");
    devalloc.deallocate(p);
  }  
};

}  // end namespace
#endif

