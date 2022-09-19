#ifndef HIOP_EXEC_SPACE_HOST
#define HIOP_EXEC_SPACE_HOST

#include <ExecSpace.hpp>

#include <cassert>
#include <cstring>

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

//
// Allocator
//
template<typename T>
struct AllocImpl<MemBackendCpp, T>
{
  inline static T* alloc(MemBackendCpp& mb, const size_t& n)
  {
    return new T[n];
  }
  inline static void dealloc(MemBackendCpp& mb, T* p)
  {
    delete[] p;
  }  
};

//
// Transfers
//
template<typename T>
struct TransferImpl<MemBackendCpp, MemBackendCpp, T>
{
  inline static bool do_it(T* p_dest,
                           ExecSpace<MemBackendCpp>& hwb_dest,
                           const T* p_src,
                           const ExecSpace<MemBackendCpp>& hwb_src,
                           const size_t& n)
  {
    std::memcpy(p_dest, p_src, n*sizeof(T));
    return true;
  }
};
 
} // end namespace hiop
#endif
