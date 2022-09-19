#ifndef HIOP_EXEC_SPACE_HOST
#define HIOP_EXEC_SPACE_HOST

#include <ExecSpace.hpp>

#include <cassert>
#include <cstring>

namespace hiop
{

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
template<class EXECPOLDEST, class EXECPOLSRC, typename T>
struct TransferImpl<MemBackendCpp, EXECPOLDEST, MemBackendCpp, EXECPOLSRC, T>
{
  inline static bool do_it(T* p_dest,
                           ExecSpace<MemBackendCpp, EXECPOLDEST>& hwb_dest,
                           const T* p_src,
                           const ExecSpace<MemBackendCpp, EXECPOLSRC>& hwb_src,
                           const size_t& n)
  {
    std::memcpy(p_dest, p_src, n*sizeof(T));
    return true;
  }
};
 
} // end namespace hiop
#endif
