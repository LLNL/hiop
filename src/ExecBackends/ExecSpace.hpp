// Copyright (c) 2022, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// LLNL-CODE-742473. All rights reserved.
//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop. HiOp 
// is released under the BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause). 
// Please also read “Additional BSD Notice” below.
//
// Redistribution and use in source and binary forms, with or without modification, 
// are permitted provided that the following conditions are met:
// i. Redistributions of source code must retain the above copyright notice, this list 
// of conditions and the disclaimer below.
// ii. Redistributions in binary form must reproduce the above copyright notice, 
// this list of conditions and the disclaimer (as noted below) in the documentation and/or 
// other materials provided with the distribution.
// iii. Neither the name of the LLNS/LLNL nor the names of its contributors may be used to 
// endorse or promote products derived from this software without specific prior written 
// permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
// SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR 
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS 
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED 
// AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Additional BSD Notice
// 1. This notice is required to be provided under our contract with the U.S. Department 
// of Energy (DOE). This work was produced at Lawrence Livermore National Laboratory under 
// Contract No. DE-AC52-07NA27344 with the DOE.
// 2. Neither the United States Government nor Lawrence Livermore National Security, LLC 
// nor any of their employees, makes any warranty, express or implied, or assumes any 
// liability or responsibility for the accuracy, completeness, or usefulness of any 
// information, apparatus, product, or process disclosed, or represents that its use would
// not infringe privately-owned rights.
// 3. Also, reference herein to any specific commercial products, process, or services by 
// trade name, trademark, manufacturer or otherwise does not necessarily constitute or 
// imply its endorsement, recommendation, or favoring by the United States Government or 
// Lawrence Livermore National Security, LLC. The views and opinions of authors expressed 
// herein do not necessarily state or reflect those of the United States Government or 
// Lawrence Livermore National Security, LLC, and shall not be used for advertising or 
// product endorsement purposes.

#ifndef HIOP_EXEC_SPACE
#define HIOP_EXEC_SPACE

#include <hiop_defs.hpp>
#include <string>

///////////////////////////////////////////////////////////////////////////////////////
// This header defines the generic execution space class and its various generic types.
// It should be used from a concrete execution space class, for example
// ExecSpaceHost.hpp or ExecSpaceCuda.hpp. The concrete execution space class should be
// then included in the code.
///////////////////////////////////////////////////////////////////////////////////////

namespace hiop
{

/** 
 * Runtime information about the execution space. Closely related to HiOp's options.
 */
struct ExecSpaceInfo
{
  ExecSpaceInfo(const std::string mem_space_in)
  {
    mem_space = mem_space_in;
    if(mem_space == "DEFAULT") {
      mem_backend = "STDCPP";
      mem_backend_host = "STDCPP";
      exec_backend = "HOST";
    } else {
      mem_backend = "UMPIRE";
      mem_backend_host = "UMPIRE";
      exec_backend = "RAJA";
    }
  }
  ExecSpaceInfo(const char* mem_space_in)
    : ExecSpaceInfo(std::string(mem_space_in))
  {
  }
  
  std::string mem_space;
  std::string mem_backend;
  std::string mem_backend_host;
  std::string exec_backend;
};

  
///////////////////////////////////////////////////////////////////////////////////////
// The memory backends classes
///////////////////////////////////////////////////////////////////////////////////////

/// Standard C++ memory backend on host 
struct MemBackendCpp
{
  /// Always on host memory space
  static bool is_host() { return true; }

  /// No host memory space is supported.
  static bool is_device() { return false; }

  /// Returns a backend set up for host memory space
  static MemBackendCpp new_backend_host()
  {
    return MemBackendCpp();
  };
};

/// Cuda memory backend for device memory space that is implemented using  Cuda API
struct MemBackendCuda
{
  /// For now does not support memory space (but can/will be implemented).
  static bool is_host() { return false; }

  static bool is_device() { return true; }
};

/**
 * Umpire-based memory backend that supports "HOST", "UM" (unified memory), and "DEVICE"
 * memory spaces.
 */
struct MemBackendUmpire
{
  MemBackendUmpire(const std::string& l)
    : mem_space_(l)
  {
  }
  MemBackendUmpire() //todo = delete;
  {
    mem_space_ = "HOST";
  }

  std::string mem_space() const
  {
    return mem_space_;
  }

  inline bool is_host() const
  {
    return mem_space_ == "HOST";
  }
  inline bool is_device() const
  {
    return mem_space_ == "DEVICE";
  }

  /// Returns a backend set up for host memory space
  static MemBackendUmpire new_backend_host()
  {
    return MemBackendUmpire("HOST");
  };

private:
  std::string mem_space_;
};

  
///////////////////////////////////////////////////////////////////////////////////////
// Execution policies
///////////////////////////////////////////////////////////////////////////////////////

struct ExePoliciesCuda
{

};
struct ExePoliciesRaja
{

};

///////////////////////////////////////////////////////////////////////////////////////
// The generic/template execution backend class
///////////////////////////////////////////////////////////////////////////////////////

//
// Forward declarations of implementation internals of class ExecSpace
//
template<class MEMBACKEND, typename T> struct AllocImpl;
template<class MEMBACKENDDEST, class MEMBACKENDSRC, typename T> struct TransferImpl;
template<class FEATURE> struct FeatureIsPresent;
template<class MEMBACKEND> struct SupportsHostMemSpace;

/** 
 * Hardware backend wrapping a concrete memory backend and a concrete set of execution policies.
 * 
 * Re: memory backends, this class provides methods for i. allocations and deallocations of "raw"
 * arrays and ii. copy of arrays between memory backends.
 *
 * Re: execution policies, TBD.
 */
template<class MEMBACKEND>
class ExecSpace
{
public:
  ExecSpace()
  {
    static_assert("HiOp was not built with the requested hardware backend/memory space." &&
                  FeatureIsPresent<MEMBACKEND>::value);
  }
  ExecSpace(const MEMBACKEND& mb)
    : mb_(mb)
  {
    static_assert("HiOp was not built with the requested hardware backend/memory space." &&
                  FeatureIsPresent<MEMBACKEND>::value);
  }

  const MEMBACKEND& mem_backend() const
  {
    return mb_;
  }
  
  template<typename T>
  inline T* alloc_array(const size_t& n)
  {
    return AllocImpl<MEMBACKEND,T>::alloc(mb_, n);
  }

  template<typename T>
  inline void dealloc_array(T* p)
  {
    AllocImpl<MEMBACKEND,T>::dealloc(mb_, p);
  }

  /**
   * Copy `n` elements of the array `p_src` to the `p_dest` array.
   * 
   * @pre `p_src` and `p_dest` should be allocated so that they can hold at least 
   * `n` elements.
   * @pre `p_dest` should be managed by the memory backend of `this`.
   * @pre `p_src` should be managed by the memory backend of `md`.
   */
  template<class MEMSRC, typename T>
  inline bool copy(T* p_dest, const T* p_src, const size_t& n, const ExecSpace<MEMSRC>& md)
  {
    return TransferImpl<MEMBACKEND,MEMSRC,T>::do_it(p_dest, *this, p_src, md, n);
  }

  /**
   * Copy `n` elements of the array `p_src` to the `p_dest` array.
   * 
   * @pre `p_src` and `p_dest` should be allocated so that they can hold at least 
   * `n` elements.
   * @pre Both `p_dest` and `p_src` should be managed by the memory backend of `this`.
   */  
  template<typename T>
  inline bool copy(T* p_dest, const T* p_src, const size_t& n)
  {
    return TransferImpl<MEMBACKEND,MEMBACKEND,T>::do_it(p_dest, *this, p_src, *this, n);
  }

private:
  MEMBACKEND mb_;
};

//
// Internals start here
//
  
/**
 * Memory allocations should be provided via `AllocImpl` for concrete memory backends. 
 */
template<class MEMBACKEND, typename T>
struct AllocImpl
{
  inline static T* alloc(MEMBACKEND& mb, const size_t& n)
  {
    return nullptr;
  }
  inline static void dealloc(MEMBACKEND& mb, T* p)
  {
  }
};

/**
 * Transfers between memory backends and memory spaces should be provided by specializations
 * of `TransferImpl` class.
 */
template<class MEMDEST, class MEMSRC, typename T>
struct TransferImpl
{
  inline static bool do_it(T* p_dest,
                           ExecSpace<MEMDEST>& hwb_dest,
                           const T* p_src,
                           const ExecSpace<MEMSRC>& hwb_src,
                           const size_t& n)
  {
    return false;
  }
};

/// Generic trait: concrete types that are supported should specialize `value` to true.
template<class Feature> struct FeatureIsPresent
{
  static constexpr bool value = false; 
};

/// Generic trait: concrete memory backends that supports Host memory space should specialize this to be true.
template<class MEMBACKEND>
struct SupportsHostMemSpace
{
  static constexpr bool value = false;
};

} // end namespace

#endif
