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
// Memory backends
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

#ifdef HIOP_USE_RAJA //HIOP_USE_UMPIRE would be better since Hiop RAJA code can now work without Umpire
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

  using MemBackendHost = MemBackendUmpire;
  /// Returns a backend set up for host memory space
  inline static MemBackendHost new_backend_host()
  {
    return MemBackendHost("HOST");
  };

private:
  std::string mem_space_;
};
#endif //HIOP_USE_RAJA //HIOP_USE_UMPIRE

#ifdef HIOP_USE_CUDA
/// Cuda memory backend for device memory space that is implemented using Cuda API
struct MemBackendCuda
{
  /// For now does not support host memory space (but can/will be implemented).
  inline static bool is_host() { return false; }

  inline static bool is_device() { return true; }

  using MemBackendHost = MemBackendCpp;
  /// Returns a backend set up for host memory space
  inline static MemBackendHost new_backend_host()
  {
    return MemBackendHost();
  };
};
#endif //HIOP_USE_CUDA

#ifdef HIOP_USE_HIP
/// Cuda memory backend for device memory space that is implemented using Hip API
struct MemBackendHip
{
  /// For now does not support host memory space (but can/will be implemented).
  inline static bool is_host() { return false; }

  inline static bool is_device() { return true; }

  using MemBackendHost = MemBackendCpp;
  /// Returns a backend set up for host memory space
  inline static MemBackendHost new_backend_host()
  {
    return MemBackendHost();
  };
};
#endif //HIOP_USE_HIP

///////////////////////////////////////////////////////////////////////////////////////
// Execution policies
///////////////////////////////////////////////////////////////////////////////////////

/// Standard C++ sequential execution
struct ExecPolicySeq
{
};

#ifdef HIOP_USE_CUDA
struct ExecPolicyCuda
{
  unsigned short int num_blocks_vector;
  unsigned short int num_blocks_search;
};
#endif

#ifdef HIOP_USE_HIP
struct ExecPolicyHip
{
  unsigned short int num_blocks_vector;
  unsigned short int num_blocks_search;
};
#endif

///////////////////////////
// RAJA execution policies
//////////////////////////
#ifdef HIOP_USE_RAJA
#ifdef HIOP_USE_CUDA
struct ExecPolicyRajaCuda
{
  //empty since no runtime info is stored
};
#endif

#ifdef HIOP_USE_HIP
struct ExecPolicyRajaHip
{
  //empty since no runtime info is stored
};
#endif

//RAJA OMP execution policies backend present but not tested
#define HIOP_USE_RAJAOMP 0
#if defined(HIOP_USE_RAJAOMP)
struct ExecPolicyRajaOmp
{
  //empty since no runtime info is stored
};
#endif //HIOP_USE_RAJA

/**
 * The backend RAJA policies that needs to be provided for each one of the ExecPolicyRajaCuda,
 * ExecPolicyRajaHip, and/or ExecPolicyRajaOmp. The class is specialized in HiOp's vendor-specific
 * Raja execution policies source files. Namely, the class' inner types are specialized to 
 * vendor-specific RAJA policies types. The inner type below are just for reference and this
 * generic templated struct should not be used.
 */
template<class RAJAEXECPOLICIES>
struct ExecRajaPoliciesBackend
{
  using hiop_raja_exec   = void;
  using hiop_raja_reduce = void; 
  using hiop_raja_atomic = void;

  // The following are primarily for _matrix_exec_
  using hiop_block_x_loop = void;
  using hiop_thread_x_loop = void;
  template<typename T> using hiop_kernel = void;
};
#endif

///////////////////////////////////////////////////////////////////////////////////////
// The generic/template execution backend class
///////////////////////////////////////////////////////////////////////////////////////

//
// Forward declarations of implementation internals of class ExecSpace
//
template<class MEMBACKEND, typename T> struct AllocImpl;
template<class MEMBACKENDDEST, class EXEPOLDEST, class MEMBACKENDSRC, class EXEPOLSRC, typename T>
struct TransferImpl;

/** 
 * Hardware backend wrapping a concrete memory backend and a concrete set of execution policies.
 * 
 * Re: memory backends, this class provides methods for i. allocations and deallocations of "raw"
 * arrays and ii. copy of arrays between memory backends.
 *
 * Re: execution policies, TBD.
 */
template<class MEMBACKEND, class EXECPOLICIES>
class ExecSpace
{
public:
  ExecSpace() = default;
  ExecSpace(const ExecSpace&) = default;

  ExecSpace(const MEMBACKEND& mb)
    : mb_(mb),
      ep_()
  {
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
   * @pre `p_src` should be managed by the memory backend of `ms`.
   */
  template<class MEMSRC, class EXEPOLSRC, typename T>
  inline bool copy(T* p_dest, const T* p_src, const size_t& n, const ExecSpace<MEMSRC,EXEPOLSRC>& ms)
  {
    return TransferImpl<MEMBACKEND, EXECPOLICIES, MEMSRC, EXEPOLSRC, T>::do_it(p_dest, *this, p_src, ms, n);
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
    return TransferImpl<MEMBACKEND, EXECPOLICIES, MEMBACKEND, EXECPOLICIES, T>::do_it(p_dest, *this, p_src, *this, n);
  }

private:
  MEMBACKEND mb_;
  EXECPOLICIES ep_;
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
template<class MEMDEST, class EXEPOLDEST, class MEMSRC, class EXEPOLSRC, typename T>
struct TransferImpl
{
  inline static bool do_it(T* p_dest,
                           ExecSpace<MEMDEST, EXEPOLDEST>& hwb_dest,
                           const T* p_src,
                           const ExecSpace<MEMSRC, EXEPOLSRC>& hwb_src,
                           const size_t& n)
  {
    return false;
  }
};

} // end namespace

#endif
