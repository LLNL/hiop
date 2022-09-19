// Copyright (c) 2022, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// LLNL-CODE-742473. All rights reserved.
//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop. HiOp 
// is released under the BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause). 
// Please also read "Additional BSD Notice" below.
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

/**
 * @file MemBackendUmpireImpl.hpp
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LLNL
 * 
 */

/**
 * This file contains Umpire implementation memory backend.
 */

#ifndef HIOP_MEM_BCK_UMPIRE
#define HIOP_MEM_BCK_UMPIRE

#include <ExecSpace.hpp>

#ifdef HIOP_USE_RAJA // can/should be HIOP_USE_UMPIRE

#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>

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
#endif //HIOP_MEM_BCK_UMPIRE

