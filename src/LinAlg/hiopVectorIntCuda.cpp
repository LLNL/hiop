// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
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
 * @file hiopVectorIntCuda.cpp
 *
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LLNL
 *
 */
#include "hiopVectorIntCuda.hpp"
#include "VectorCudaKernels.hpp"
#include "MathDeviceKernels.hpp"
#include <cuda_runtime.h>
#include <cassert>

//void hiop::cuda::set_to_linspace_kernel(int sz, int* buf, int i0, int di);

namespace hiop
{



hiopVectorIntCuda::hiopVectorIntCuda(size_type sz, std::string mem_space)
  : hiopVectorInt(sz),
    mem_space_(mem_space)
{
  // Size in bytes
  size_t bytes = sz * sizeof(index_type);

  // Allocate memory on GPU
  cudaError_t cuerr = cudaMalloc(&buf_, bytes);
  assert(cudaSuccess == cuerr);
  
  // Allocate memory on host
  buf_host_ = new index_type[bytes];
}

hiopVectorIntCuda::~hiopVectorIntCuda()
{
  delete buf_host_;

  // Delete workspaces and handles
  cudaFree(buf_);
}

void hiopVectorIntCuda::copy_from_dev()
{
  if (buf_ != buf_host_) {
    cudaError_t cuerr = cudaMemcpy(buf_host_, buf_, (sz_)*sizeof(index_type), cudaMemcpyDeviceToHost);
    assert(cuerr == cudaSuccess);
  }
}

void hiopVectorIntCuda::copy_to_dev()
{
  if (buf_ != buf_host_) {
    cudaError_t cuerr = cudaMemcpy(buf_, buf_host_, (sz_)*sizeof(index_type), cudaMemcpyHostToDevice);
    assert(cuerr == cudaSuccess);
  }
}

void hiopVectorIntCuda::copy_from(const index_type* v_local)
{
  if(v_local) {
    cudaError_t cuerr = cudaMemcpy(buf_, v_local, (sz_)*sizeof(index_type), cudaMemcpyDeviceToDevice);
    assert(cuerr == cudaSuccess);
  }
}

void hiopVectorIntCuda::set_to_zero()
{
  cudaError_t cuerr = cudaMemset(buf_, 0, sz_);
  assert(cuerr == cudaSuccess);
}

/// Set all vector elements to constant c
void hiopVectorIntCuda::set_to_constant(const index_type c)
{
  cudaError_t cuerr = cudaMemset(buf_, c, sz_);
  assert(cuerr == cudaSuccess);
}

/**
 * @brief Set the vector entries to be a linear space of starting at i0 containing evenly 
 * incremented integers up to i0+(n-1)di, when n is the length of this vector
 *
 * @pre The elements of the linear space should not overflow the index_type type
 *  
 * @param i0 the starting element in the linear space (entry 0 in vector)
 * @param di the increment for subsequent entries in the vector
 *
 */ 
void hiopVectorIntCuda::linspace(const index_type& i0, const index_type& di)
{
  hiop::cuda::set_to_linspace_kernel(sz_, buf_, i0, di);
}
  
} // namespace hiop



