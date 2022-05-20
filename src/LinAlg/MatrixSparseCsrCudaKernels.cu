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
 * @file MatrixSparseCSRCudaKernels.cu
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LNNL
 *
 */

#include "MatrixSparseCsrCudaKernels.hpp"

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__
void csr_add_vec_to_diag(int n, int nnz, int* irowptr, int* jcolind, double* values, double alpha, const double* diag_values)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int row=tid; row<n; row+=stride) {
    int L=irowptr[row];
    int R=irowptr[row+1]-1;
    assert(L<nnz);
    assert(R<nnz);
    assert(L<=R);

#ifndef NDEBUG
    int idx_found = -1;
#endif
    do { //binary search
      const int midpoint = (R+L)/2;
      if(jcolind[midpoint]>row) {
        R = midpoint-1;
      } else if(jcolind[midpoint]<row) {
        L = midpoint+1;
      } else {
        assert(idx_found<nnz);
        values[midpoint] += alpha*diag_values[row];
#ifndef NDEBUG
        idx_found = midpoint;
#endif        
        break;
      }
    } while(L<=R);
#ifndef NDEBUG
    assert(idx_found>=0 &&
           "add_val(vector)_to_diag(cuda): diagonal element not part of the nonzeros or column indexes not sorted");
#endif    
  }
}


__global__
void csr_add_val_to_diag(int n, int nnz, int* irowptr, int* jcolind, double* values, double val)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int row=tid; row<n; row+=stride) {
    int L=irowptr[row];
    int R=irowptr[row+1]-1;
    assert(L<nnz);
    assert(R<nnz);
    assert(L<=R);

#ifndef NDEBUG
    int idx_found = -1;
#endif
    do { //binary search
      const int midpoint = (R+L)/2;
      if(jcolind[midpoint]>row) {
        R = midpoint-1;
      } else if(jcolind[midpoint]<row) {
        L = midpoint+1;
      } else {
        assert(idx_found<nnz);
        values[midpoint] += val;
#ifndef NDEBUG
        idx_found = midpoint;
#endif        
        break;
      }
    } while(L<=R);
#ifndef NDEBUG
    assert(idx_found>=0 &&
           "add_val(scalar)_to_diag(cuda): diagonal element not part of the nonzeros or column indexes not sorted");
#endif    
  }
}

__global__
void csr_set_diag_to_val(int n, int nnz, int* irowptr, int* jcolind, double* values, double val)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int row=tid; row<n; row+=stride) {
    int L=irowptr[row];
    int R=irowptr[row+1]-1;
    assert(L<nnz);
    assert(R<nnz);
    assert(L<=R);

#ifndef NDEBUG
    int idx_found = -1;
#endif
    do { //binary search
      const int midpoint = (R+L)/2;
      if(jcolind[midpoint]>row) {
        R = midpoint-1;
      } else if(jcolind[midpoint]<row) {
        L = midpoint+1;
      } else {
        assert(idx_found<nnz);
        values[midpoint] = val;
#ifndef NDEBUG
        idx_found = midpoint;
#endif        
        break;
      }
    } while(L<=R);
#ifndef NDEBUG
    assert(idx_found>=0 && "set_diag(cuda): diagonal element not part of the nonzeros or column indexes not sorted");
#endif
  }
}

__global__
void csr_copy_diag_to_vec(int n,
                          int nnz,
                          const int* irowptr,
                          const int* jcolind,
                          const double* values,
                          double* diag_out)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int row=tid; row<n; row+=stride) {
    int L=irowptr[row];
    int R=irowptr[row+1]-1;
    assert(L<nnz);
    assert(R<nnz);
    assert(L<=R);
    diag_out[row] = 0.0; //in case elem (row,row) is not a nonzero
#ifndef NDEBUG
    int idx_found = -1;
#endif
    do { //binary search
      const int midpoint = (R+L)/2;
      if(jcolind[midpoint]>row) {
        R = midpoint-1;
      } else if(jcolind[midpoint]<row) {
        L = midpoint+1;
      } else {
        assert(idx_found<nnz);
        diag_out[row] = values[midpoint];
#ifndef NDEBUG
        idx_found = midpoint;
#endif        
        break;
      }
    } while(L<=R);
#ifndef NDEBUG
    assert(idx_found>=0 && "set_diag(cuda): diagonal element not part of the nonzeros or column indexes not sorted");
#endif
  }//end of for


}

namespace hiop
{
namespace cuda
{

void csr_set_diag_kernel(int n, int nnz, int* irowptr, int* jcolind, double* values, double val)
{
  //block of smaller sizes tend to perform 1.5-2x faster than the usual 256 or 128 blocks
  //300 microseconds for 1Mx1M matrix with 24M nnz
  int block_size=16;
  int num_blocks = (n+block_size-1)/block_size;
  csr_set_diag_to_val<<<num_blocks,block_size>>>(n, nnz, irowptr, jcolind, values, val);
}

void csr_add_diag_kernel(int n, int nnz, int* irowptr, int* jcolind, double* values, double val)
{
  //block of smaller sizes tend to perform 1.5-2x faster than the usual 256 or 128 blocks
  int block_size=16;
  int num_blocks = (n+block_size-1)/block_size;
  csr_add_val_to_diag<<<num_blocks,block_size>>>(n, nnz, irowptr, jcolind, values, val);
}

void csr_add_diag_kernel(int n, int nnz, int* irowptr, int* jcolind, double* values, double alpha, const double* Dvalues)
{
  //block of smaller sizes tend to perform 1.5-2x faster than the usual 256 or 128 blocks
  int block_size=16;
  int num_blocks = (n+block_size-1)/block_size;
  csr_add_vec_to_diag<<<num_blocks,block_size>>>(n, nnz, irowptr, jcolind, values, alpha, Dvalues);
}

void csr_get_diag_kernel(int n,
                         int nnz,
                         const int* irowptr,
                         const int* jcolind,
                         const double* values,
                         double* diag_out)
{
  //block of smaller sizes tend to perform 1.5-2x faster than the usual 256 or 128 blocks
  int block_size=16;
  int num_blocks = (n+block_size-1)/block_size;
  csr_copy_diag_to_vec<<<num_blocks,block_size>>>(n, nnz, irowptr, jcolind, values, diag_out);
}
}  //end of namespace
} //end of namespace
