// This file is part of HiOp. For details, see https://github.com/LLNL/hiop.
// HiOp is released under the BSD 3-clause license
// (https://opensource.org/licenses/BSD-3-Clause). Please also read “Additional
// BSD Notice” below.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// i. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the disclaimer below. ii. Redistributions in
// binary form must reproduce the above copyright notice, this list of
// conditions and the disclaimer (as noted below) in the documentation and/or
// other materials provided with the distribution.
// iii. Neither the name of the LLNS/LLNL nor the names of its contributors may
// be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC,
// THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Additional BSD Notice
// 1. This notice is required to be provided under our contract with the U.S.
// Department of Energy (DOE). This work was produced at Lawrence Livermore
// National Laboratory under Contract No. DE-AC52-07NA27344 with the DOE.
// 2. Neither the United States Government nor Lawrence Livermore National
// Security, LLC nor any of their employees, makes any warranty, express or
// implied, or assumes any liability or responsibility for the accuracy,
// completeness, or usefulness of any information, apparatus, product, or
// process disclosed, or represents that its use would not infringe
// privately-owned rights.
// 3. Also, reference herein to any specific commercial products, process, or
// services by trade name, trademark, manufacturer or otherwise does not
// necessarily constitute or imply its endorsement, recommendation, or favoring
// by the United States Government or Lawrence Livermore National Security,
// LLC. The views and opinions of authors expressed herein do not necessarily
// state or reflect those of the United States Government or Lawrence Livermore
// National Security, LLC, and shall not be used for advertising or product
// endorsement purposes.

/**
 * @file KrylovSolverKernels.cu
 *
 * @author Kasia Swirydowicz <kasia.Swirydowicz@pnnl.gov>, PNNL
 */
#include "KrylovSolverKernels.h"
#define maxk 1024
#define Tv5 1024
//computes V^T[u1 u2] where v is n x k and u1 and u2 are nx1
__global__ void MassIPTwoVec_kernel(const double* __restrict__ u1, 
                                    const double* __restrict__ u2, 
                                    const double* __restrict__ v, 
                                    double* result,
                                    const int k, 
                                    const int N)
{
  int t = threadIdx.x;
  int bsize = blockDim.x;

  // assume T threads per thread block (and k reductions to be performed)
  volatile __shared__ double s_tmp1[Tv5];

  volatile __shared__ double s_tmp2[Tv5];
  // map between thread index space and the problem index space
  int j = blockIdx.x;
  s_tmp1[t] = 0.0f;
  s_tmp2[t] = 0.0f;
  int nn = t;
  double can1, can2, cbn;

  while(nn < N) {
    can1 = u1[nn];
    can2 = u2[nn];

    cbn = v[N * j + nn];
    s_tmp1[t] += can1 * cbn;
    s_tmp2[t] += can2 * cbn;

    nn += bsize;
  }

  __syncthreads();

  if(Tv5 >= 1024) {
    if(t < 512) {
      s_tmp1[t] += s_tmp1[t + 512];
      s_tmp2[t] += s_tmp2[t + 512];
    }
    __syncthreads();
  }
  if(Tv5 >= 512) {
    if(t < 256) {
      s_tmp1[t] += s_tmp1[t + 256];
      s_tmp2[t] += s_tmp2[t + 256];
    }
    __syncthreads();
  }
  {
    if(t < 128) {
      s_tmp1[t] += s_tmp1[t + 128];
      s_tmp2[t] += s_tmp2[t + 128];
    }
    __syncthreads();
  }
  {
    if(t < 64) {
      s_tmp1[t] += s_tmp1[t + 64];
      s_tmp2[t] += s_tmp2[t + 64];
    }
    __syncthreads();
  }

  if(t < 32) {
    s_tmp1[t] += s_tmp1[t + 32];
    s_tmp2[t] += s_tmp2[t + 32];

    s_tmp1[t] += s_tmp1[t + 16];
    s_tmp2[t] += s_tmp2[t + 16];

    s_tmp1[t] += s_tmp1[t + 8];
    s_tmp2[t] += s_tmp2[t + 8];

    s_tmp1[t] += s_tmp1[t + 4];
    s_tmp2[t] += s_tmp2[t + 4];

    s_tmp1[t] += s_tmp1[t + 2];
    s_tmp2[t] += s_tmp2[t + 2];

    s_tmp1[t] += s_tmp1[t + 1];
    s_tmp2[t] += s_tmp2[t + 1];
  }
  if(t == 0) {
    result[blockIdx.x] = s_tmp1[0];
    result[blockIdx.x + k] = s_tmp2[0];
  }
}


//mass AXPY i.e y = y - x*alpha where alpha is [k x 1], needed in 1 and 2 synch GMRES

__global__ void massAxpy3_kernel(int N,
                                 int k,
                                 const double* x_data,
                                 double* y_data,
                                 const double* alpha) {

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int t = threadIdx.x;
  __shared__ double s_alpha[maxk];
  if(t < k) {
    s_alpha[t] = alpha[t];
  }
  __syncthreads();

  if(i < N) {
    double temp = 0.0f;
    for(int j = 0; j < k; ++j) {
      temp += x_data[j * N + i] * s_alpha[j];
    }
    y_data[i] -= temp;
  }
}

__global__ void matrixInfNormPart1(const int n, 
                                   const int nnz, 
                                   const int* a_ia,
                                   const double* a_val, 
                                   double* result) {

  // one thread per row, pass through rows
  // and sum
  // can be done through atomics
  //\sum_{j=1}^m abs(a_{ij})

  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  while (idx < n){
    double sum = 0.0f;
    for (int i = a_ia[idx]; i < a_ia[idx+1]; ++i) {
      sum = sum + fabs(a_val[i]);
    }
    result[idx] = sum;
    idx += (blockDim.x*gridDim.x);
  }
}


void mass_inner_product_two_vectors(int n, 
                                    int i, 
                                    double* vec1, 
                                    double* vec2, 
                                    double* mvec, 
                                    double* result)
{
  MassIPTwoVec_kernel<<<i + 1, 1024>>>(vec1, vec2, mvec, result, i + 1, n);
}
void mass_axpy(int n, int i, double* x, double* y, double* alpha)
{
  massAxpy3_kernel<<<(n + 384 - 1) / 384, 384>>>(n, i + 1, x, y, alpha);
}

void matrix_row_sums(int n, 
                     int nnz, 
                     int* a_ia,
                     double* a_val, 
                     double* result)
{
  matrixInfNormPart1<<<1000,1024>>>(n, nnz, a_ia, a_val, result);
}
