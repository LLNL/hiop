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
 * @file MathCudaKernels.cu
 *
 * @author Cosmin G. Petra <petra1@llnl.gov>, LNNL
 * @author Nai-Yuan Chiang <chiang7@llnl.gov>, LNNL
 *
 */


#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "hiopCppStdUtils.hpp"
#include "MathDeviceKernels.hpp"
#include <thrust/functional.h>
#include <functional>

__global__
void array_random_uniform_cuda(int n, double* d_array, unsigned long seed, double minv, double maxv)
{
    const int num_threads = blockDim.x * gridDim.x;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;    
    const double delta = maxv - minv;
    curandState state;
    curand_init(seed, tid, 0, &state);
    for (int i = tid; i < n; i += num_threads) {
      const double ranv = curand_uniform_double( &state ); // from 0 to 1
      d_array[i] = ranv * delta + minv;	
    }
}

__global__ void set_to_constant_cu(int n, double *vec, double val)
{

  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;    
  for (int i = tid; i < n; i += num_threads) {
    vec[i] = val;	
  }
}

__global__ void copy_to_mapped_dest_cu(int n, double* src, double* dest, int* mapping)
{

  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;    
  for (int i = tid; i < n; i += num_threads) {
    dest[mapping[i]] = src[i];	
  }
}

__global__ void copy_from_mapped_src_cu(int n, double* src, double* dest, int* mapping)
{

  const int num_threads = blockDim.x * gridDim.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;    
  for (int i = tid; i < n; i += num_threads) {
    dest[i] = src[mapping[i]];	
  }
}

namespace hiop
{
namespace device
{

int array_random_uniform_kernel(int n, double* d_array, double minv, double maxv)
{
  int block_size=256;
  int grid_size = (n+block_size-1)/block_size;
  
  unsigned long seed = generate_seed();
  array_random_uniform_cuda<<<grid_size,block_size>>>(n, d_array, seed, minv, maxv);
  cudaDeviceSynchronize();

  return 1;
}

int array_random_uniform_kernel(int n, double* d_array)
{
  unsigned long seed = generate_seed();

  curandGenerator_t generator;
  curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(generator, seed);

  // generate random val from 0 to 1
  curandGenerateUniformDouble(generator, d_array, n);

  curandDestroyGenerator(generator);
  return 1;
}

void set_to_val_kernel(int n, double* values, double val)
{
  int block_size=256;
  int num_blocks = (n+block_size-1)/block_size;
  set_to_constant_cu<<<num_blocks,block_size>>>(n, values, val);
}

void copy_src_to_mapped_dest_kernel(int n, double* src, double* dest, int* mapping)
{
  int block_size=256;
  int num_blocks = (n+block_size-1)/block_size;
  copy_to_mapped_dest_cu<<<num_blocks,block_size>>>(n, src, dest, mapping);
}

void copy_mapped_src_to_dest_kernel(int n, double* src, double* dest, int* mapping)
{
  int block_size=256;
  int num_blocks = (n+block_size-1)/block_size;
  copy_from_mapped_src_cu<<<num_blocks,block_size>>>(n, src, dest, mapping);
}

} //end of namespace device
} //end of namespace hiop

