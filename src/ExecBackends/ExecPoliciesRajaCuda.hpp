#ifndef HIOP_EXEC_POL_RAJA_CUDA
#define HIOP_EXEC_POL_RAJA_CUDA

#include "ExecSpace.hpp"

#include <cuda.h>
#define HIOP_RAJA_GPU_BLOCK_SIZE 128
#include <RAJA/RAJA.hpp>

namespace hiop
{

template<>
struct ExecRajaPoliciesBackend<ExecPolicyRajaCuda>
{
  using hiop_raja_exec   = RAJA::cuda_exec<HIOP_RAJA_GPU_BLOCK_SIZE>;
  using hiop_raja_reduce = RAJA::cuda_reduce;
  using hiop_raja_atomic = RAJA::cuda_atomic;

  // The following are primarily for _matrix_exec_
  using hiop_block_x_loop = RAJA::cuda_block_x_loop;
  using hiop_thread_x_loop = RAJA::cuda_thread_x_loop;
  template<typename T>
  using hiop_kernel = RAJA::statement::CudaKernel<T>;
};

}
#endif
