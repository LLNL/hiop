#include "hiop_defs.hpp"

namespace hiop {

#if defined(HIOP_USE_RAJA)

#if defined(HIOP_USE_GPU)

  #define HIOP_RAJA_GPU_BLOCK_SIZE 128

#if defined(HIOP_USE_CUDA)

  #include "cuda.h"
  using hiop_raja_exec   = RAJA::cuda_exec<HIOP_RAJA_GPU_BLOCK_SIZE>;
  using hiop_raja_reduce = RAJA::cuda_reduce;
  using hiop_raja_atomic = RAJA::cuda_atomic;

  // The following are primarily for _matrix_exec_
  using hiop_block_x_loop = RAJA::cuda_block_x_loop;
  using hiop_thread_x_loop = RAJA::cuda_thread_x_loop;
  template<typename T>
  using hiop_kernel = RAJA::statement::CudaKernel<T>;

#elif defined(HIOP_USE_HIP)

  using hiop_raja_exec   = RAJA::hip_exec<HIOP_RAJA_GPU_BLOCK_SIZE>;
  using hiop_raja_reduce = RAJA::hip_reduce;
  using hiop_raja_atomic = RAJA::hip_atomic;

  // The following are primarily for _matrix_exec_
  using hiop_block_x_loop = RAJA::hip_block_x_loop;
  using hiop_thread_x_loop = RAJA::hip_thread_x_loop;
  template<typename T>
  using hiop_kernel = RAJA::statement::HipKernel<T>;

#else
#error "HIOP_USE_GPU Requires HIOP_USE_HIP or HIOP_USE_CUDA to be enabled."
#endif

  #define RAJA_LAMBDA [=] __device__
  using matrix_exec =
    RAJA::KernelPolicy<
      hiop_kernel<
        RAJA::statement::For<1, hiop_block_x_loop,
          RAJA::statement::For<0, hiop_thread_x_loop,
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;

#else

  using hiop_raja_exec   = RAJA::omp_parallel_for_exec;
  using hiop_raja_reduce = RAJA::omp_reduce;
  using hiop_raja_atomic = RAJA::omp_atomic;
  #define RAJA_LAMBDA [=]
  using matrix_exec = 
    RAJA::KernelPolicy<
      RAJA::statement::For<1, hiop_raja_exec,    // row
        RAJA::statement::For<0, hiop_raja_exec,  // col
          RAJA::statement::Lambda<0> 
        >
      >
    >;

#endif // HIOP_USE_GPU

#endif // HIOP_USE_RAJA

} // namespace hiop
