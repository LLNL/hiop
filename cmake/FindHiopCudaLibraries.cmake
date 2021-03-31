
#[[

Exports target `hiop_cuda` which finds all cuda libraries needed by hiop.

]]

add_library(hiop_cuda INTERFACE)

find_package(CUDA REQUIRED)

find_library(CUDA_culibos_LIBRARY
  NAMES
  culibos
  PATHS
  ${CUDA_TOOLKIT_ROOT_DIR} $ENV{CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES
  lib64 lib
  )

find_library(CUDA_cublasLt_LIBRARY
  NAMES
  cublasLt
  PATHS
  ${CUDA_TOOLKIT_ROOT_DIR} $ENV{CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES
  lib64 lib
  )

find_library(CUDA_nvblas_LIBRARY
  NAMES
  nvblas
  PATHS
  ${CUDA_TOOLKIT_ROOT_DIR} $ENV{CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES
  lib64 lib
  )

target_link_libraries(hiop_cuda INTERFACE
  ${CUDA_cublas_LIBRARY}
  ${CUDA_CUDART_LIBRARY}
  ${CUDA_cudadevrt_LIBRARY}
  ${CUDA_cusparse_LIBRARY}
  ${CUDA_cublasLt_LIBRARY}
  ${CUDA_nvblas_LIBRARY}
  ${CUDA_culibos_LIBRARY}
  )

target_include_directories(hiop_cuda INTERFACE ${CUDA_TOOLKIT_INCLUDE})

# for now we rely on MAGMA for GPUs computations
include(FindHiopMagma)
target_link_libraries(hiop_cuda INTERFACE Magma)

message(STATUS "HiOp support for GPUs is on")
get_target_property(cuda_includes hiop_cuda INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "CUDA include directories: ${cuda_includes}")
get_target_property(cuda_libraries hiop_cuda INTERFACE_LINK_LIBRARIES)
message(STATUS "CUDA linked libraries: ${cuda_libraries}")

