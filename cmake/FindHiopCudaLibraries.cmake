
#[[

Exports target `hiop_cuda` which finds all cuda libraries needed by hiop.

Users may set the following variables:

- HIOP_NVCC_ARCH
- HIOP_CUDA_INCLUDE_DIR
- HIOP_CUDA_LIB_DIR

]]

add_library(hiop_cuda INTERFACE)

include(CheckLanguage)
enable_language(CUDA)
check_language(CUDA)
find_package(CUDA REQUIRED)

target_link_libraries(hiop_cuda INTERFACE
  culibos cublas cublasLt nvblas cusparse cudart cudadevrt)

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.17)
  target_link_libraries(hiop_cuda INTERFACE 
    CUDA_CUDART_LIBRARY
    CUDA_cudadevrt_LIBRARY
    CUDA_cublas_LIBRARY
    CUDA_cusparse_LIBRARY)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(HIOP_NVCC_ARCH)
  set(CMAKE_CUDA_FLAGS "-arch=${HIOP_NVCC_ARCH}")
  message(STATUS "Using CUDA arch ${HIOP_NVCC_ARCH}")
else()
  set(CMAKE_CUDA_FLAGS "-arch=sm_35")
  message(STATUS "Using CUDA arch sm_35")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

if(HIOP_CUDA_LIB_DIR)
  target_compile_options(hiop_cuda INTERFACE -L${HIOP_CUDA_LIB_DIR})
endif(HIOP_CUDA_LIB_DIR)

# on some systems RHEL Nvidia toolkit puts cuBlas in different locations
# user can setup/overwrite the lib directory
if(HIOP_CUBLAS_LIB_DIR)
  target_compile_options(hiop_cuda INTERFACE -L${HIOP_CUBLAS_LIB_DIR})
endif(HIOP_CUBLAS_LIB_DIR)

if(HIOP_CUDA_INCLUDE_DIR)
  target_include_directories(hiop_cuda INTERFACE ${HIOP_CUDA_INCLUDE_DIR})
else()
  target_include_directories(hiop_cuda INTERFACE ${CUDA_TOOLKIT_INCLUDE})
endif()

# for now we rely on MAGMA for GPUs computations
include(FindMagma)
target_link_libraries(hiop_cuda INTERFACE Magma)

message(STATUS "HiOp support for GPUs is on")
get_target_property(cuda_includes hiop_cuda INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "CUDA include directories: ${cuda_includes}")
get_target_property(cuda_libraries hiop_cuda INTERFACE_LINK_LIBRARIES)
message(STATUS "CUDA linked libraries: ${cuda_libraries}")

