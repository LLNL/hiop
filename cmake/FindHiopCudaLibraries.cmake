
#[[

Exports target `hiop_cuda` which finds all cuda libraries needed by hiop.

]]

add_library(hiop_cuda INTERFACE)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(hiop_cuda INTERFACE 
  culibos
  nvblas
  cusparse
  cusolver
  cudart
  cublasLt
  )

# if(HIOP_BUILD_SHARED)
#   target_link_libraries(hiop_cuda INTERFACE 
#     CUDA::cusparse
#     CUDA::cudart
#     CUDA::cublasLt
#     CUDA::cusolver
#     )
# endif()
# if(HIOP_BUILD_STATIC)
#   target_link_libraries(hiop_cuda INTERFACE
#     CUDA::culibos
#     CUDA::cusparse_static
#     CUDA::cudart_static
#     CUDA::cublasLt_static
#     CUDA::cusolver_static
#     )
# endif()

install(TARGETS hiop_cuda EXPORT hiop-targets)

message(STATUS "HiOp support for GPUs is on")
message(STATUS "CUDA include directories: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
get_target_property(cuda_libraries hiop_cuda INTERFACE_LINK_LIBRARIES)
message(STATUS "CUDA linked libraries: ${cuda_libraries}")

