
#[[

Exports target `hiop_cuda` which finds all cuda libraries needed by hiop.

]]

add_library(hiop_cuda INTERFACE)

target_link_libraries(hiop_cuda INTERFACE culibos cublasLt nvblas cusparse cudart)

target_include_directories(hiop_cuda INTERFACE ${CUDAToolkit_INCLUDE_DIRS})

# for now we rely on MAGMA for GPUs computations
include(FindHiopMagma)
target_link_libraries(hiop_cuda INTERFACE Magma)

message(STATUS "HiOp support for GPUs is on")
get_target_property(cuda_libraries hiop_cuda INTERFACE_LINK_LIBRARIES)
message(STATUS "CUDA linked libraries: ${cuda_libraries}")

