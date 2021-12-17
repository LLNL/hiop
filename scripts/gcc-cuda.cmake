#[[

Default CMake cache for building in CI with most options enabled and a sane
default value for CUDA architectures.

#]]
message(STATUS "Enabling both shared and static libraries")
set(HIOP_BUILD_SHARED OFF CACHE BOOL "")
set(HIOP_BUILD_STATIC ON CACHE BOOL "")

set(ENABLE_TESTS ON CACHE BOOL "")

message(STATUS "Enabling MPI, RAJA, Umpire, Kron Reduction, and CUDA")
set(HIOP_USE_MPI ON CACHE BOOL "")
set(HIOP_USE_RAJA ON CACHE BOOL "")
set(HIOP_USE_UMPIRE ON CACHE BOOL "")
set(HIOP_WITH_KRON_REDUCTION ON CACHE BOOL "")
set(HIOP_USE_GPU ON CACHE BOOL "")
set(HIOP_USE_CUDA ON CACHE BOOL "")

message(STATUS "Enabling HiOp's Sparse Interface")
set(HIOP_SPARSE ON CACHE BOOL "")

message(STATUS "Enabling HiOp's deepchecking")
set(HIOP_DEEPCHECKS ON CACHE BOOL "")

message(STATUS "Setting default cuda architecture to 60")
set(CMAKE_CUDA_ARCHITECTURES 60 CACHE STRING "")

message(STATUS "Done preloading CMake cache with values for continuous integration")
