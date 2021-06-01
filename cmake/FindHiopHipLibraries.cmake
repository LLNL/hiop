#[[

Exports target `hiop_hip` which finds all hip libraries needed by hiop.

#]]

add_library(hiop_hip INTERFACE)

find_library(HIPBLAS_LIBRARY
  NAMES
  hipblas
  PATHS
  ENV LD_LIBRARY_PATH ENV DYLD_LIBRARY_PATH
  PATH_SUFFIXES
  lib64 lib)

if(HIPBLAS_LIBRARY)
  get_filename_component(HIPBLAS_LIBRARY_DIR ${HIPBLAS_LIBRARY} DIRECTORY)
endif()

find_path(HIPBLAS_INCLUDE_DIR
  NAMES
  hipblas.h
  PATHS
  ${HIPBLAS_LIBRARY_DIR}/..
  PATH_SUFFIXES
  include)

if(HIPBLAS_LIBRARY)
  add_library(Hipblas INTERFACE)
  target_link_libraries(Hipblas INTERFACE ${HIPBLAS_LIBRARY})
  target_include_directories(Hipblas INTERFACE ${HIPBLAS_INCLUDE_DIR})
  message(STATUS "Found Hipblas include: ${HIPBLAS_INCLUDE_DIR}")
  message(STATUS "Found Hipblas library: ${HIPBLAS_LIBRARY}")
else()
  message(STATUS "Hipblas was not found.")
endif()

# for now we rely on MAGMA for GPUs computations
include(FindHiopMagma)
target_link_libraries(Magma    INTERFACE Hipblas)

# Find hip cmake targets
find_package(hip REQUIRED)
target_link_libraries(hiop_hip INTERFACE hip::device Magma)

message(STATUS "HiOp support for GPUs is on")
get_target_property(hip_includes hiop_hip INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "HIP include directories: ${hip_includes}")
get_target_property(hip_libraries hiop_hip INTERFACE_LINK_LIBRARIES)
message(STATUS "HIP linked libraries: ${hip_libraries}")

