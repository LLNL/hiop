
#[[

Looks for `umfpack` library and header directory.

Exports target `UMFPACK` which links to umfpack.(so|a)
and add include directories where umfpack.h was found.

Users may set the following variables:

- HIOP_UMFPACK_DIR

]]

find_library(UMFPACK_LIBRARY
  NAMES
  umfpack
  PATHS
  ${UMFPACK_DIR} $ENV{UMFPACK_DIR} ${HIOP_UMFPACK_DIR}
  ENV LD_LIBRARY_PATH ENV DYLD_LIBRARY_PATH
  PATH_SUFFIXES
  lib64 lib)

if(UMFPACK_LIBRARY)
  get_filename_component(UMFPACK_LIBRARY_DIR ${UMFPACK_LIBRARY} DIRECTORY)
endif()

find_path(UMFPACK_INCLUDE_DIR
  NAMES
  umfpack.h
  PATHS
  ${UMFPACK_DIR} $ENV{UMFPACK_DIR} ${HIOP_UMFPACK_DIR} ${UMFPACK_LIBRARY_DIR}/..
  PATH_SUFFIXES
  include
  include/suitesparse
  include/ufsparse)

if(UMFPACK_LIBRARY)
  message(STATUS "Found umfpack include: ${UMFPACK_INCLUDE_DIR}")
  message(STATUS "Found umfpack library: ${UMFPACK_LIBRARY}")
  add_library(UMFPACK INTERFACE)
  target_link_libraries(UMFPACK INTERFACE ${UMFPACK_LIBRARY})
  target_include_directories(UMFPACK INTERFACE ${UMFPACK_INCLUDE_DIR})
  message(STATUS "Found UMFPACK library: ${UMFPACK_LIBRARY}")
else()
  message(STATUS "UMFPACK was not found.")
endif()

set(UMFPACK_INCLUDE_DIR CACHE PATH "Path to umfpack.h")
set(UMFPACK_LIBRARY CACHE PATH "Path to umfpack library")
