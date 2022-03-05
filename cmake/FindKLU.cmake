
#[[

Looks for `klu` library and header directory.

Exports target `KLU` which links to umfpack.(so|a)
and add include directories where klu.h was found.

Users may set the following variables:

- HIOP_KLU_DIR

]]


find_library(KLU_LIBRARY
  NAMES
  klu
  PATHS
  ${KLU_DIR} $ENV{KLU_DIR} ${HIOP_KLU_DIR}
  ENV LD_LIBRARY_PATH ENV DYLD_LIBRARY_PATH
  PATH_SUFFIXES
  lib64 lib)

if(KLU_LIBRARY)
  get_filename_component(KLU_LIBRARY_DIR ${KLU_LIBRARY} DIRECTORY)
endif()

find_path(KLU_INCLUDE_DIR
  NAMES
  klu.h
  PATHS
  ${KLU_DIR} $ENV{KLU_DIR} ${HIOP_KLU_DIR} ${KLU_LIBRARY_DIR}/..
  PATH_SUFFIXES
  include
  include/suitesparse
  include/klu)

if(KLU_LIBRARY)
  message(STATUS "Found klu include: ${KLU_INCLUDE_DIR}")
  message(STATUS "Found klu library: ${KLU_LIBRARY}")
  add_library(KLU INTERFACE)
  target_link_libraries(KLU INTERFACE ${KLU_LIBRARY})
  target_include_directories(KLU INTERFACE ${KLU_INCLUDE_DIR})
  get_filename_component(KLU_LIB_DIR ${KLU_LIBRARY} DIRECTORY)
  set(CMAKE_INSTALL_RPATH "${KLU_LIB_DIR}")
  install(TARGETS KLU EXPORT hiop-targets)
else()
  message(STATUS "KLU was not found.")
endif()

set(KLU_INCLUDE_DIR CACHE PATH "Path to klu.h")
set(KLU_LIBRARY CACHE PATH "Path to klu library")

