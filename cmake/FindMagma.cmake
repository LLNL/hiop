
#[[

Exports target `Magma`.

Users may set the following variables:

- HIOP_MAGMA_DIR

]]

find_library(MAGMA_LIBRARY
  NAMES
  magma
  PATHS
  ${MAGMA_DIR} $ENV{MAGMA_DIR} ${HIOP_MAGMA_DIR}
  ENV LD_LIBRARY_PATH ENV DYLD_LIBRARY_PATH
  PATH_SUFFIXES
  lib64 lib)

if(MAGMA_LIBRARY)
  get_filename_component(MAGMA_LIBRARY_DIR ${MAGMA_LIBRARY} DIRECTORY)
endif()

find_path(MAGMA_INCLUDE_DIR
  NAMES
  magma.h
  PATHS
  ${MAGMA_DIR} $ENV{MAGMA_DIR} ${HIOP_MAGMA_DIR} ${MAGMA_LIBRARY_DIR}/..
  PATH_SUFFIXES
  include)

if(MAGMA_LIBRARY)
  add_library(Magma INTERFACE)
  target_link_libraries(Magma INTERFACE ${MAGMA_LIBRARY})
  target_include_directories(Magma INTERFACE ${MAGMA_INCLUDE_DIR})
  message(STATUS "Found Magma include: ${MAGMA_INCLUDE_DIR}")
  message(STATUS "Found Magma library: ${MAGMA_LIBRARY}")
else()
  message(STATUS "Magma was not found.")
endif()

set(MAGMA_INCLUDE_DIR CACHE PATH "Path to magma.h")
set(MAGMA_LIBRARY CACHE PATH "Path to magma library")
