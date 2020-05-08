
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
  PATH_SUFFIXES
  lib)

find_path(MAGMA_INCLUDE_DIR
  NAMES
  magma.h
  PATHS
  ${MAGMA_DIR} $ENV{MAGMA_DIR} ${HIOP_MAGMA_DIR}
  PATH_SUFFIXES
  include)

set(MAGMA_INCLUDE_DIR ${MAGMA_INCLUDE_DIR} CACHE PATH "Path to magma.h")
set(MAGMA_LIBRARY ${MAGMA_LIBRARY} CACHE PATH "Path to magma library")

if(MAGMA_LIBRARY)
  add_library(Magma INTERFACE)
  target_link_libraries(Magma INTERFACE ${MAGMA_LIBRARY})
  target_include_directories(Magma INTERFACE ${MAGMA_INCLUDE_DIR})
else()
  message(STATUS "Magma was not found.")
endif()

