
#[[

Exports target `METIS`

Users may set the following variables:

- HIOP_METIS_DIR

]]

find_library(METIS_LIBRARY
  NAMES
  metis
  PATHS
  ${METIS_DIR} $ENV{METIS_DIR} ${HIOP_METIS_DIR}
  PATH_SUFFIXES
  lib)

find_path(METIS_INCLUDE_DIR
  NAMES
  metis.h
  PATHS
  ${METIS_DIR} $ENV{METIS_DIR} ${HIOP_METIS_DIR}
  PATH_SUFFIXES
  include)

set(METIS_INCLUDE_DIR ${METIS_INCLUDE_DIR} CACHE PATH "Path to metis.h")
set(METIS_LIBRARY ${METIS_LIBRARY} CACHE PATH "Path to metis library")

if(METIS_LIBRARY)
  add_library(METIS INTERFACE)
  target_link_libraries(METIS INTERFACE ${METIS_LIBRARY})
  target_include_directories(METIS INTERFACE ${METIS_INCLUDE_DIR})
else()
  message(STATUS "METIS was not found.")
endif()
