
#[[

Exports target `COINHSL`

Users may set the following variables:

- HIOP_COINHSL_DIR

]]

find_library(COINHSL_LIBRARY
  NAMES
  coinhsl
  PATHS
  ${COINHSL_DIR} $ENV{COINHSL_DIR} ${HIOP_COINHSL_DIR}
  ENV LD_LIBRARY_PATH ENV DYLD_LIBRARY_PATH
  PATH_SUFFIXES
  lib64 lib)

if(COINHSL_LIBRARY)
  get_filename_component(COINHSL_LIBRARY_DIR ${COINHSL_LIBRARY} DIRECTORY)
endif()

find_path(COINHSL_INCLUDE_DIR
  NAMES
  CoinHslConfig.h
  PATHS
  ${COINHSL_DIR}/coin-or/hsl $ENV{COINHSL_DIR}/coin-or/hsl ${HIOP_COINHSL_DIR} ${COINHSL_LIBRARY_DIR}/../include/coin-or/hsl
  PATH_SUFFIXES
  include)

if(COINHSL_LIBRARY)
  message(STATUS "Found coinhsl include: ${COINHSL_INCLUDE_DIR}")
  message(STATUS "Found coinhsl library: ${COINHSL_LIBRARY}")
  add_library(COINHSL INTERFACE)
  target_link_libraries(COINHSL INTERFACE ${COINHSL_LIBRARY})
  target_include_directories(COINHSL INTERFACE ${COINHSL_INCLUDE_DIR})
  message(STATUS "Found COINHSL library: ${COINHSL_LIBRARY}")
else()
  message(STATUS "COINHSL was not found.")
endif()

set(COINHSL_INCLUDE_DIR CACHE PATH "Path to coinhsl.h")
set(COINHSL_LIBRARY CACHE PATH "Path to coinhsl library")
