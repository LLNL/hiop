
#[[

Exports target `STRUMPACK`

Users may set the following variables:

- HIOP_PARDISO_DIR

]]

find_library(PARDISO_LIBRARY
  NAMES
  pardiso
  PATHS
  ${PARDISO_DIR} $ENV{PARDISO_DIR} ${HIOP_PARDISO_DIR}
  ENV LD_LIBRARY_PATH ENV DYLD_LIBRARY_PATH
  PATH_SUFFIXES
  lib64 lib)

if(PARDISO_LIBRARY)
  get_filename_component(PARDISO_LIBRARY_DIR ${PARDISO_LIBRARY} DIRECTORY)
endif()

if(PARDISO_LIBRARY)
  message(STATUS "Found PARDISO library: ${PARDISO_LIBRARY}")
  add_library(PARDISO INTERFACE)
  target_link_libraries(PARDISO INTERFACE ${PARDISO_LIBRARY})
  message(STATUS "Found PARDISO library: ${PARDISO_LIBRARY}")
  install(TARGETS PARDISO EXPORT hiop-targets)
else()
  message(STATUS "PARDISO was not found.")
endif()

set(METIS_LIBRARY CACHE PATH "Path to PARDISO library")


