
#[[

Exports target `STRUMPACK`

Users may set the following variables:

- HIOP_STRUMPACK_DIR
- HIOP_SCALAPACK_DIR

]]

find_package(STRUMPACK CONFIG
    PATHS ${STRUMPACK_DIR} ${HIOP_STRUMPACK_DIR}
    REQUIRED)
    
find_library(SCALAPACK_LIBRARY
  NAMES
  scalapack
  PATHS
  ${SCALAPACK_DIR} $ENV{SCALAPACK_DIR} ${HIOP_SCALAPACK_DIR}
  ENV LD_LIBRARY_PATH ENV DYLD_LIBRARY_PATH
  PATH_SUFFIXES
  lib64 lib)
    
if(STRUMPACK_LIBRARIES) 
  message(STATUS "Found STRUMPACK library: ${STRUMPACK_LIBRARIES}")
  add_library(STRUMPACK INTERFACE)
  target_link_libraries(STRUMPACK INTERFACE STRUMPACK::strumpack)
  install(TARGETS STRUMPACK EXPORT hiop-targets)

  # ignore SCALAPACK not_found: it may be that strumpack was built without MPI/SCALAPCK
  if(SCALAPACK_LIBRARY)
    target_link_libraries(STRUMPACK INTERFACE ${SCALAPACK_LIBRARY})
    message(STATUS "Found SCALAPACK library: ${SCALAPACK_LIBRARY}")
  endif(SCALAPACK_LIBRARY)
else()
  message(STATUS "STRUMPACK was not found.")
endif()

