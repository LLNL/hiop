
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
    
if(STRUMPACK_LIBRARIES AND SCALAPACK_LIBRARY)
  message(STATUS "Found scalapack library: ${SCALAPACK_LIBRARY}")
  message(STATUS "Found strumpack library: ${STRUMPACK_LIBRARIES}")
  add_library(STRUMPACK INTERFACE)
  target_link_libraries(STRUMPACK INTERFACE STRUMPACK::strumpack ${SCALAPACK_LIBRARY})
#  target_include_directories(STRUMPACK INTERFACE ${STRUMPACK_INCLUDE_DIR})
  message(STATUS "Found STRUMPACK library: ${STRUMPACK_LIBRARIES} ${SCALAPACK_LIBRARY}")
else()
  message(STATUS "STRUMPACK was not found.")
endif()

#set(STRUMPACK_INCLUDE_DIR CACHE PATH "Path to StrumpackConfig.hpp")
#set(STRUMPACK_LIBRARIES CACHE PATH "Path to strumpack library")

