
# Create legacy library
add_library(hiopUtils OBJECT hiopLogger.cpp hiopOptions.cpp)
target_link_libraries(hiopUtils PUBLIC hiop_math)
if(HIOP_WITH_KRON_REDUCTION)
  add_library(hiopKronRed OBJECT hiopKronReduction.cpp)
  target_link_libraries(hiopKronRed PUBLIC hiop_math)
endif(HIOP_WITH_KRON_REDUCTION)
# ---

set(hiopUtils_SRC
  hiopLogger.cpp
  hiopOptions.cpp
  )

if(HIOP_WITH_KRON_REDUCTION)
  list(APPEND hiopUtils_SRC hiopKronReduction.cpp)
endif()

hiop_add_component(
  SOURCES
  ${hiopUtils_SRC}
  INTERFACE_HEADERS
  hiopCSR_IO.hpp
  hiopCppStdUtils.hpp
  hiopKronReduction.hpp
  hiopLogger.hpp
  hiopMPI.hpp
  hiopOptions.hpp
  hiopRunStats.hpp
  hiopTimer.hpp
  COMPONENT_NAME
  utils
  )
