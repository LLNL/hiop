
# Create legacy library
add_library(hiopUtils OBJECT hiopLogger.cpp hiopOptions.cpp)
target_link_libraries(hiopUtils PUBLIC hiop_math)
if(HIOP_WITH_KRON_REDUCTION)
  add_library(hiopKronRed OBJECT hiopKronReduction.cpp)
  target_link_libraries(hiopKronRed PUBLIC hiop_math)
endif(HIOP_WITH_KRON_REDUCTION)
# ---

hiop_add_component(
  SOURCES
  hiopLogger.cpp
  hiopOptions.cpp
  $<$<BOOL:${HIOP_WITH_KRON_REDUCTION}>:hiopKronReduction.cpp>
  INTERFACE_HEADERS
  hiopCSR_IO.hpp
  hiopCppStdUtils.hpp
  hiopKronReduction.hpp
  hiopLogger.hpp
  hiopMPI.hpp
  hiopOptions.hpp
  hiopRunStats.hpp
  hiopTimer.hpp
  )
