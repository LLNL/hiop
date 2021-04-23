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
