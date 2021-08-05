set(hiopUtils_SRC
  hiopLogger.cpp
  hiopOptions.cpp
  )

set(hiopUtils_KRON_REDUCTION_SRC
  hiopKronReduction.cpp
  )

set(hiopUtils_INTERFACE_HEADERS
  hiopCSR_IO.hpp
  hiopCppStdUtils.hpp
  hiopKronReduction.hpp
  hiopLogger.hpp
  hiopMPI.hpp
  hiopOptions.hpp
  hiopRunStats.hpp
  hiopTimer.hpp
  )

if(HIOP_WITH_KRON_REDUCTION)
  list(APPEND hiopUtils_SRC ${hiopUtils_KRON_REDUCTION_SRC})
endif()

install(
  FILES ${hiopUtils_INTERFACE_HEADERS}
  DESTINATION include
  )

add_library(hiopUtils OBJECT ${hiopUtils_SRC})
target_link_libraries(hiopUtils PRIVATE hiop_tpl)
