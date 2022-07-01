set(hiopUtils_SRC
  hiopLogger.cpp
  hiopOptions.cpp
)

set(hiopUtils_RAJA_SRC
  RajaUmpireUtils.cpp
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

if(HIOP_USE_RAJA)
  list(APPEND hiopUtils_INTERFACE_HEADERS hiopRajaUmpireUtils.hpp)
  list(APPEND hiopUtils_SRC ${hiopUtils_RAJA_SRC})
endif()

if(HIOP_USE_CUDA)
  set_source_files_properties(${hiopUtils_RAJA_SRC} PROPERTIES LANGUAGE CUDA)
endif()
if(HIOP_USE_HIP)
  #this needs to be checked with hip/rocm
  set_source_files_properties(${hiopUtils_RAJA_SRC} PROPERTIES LANGUAGE CXX)
  set_source_files_properties(${hiopUtils_RAJA_SRC} PROPERTIES COMPILE_FLAGS "-x hip")
endif()

if(HIOP_WITH_KRON_REDUCTION)
  list(APPEND hiopUtils_SRC ${hiopUtils_KRON_REDUCTION_SRC})
endif()

install(FILES ${hiopUtils_INTERFACE_HEADERS} DESTINATION include)

add_library(hiopUtils OBJECT ${hiopUtils_SRC})
target_link_libraries(hiopUtils PRIVATE hiop_tpl)
