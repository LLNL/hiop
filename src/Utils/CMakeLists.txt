set(hiopUtils_SRC
  hiopLogger.cpp
  hiopOptions.cpp
  MathKernelsHost.cpp
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
  MathKernelsHost.hpp
)

if(HIOP_USE_CUDA)
  list(APPEND hiopUtils_INTERFACE_HEADERS MathKernelsCuda.hpp)
endif()

if(HIOP_USE_HIP)
  list(APPEND hiopUtils_INTERFACE_HEADERS MathKernelsHip.hpp)
endif()

if(HIOP_USE_RAJA)
  list(APPEND hiopUtils_INTERFACE_HEADERS RajaUmpireUtils.hpp)
  list(APPEND hiopUtils_SRC ${hiopUtils_RAJA_SRC})
endif()

if(HIOP_USE_CUDA)
  # RAJA-related
  set_source_files_properties(${hiopUtils_RAJA_SRC} PROPERTIES LANGUAGE CUDA)

  # CUDA kernels
  set(hiopUtils_CUDA_SRC MathKernelsCuda.cu)
  set_source_files_properties(${hiopUtils_CUDA_SRC} PROPERTIES LANGUAGE CUDA)
  list(APPEND hiopUtils_SRC ${hiopUtils_CUDA_SRC})
endif()

if(HIOP_USE_HIP)
  # Mark appropriate source files as HIP. I guess in the future just a
  # LANGUAGE HIP property will suffice. For now do it via compile flags
  # set_source_files_properties(${hiopUtils_HIP_SRC} PROPERTIES LANGUAGE HIP)
  # Reference: https://www.olcf.ornl.gov/wp-content/uploads/2021/01/2021UM-Day-3-Joo-Developing-for-Frontier-using-HIP-on-Spock.pdf

  # RAJA-related
  set_source_files_properties(${hiopUtils_RAJA_SRC} PROPERTIES LANGUAGE CXX)
  set_source_files_properties(${hiopUtils_RAJA_SRC} PROPERTIES COMPILE_FLAGS "-x hip")

  # HIP kernels
  set(hiopUtils_HIP_SRC MathKernelsHip.cpp)
  set_source_files_properties( ${hiopUtils_HIP_SRC} PROPERTIES LANGUAGE CXX)
  set_source_files_properties( ${hiopUtils_HIP_SRC} PROPERTIES COMPILE_FLAGS "-x hip")
  list(APPEND hiopUtils_SRC ${hiopUtils_HIP_SRC})
endif()

if(HIOP_WITH_KRON_REDUCTION)
  list(APPEND hiopUtils_SRC ${hiopUtils_KRON_REDUCTION_SRC})
endif()

install(FILES ${hiopUtils_INTERFACE_HEADERS} DESTINATION include)

add_library(hiopUtils OBJECT ${hiopUtils_SRC})
target_link_libraries(hiopUtils PRIVATE hiop_tpl)
