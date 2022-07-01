set(hiopUtils_SRC
  hiopLogger.cpp
  hiopOptions.cpp
  MathHostKernels.cpp
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
  MathHostKernels.hpp
  )


if(HIOP_WITH_KRON_REDUCTION)
  list(APPEND hiopUtils_SRC ${hiopUtils_KRON_REDUCTION_SRC})
endif()

install(
  FILES ${hiopUtils_INTERFACE_HEADERS}
  DESTINATION include
  )

if(HIOP_USE_CUDA)
  set(hiopUtils_CUDA_SRC
    MathCudaKernels.cu
  )
  set_source_files_properties(${hiopUtils_CUDA_SRC} PROPERTIES LANGUAGE CUDA)
  list(APPEND hiopUtils_SRC ${hiopUtils_CUDA_SRC})
elseif(HIOP_USE_HIP)
  set(hiopUtils_HIP_SRC
    MathHipKernels.cpp
  )
# Mark source files as HIP. I guess in the future just a
# LANGUAGE HIP property will suffice. For now do it via compile flags
#  set_source_files_properties(${hiopUtils_HIP_SRC} PROPERTIES LANGUAGE HIP)
  set_source_files_properties( ${hiopUtils_HIP_SRC} PROPERTIES LANGUAGE CXX)
  set_source_files_properties( ${hiopUtils_HIP_SRC} PROPERTIES COMPILE_FLAGS "-x hip")
  list(APPEND hiopUtils_SRC ${hiopUtils_HIP_SRC})
endif(HIOP_USE_CUDA)

add_library(hiopUtils OBJECT ${hiopUtils_SRC})
target_link_libraries(hiopUtils PRIVATE hiop_tpl)
