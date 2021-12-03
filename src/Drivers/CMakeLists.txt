
set(drivers
  )

install(
  FILES IpoptAdapter.hpp
  DESTINATION include
  )

#add_executable(linalgTest.exe linalgTest.cpp)
#target_link_libraries(linalgTest.exe hiop ${HIOP_MATH_LIBRARIES})

add_executable(nlpDenseCons_ex1.exe nlpDenseCons_ex1.cpp nlpDenseCons_ex1_driver.cpp)
target_link_libraries(nlpDenseCons_ex1.exe HiOp::HiOp)

add_executable(nlpDenseCons_ex2.exe nlpDenseCons_ex2.cpp nlpDenseCons_ex2_driver.cpp)
target_link_libraries(nlpDenseCons_ex2.exe HiOp::HiOp)

add_executable(nlpDenseCons_ex3.exe nlpDenseCons_ex3_driver.cpp)
target_link_libraries(nlpDenseCons_ex3.exe HiOp::HiOp)

add_executable(nlpMDS_ex4.exe nlpMDS_ex4_driver.cpp)
target_link_libraries(nlpMDS_ex4.exe HiOp::HiOp)

list(APPEND drivers
  nlpDenseCons_ex1.exe 
  nlpDenseCons_ex2.exe 
  nlpDenseCons_ex3.exe 
  nlpMDS_ex4.exe
  )

if(HIOP_USE_RAJA)
  if(HIOP_USE_GPU AND HIOP_USE_CUDA)
    set_source_files_properties(
      nlpMDS_raja_ex4.cpp 
      nlpMDS_ex4_raja_driver.cpp 
      PROPERTIES LANGUAGE CUDA
    )
  endif()
  add_executable(nlpMDS_ex4_raja.exe nlpMDS_ex4_raja_driver.cpp nlpMDS_raja_ex4.cpp)
  target_link_libraries(nlpMDS_ex4_raja.exe HiOp::HiOp)
  list(APPEND drivers nlpMDS_ex4_raja.exe)
endif()

add_executable(nlpMDS_ex5.exe nlpMDS_ex5_driver.cpp)
target_link_libraries(nlpMDS_ex5.exe HiOp::HiOp)
list(APPEND drivers nlpMDS_ex5.exe)

if(HIOP_USE_MPI)
  add_executable(hpc_multisolves.exe hpc_multisolves.cpp)
  target_link_libraries(hpc_multisolves.exe HiOp::HiOp)
  list(APPEND drivers hpc_multisolves.exe)
endif()

if(HIOP_SPARSE)
    add_executable(nlpSparse_ex6.exe nlpSparse_ex6.cpp nlpSparse_ex6_driver.cpp)
    target_link_libraries(nlpSparse_ex6.exe HiOp::HiOp)

    add_executable(nlpSparse_ex7.exe nlpSparse_ex7.cpp nlpSparse_ex7_driver.cpp)
    target_link_libraries(nlpSparse_ex7.exe HiOp::HiOp)

    list(APPEND drivers nlpSparse_ex6.exe nlpSparse_ex7.exe)
endif()

if(HIOP_BUILD_SHARED)
  add_executable(nlpMDS_cex4.exe nlpMDS_ex4.c)
  target_link_libraries(nlpMDS_cex4.exe HiOp::HiOp)
  list(APPEND drivers nlpMDS_cex4.exe)
endif()

if(HIOP_USE_MPI)
  add_executable(nlpPriDec_ex8.exe nlpPriDec_ex8.cpp nlpPriDec_ex8_driver.cpp)
  target_link_libraries(nlpPriDec_ex8.exe HiOp::HiOp)
  list(APPEND drivers nlpPriDec_ex8.exe)
endif()

if(HIOP_USE_MPI AND HIOP_SPARSE)
  add_executable(nlpPriDec_ex9.exe nlpPriDec_ex9_driver.cpp nlpPriDec_ex9.cpp nlpSparse_ex6.cpp)
  target_link_libraries(nlpPriDec_ex9.exe HiOp::HiOp)
  add_executable(nlpSparse_ex9.exe nlpSparse_ex9_driver.cpp)
  target_link_libraries(nlpSparse_ex9.exe HiOp::HiOp)
  list(APPEND drivers nlpPriDec_ex9.exe nlpSparse_ex9.exe)
endif()

if(HIOP_USE_CUDA)
  foreach(target ${drivers})
    set_property(TARGET ${target} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  endforeach()
endif()
