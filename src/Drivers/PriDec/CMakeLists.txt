if(HIOP_USE_MPI)
  add_executable(nlpPriDec_EX1.exe nlpPriDec_EX1.cpp  nlpPriDec_EX1_driver.cpp)
  target_link_libraries(nlpPriDec_EX1.exe HiOp::HiOp)
endif()

if(HIOP_USE_MPI AND HIOP_SPARSE)
  include_directories("../Sparse")
  add_executable(nlpPriDec_EX2.exe nlpPriDec_EX2_driver.cpp  nlpPriDec_EX2.cpp ../Sparse/nlpSparse_EX1.cpp)
  target_link_libraries(nlpPriDec_EX2.exe HiOp::HiOp)
  add_executable(nlpPriDec_EX2_sparse.exe nlpPriDec_EX2_sparse_driver.cpp nlpPriDec_EX2_sparse.cpp ../Sparse/nlpSparse_EX1.cpp)
  target_link_libraries(nlpPriDec_EX2_sparse.exe HiOp::HiOp)
  add_executable(nlpPriDec_EX3_sparse.exe nlpPriDec_EX3_sparse_driver.cpp)
  target_link_libraries(nlpPriDec_EX3_sparse.exe HiOp::HiOp)

  if(HIOP_USE_RAJA)
    if(HIOP_USE_CUDA)
      set_source_files_properties(
        nlpPriDec_EX2_sparse_raja.cpp 
        nlpPriDec_EX2_sparse_raja_driver.cpp 
        PROPERTIES LANGUAGE CUDA)
    endif()
    add_executable(nlpPriDec_EX2_sparse_raja.exe nlpPriDec_EX2_sparse_raja_driver.cpp nlpPriDec_EX2_sparse_raja.cpp ../Sparse/nlpSparse_EX1.cpp)
    target_link_libraries(nlpPriDec_EX2_sparse_raja.exe HiOp::HiOp)
  endif()
endif()

##########################################################
# CMake Tests
##########################################################


  if(HIOP_USE_MPI)
    add_test(NAME NlpPriDec1_1 COMMAND ${RUNCMD} "$<TARGET_FILE:nlpPriDec_EX1.exe>" "-selfcheck")
    add_test(NAME NlpPriDec1_mpi COMMAND ${MPICMD} -n 2 "$<TARGET_FILE:nlpPriDec_EX1.exe>" "-selfcheck")
    if(HIOP_SPARSE)
      add_test(NAME NlpPriDec2_1 COMMAND ${RUNCMD} "$<TARGET_FILE:nlpPriDec_EX2.exe>" "-selfcheck")    
      add_test(NAME NlpPriDec2_mpi COMMAND ${MPICMD} -n 2 "$<TARGET_FILE:nlpPriDec_EX2.exe>" "-selfcheck")
    endif(HIOP_SPARSE)
  endif(HIOP_USE_MPI)


