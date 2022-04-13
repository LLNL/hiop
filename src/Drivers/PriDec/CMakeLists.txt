if(HIOP_USE_MPI)
  add_executable(NlpPriDecEx1.exe NlpPriDecEx1.cpp  NlpPriDecEx1Driver.cpp)
  target_link_libraries(NlpPriDecEx1.exe HiOp::HiOp)
endif()

if(HIOP_USE_MPI AND HIOP_SPARSE)
  include_directories("../Sparse")
  add_executable(NlpPriDecEx2.exe NlpPriDecEx2Driver.cpp  NlpPriDecEx2.cpp ../Sparse/NlpSparseEx1.cpp)
  target_link_libraries(NlpPriDecEx2.exe HiOp::HiOp)
  add_executable(NlpPriDecEx2Sparse.exe NlpPriDecEx2SparseDriver.cpp NlpPriDecEx2Sparse.cpp ../Sparse/NlpSparseEx1.cpp)
  target_link_libraries(NlpPriDecEx2Sparse.exe HiOp::HiOp)
  add_executable(NlpPriDecEx3Sparse.exe NlpPriDecEx3SparseDriver.cpp)
  target_link_libraries(NlpPriDecEx3Sparse.exe HiOp::HiOp)

  if(HIOP_USE_RAJA)
    if(HIOP_USE_CUDA)
      set_source_files_properties(
        NlpPriDecEx2SparseRaja.cpp 
        NlpPriDecEx2SparseRajaDriver.cpp 
        PROPERTIES LANGUAGE CUDA)
    endif()
    add_executable(NlpPriDecEx2SparseRaja.exe NlpPriDecEx2SparseRajaDriver.cpp NlpPriDecEx2SparseRaja.cpp ../Sparse/NlpSparseEx1.cpp)
    target_link_libraries(NlpPriDecEx2SparseRaja.exe HiOp::HiOp)
  endif()
endif()

##########################################################
# CMake Tests
##########################################################
if(HIOP_USE_MPI)
  add_test(NAME NlpPriDec1_1 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpPriDecEx1.exe>" "-selfcheck")
  add_test(NAME NlpPriDec1_mpi COMMAND ${MPICMD} -n 2 "$<TARGET_FILE:NlpPriDecEx1.exe>" "-selfcheck")
  if(HIOP_SPARSE)
    add_test(NAME NlpPriDec2_1 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpPriDecEx2.exe>" "-selfcheck")    
    add_test(NAME NlpPriDec2_mpi COMMAND ${MPICMD} -n 2 "$<TARGET_FILE:NlpPriDecEx2.exe>" "-selfcheck")
  endif(HIOP_SPARSE)
endif(HIOP_USE_MPI)


