include_directories(Dense)
include_directories(MDS)
include_directories(Sparse)
include_directories(PriDec)
include_directories(IpoptAdapter)

install(
  FILES IpoptAdapter.hpp
  DESTINATION include
  )

#add_executable(linalgTest.exe linalgTest.cpp)
#target_link_libraries(linalgTest.exe hiop ${HIOP_MATH_LIBRARIES})

add_executable(nlpDenseCons_EX1.exe ${CMAKE_CURRENT_SOURCE_DIR}/Dense/nlpDenseCons_EX1.cpp  ${CMAKE_CURRENT_SOURCE_DIR}/Dense/nlpDenseCons_EX1_driver.cpp)
target_link_libraries(nlpDenseCons_EX1.exe HiOp::HiOp)

add_executable(nlpDenseCons_EX2.exe  ${CMAKE_CURRENT_SOURCE_DIR}/Dense/nlpDenseCons_EX2.cpp  ${CMAKE_CURRENT_SOURCE_DIR}/Dense/nlpDenseCons_EX2_driver.cpp)
target_link_libraries(nlpDenseCons_EX2.exe HiOp::HiOp)

add_executable(nlpDenseCons_EX3.exe  ${CMAKE_CURRENT_SOURCE_DIR}/Dense/nlpDenseCons_EX3_driver.cpp)
target_link_libraries(nlpDenseCons_EX3.exe HiOp::HiOp)

add_executable(nlpMDS_EX1.exe  ${CMAKE_CURRENT_SOURCE_DIR}/MDS/nlpMDS_EX1_driver.cpp)
target_link_libraries(nlpMDS_EX1.exe HiOp::HiOp)

if(HIOP_USE_RAJA)
  if(HIOP_USE_GPU AND HIOP_USE_CUDA)
    set_source_files_properties(
      ${CMAKE_CURRENT_SOURCE_DIR}/MDS/nlpMDS_raja_EX1.cpp 
      ${CMAKE_CURRENT_SOURCE_DIR}/MDS/nlpMDS_EX1_raja_driver.cpp 
      PROPERTIES LANGUAGE CUDA
    )
  endif()
  add_executable(nlpMDS_EX1_raja.exe  ${CMAKE_CURRENT_SOURCE_DIR}/MDS/nlpMDS_EX1_raja_driver.cpp  ${CMAKE_CURRENT_SOURCE_DIR}/MDS/nlpMDS_raja_EX1.cpp)
  target_link_libraries(nlpMDS_EX1_raja.exe HiOp::HiOp)
endif()

add_executable(nlpMDS_EX2.exe  ${CMAKE_CURRENT_SOURCE_DIR}/MDS/nlpMDS_EX2_driver.cpp)
target_link_libraries(nlpMDS_EX2.exe HiOp::HiOp)

if(HIOP_USE_MPI)
  add_executable( hpc_multisolves.exe ${CMAKE_CURRENT_SOURCE_DIR}/MDS/hpc_multisolves.cpp)
  target_link_libraries(hpc_multisolves.exe HiOp::HiOp)
endif()

if(HIOP_BUILD_SHARED)
  add_executable(nlpMDS_cEx1.exe ${CMAKE_CURRENT_SOURCE_DIR}/MDS/nlpMDS_EX1.c)
  target_link_libraries(nlpMDS_cEx1.exe HiOp::HiOp)
endif()

if(HIOP_SPARSE)
    add_executable(nlpSparse_EX1.exe ${CMAKE_CURRENT_SOURCE_DIR}/Sparse/nlpSparse_EX1.cpp ${CMAKE_CURRENT_SOURCE_DIR}/Sparse/nlpSparse_EX1_driver.cpp)
    target_link_libraries(nlpSparse_EX1.exe HiOp::HiOp)

    add_executable(nlpSparse_EX2.exe ${CMAKE_CURRENT_SOURCE_DIR}/Sparse/nlpSparse_EX2.cpp ${CMAKE_CURRENT_SOURCE_DIR}/Sparse/nlpSparse_EX2_driver.cpp)
    target_link_libraries(nlpSparse_EX2.exe HiOp::HiOp)

    add_executable(nlpSparse_EX3.exe nlpSparse_EX3.cpp nlpSparse_EX3_driver.cpp)
    target_link_libraries(nlpSparse_EX3.exe HiOp::HiOp)
endif()

if(HIOP_USE_MPI)
  add_executable(nlpPriDec_EX1.exe ${CMAKE_CURRENT_SOURCE_DIR}/PriDec/nlpPriDec_EX1.cpp  ${CMAKE_CURRENT_SOURCE_DIR}/PriDec/nlpPriDec_EX1_driver.cpp)
  target_link_libraries(nlpPriDec_EX1.exe HiOp::HiOp)
endif()

if(HIOP_USE_MPI AND HIOP_SPARSE)
  add_executable(nlpPriDec_EX2.exe ${CMAKE_CURRENT_SOURCE_DIR}/PriDec/nlpPriDec_EX2_driver.cpp  ${CMAKE_CURRENT_SOURCE_DIR}/PriDec/nlpPriDec_EX2.cpp ${CMAKE_CURRENT_SOURCE_DIR}/Sparse/nlpSparse_EX1.cpp)
  target_link_libraries(nlpPriDec_EX2.exe HiOp::HiOp)
  add_executable(nlpPriDec_EX2_sparse.exe ${CMAKE_CURRENT_SOURCE_DIR}/PriDec/nlpPriDec_EX2_sparse_driver.cpp ${CMAKE_CURRENT_SOURCE_DIR}/PriDec/nlpPriDec_EX2_sparse.cpp ${CMAKE_CURRENT_SOURCE_DIR}/Sparse/nlpSparse_EX1.cpp)
  target_link_libraries(nlpPriDec_EX2_sparse.exe HiOp::HiOp)
  add_executable(nlpPriDec_EX3_sparse.exe ${CMAKE_CURRENT_SOURCE_DIR}/PriDec/nlpPriDec_EX3_sparse_driver.cpp)
  target_link_libraries(nlpPriDec_EX3_sparse.exe HiOp::HiOp)

  if(HIOP_USE_RAJA)
    if(HIOP_USE_CUDA)
      set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/PriDec/nlpPriDec_EX2_sparse_raja.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/PriDec/nlpPriDec_EX2_sparse_raja_driver.cpp 
        PROPERTIES LANGUAGE CUDA)
    endif()
    add_executable(nlpPriDec_EX2_sparse_raja.exe ${CMAKE_CURRENT_SOURCE_DIR}/PriDec/nlpPriDec_EX2_sparse_raja_driver.cpp ${CMAKE_CURRENT_SOURCE_DIR}/PriDec/nlpPriDec_EX2_sparse_raja.cpp ${CMAKE_CURRENT_SOURCE_DIR}/Sparse/nlpSparse_EX1.cpp)
    target_link_libraries(nlpPriDec_EX2_sparse_raja.exe HiOp::HiOp)
  endif()
endif()



##########################################################
# CMake Tests
##########################################################
# If running in a BSUB allocaation, use jrun commands and make sure we are
# requesting the correct resources.
if(HIOP_TEST_WITH_BSUB)
  set(RUNCMD "jsrun" "-n" "1")
  set(MPICMD "jsrun")
  if(HIOP_USE_GPU)
    set(MPICMD ${MPICMD} "-g" "1")
    set(RUNCMD ${RUNCMD} "-g" "1")
  endif()
else()
  set(MPICMD "mpirun")
  if(HIOP_USE_MPI)
    set(RUNCMD "mpirun" "-n" "1")
    set(RUNCMD ${RUNCMD} ${HIOP_EXTRA_MPI_FLAGS})
    set(MPICMD ${MPICMD} ${HIOP_EXTRA_MPI_FLAGS})
  else()
    set(RUNCMD "") # No special command is needed to run this program  
  endif()
endif()

if(NOT (HIOP_CTEST_LAUNCH_COMMAND STREQUAL ""))
  message(STATUS "Manually overriding launch command for CTest with \"${HIOP_CTEST_LAUNCH_COMMAND}\"")
  string(REPLACE " " ";" TMP_RUNCMD "${HIOP_CTEST_LAUNCH_COMMAND}")
  set(RUNCMD ${TMP_RUNCMD})
  set(MPICMD ${TMP_RUNCMD})
endif()

if (HIOP_WITH_MAKETEST)
  enable_testing()
  add_test(NAME NlpDenseCons1_5H  COMMAND ${RUNCMD} "$<TARGET_FILE:nlpDenseCons_EX1.exe>"  "500" "1.0" "-selfcheck")
  add_test(NAME NlpDenseCons1_5K  COMMAND ${RUNCMD} "$<TARGET_FILE:nlpDenseCons_EX1.exe>" "5000" "1.0" "-selfcheck")
  add_test(NAME NlpDenseCons1_50K COMMAND ${RUNCMD} "$<TARGET_FILE:nlpDenseCons_EX1.exe>" "50000" "1.0" "-selfcheck")
  if(HIOP_USE_MPI)
    add_test(NAME NlpDenseCons1_50K_mpi COMMAND ${MPICMD} -n 2 "$<TARGET_FILE:nlpDenseCons_EX1.exe>" "50000" "1.0" "-selfcheck")
  endif(HIOP_USE_MPI)
  add_test(NAME NlpDenseCons2_5H    COMMAND  ${RUNCMD} "$<TARGET_FILE:nlpDenseCons_EX2.exe>"   "500" "-selfcheck")
  add_test(NAME NlpDenseCons2_5K    COMMAND  ${RUNCMD} "$<TARGET_FILE:nlpDenseCons_EX2.exe>"  "5000" "-selfcheck")
  add_test(NAME NlpDenseCons2_UN_5K COMMAND  ${RUNCMD} "$<TARGET_FILE:nlpDenseCons_EX2.exe>"  "5000" "-unconstrained" "-selfcheck")
  add_test(NAME NlpDenseCons3_5H    COMMAND  ${RUNCMD} "$<TARGET_FILE:nlpDenseCons_EX3.exe>"   "500" "-selfcheck")
  add_test(NAME NlpDenseCons3_5K    COMMAND  ${RUNCMD} "$<TARGET_FILE:nlpDenseCons_EX3.exe>"  "5000" "-selfcheck")
  add_test(NAME NlpDenseCons3_50K   COMMAND  ${RUNCMD} "$<TARGET_FILE:nlpDenseCons_EX3.exe>" "50000" "-selfcheck")
  if(HIOP_USE_MPI)
    add_test(NAME NlpDenseCons3_50K_mpi COMMAND ${MPICMD} -n 2 "$<TARGET_FILE:nlpDenseCons_EX3.exe>" "50000" "-selfcheck")
  endif(HIOP_USE_MPI)

  set(STRIP_TABLE_CMD "awk '/Problem Summary/ { while (!match($0, /termination/)){ if(match($0, /^[ ]+[0-9]/)) { print $0; } getline } }'")
  add_test(NAME NlpMixedDenseSparse4_1 COMMAND ${RUNCMD} bash -c "$<TARGET_FILE:nlpMDS_EX1.exe> 400 100 0 -selfcheck \
    | ${STRIP_TABLE_CMD} \
    | tee ${HIOP_CTEST_OUTPUT_DIR}/mds4_1.out")
  add_test(NAME NlpMixedDenseSparse4_2 COMMAND ${RUNCMD} bash -c "$<TARGET_FILE:nlpMDS_EX1.exe> 400 100 1 -selfcheck \
    | ${STRIP_TABLE_CMD} \
    | tee ${HIOP_CTEST_OUTPUT_DIR}/mds4_2.out")
  
  add_test(NAME NlpMixedDenseSparse4_3 COMMAND ${RUNCMD} "$<TARGET_FILE:nlpMDS_EX1.exe>" "400" "100" "0" "-empty_sp_row" "-selfcheck")

  if(HIOP_USE_RAJA)
    add_test(NAME NlpMixedDenseSparseRaja4_1 COMMAND ${RUNCMD} bash -c "$<TARGET_FILE:nlpMDS_EX1_raja.exe> 400 100 0 -selfcheck \
      | ${STRIP_TABLE_CMD} \
      | tee ${HIOP_CTEST_OUTPUT_DIR}/mds4_raja_1.out")
    add_test(NAME NlpMixedDenseSparseRaja4_2 COMMAND ${RUNCMD} bash -c "$<TARGET_FILE:nlpMDS_EX1_raja.exe> 400 100 1 -selfcheck \
      | ${STRIP_TABLE_CMD} \
      | tee ${HIOP_CTEST_OUTPUT_DIR}/mds4_raja_2.out")
    add_test(NAME NlpMixedDenseSparseRaja4_3 COMMAND ${RUNCMD} bash -c "$<TARGET_FILE:nlpMDS_EX1_raja.exe> 400 100 0 -empty_sp_row -selfcheck")
    
    if(HIOP_DEEPCHECKS)
      foreach(iter 1 2)
        add_test(
          NAME "CompareExample4_NumIterations_${iter}" 
          COMMAND bash -c "\
          if [[ $(wc -l ${HIOP_CTEST_OUTPUT_DIR}/mds4_${iter}.out|cut -f1 -d' ') == $(wc -l ${HIOP_CTEST_OUTPUT_DIR}/mds4_raja_${iter}.out|cut -f1 -d' ') ]]
          then
          echo 'Output tables have the same number of iterations.'
          exit 0
          else
          echo 'Output tables have a different number of iterations!'
          exit 1
          fi")
        add_test(
          NAME "CompareExample4_ElementWise_${iter}"
          COMMAND bash -c "\
          join ${HIOP_CTEST_OUTPUT_DIR}/mds4_${iter}.out ${HIOP_CTEST_OUTPUT_DIR}/mds4_raja_${iter}.out \
          | ${PROJECT_SOURCE_DIR}/tests/testEx1CompareIterations.awk")
      endforeach()
    endif(HIOP_DEEPCHECKS)
  endif()

  add_test(NAME NlpMixedDenseSparse5_1 COMMAND ${RUNCMD} "$<TARGET_FILE:nlpMDS_EX2.exe>" "400" "100" "-selfcheck")
  
  if(HIOP_SPARSE)
    add_test(NAME NlpSparse6_1 COMMAND ${RUNCMD} "$<TARGET_FILE:nlpSparse_EX1.exe>" "500" "-selfcheck")
    add_test(NAME NlpSparse6_2 COMMAND ${RUNCMD} "$<TARGET_FILE:nlpSparse_EX1.exe>" "500" "-fr" "-selfcheck")
    if(HIOP_USE_PARDISO)
      add_test(NAME NlpSparse6_3 COMMAND ${RUNCMD} "$<TARGET_FILE:nlpSparse_EX1.exe>" "500" "-pardiso" "-selfcheck")
    endif(HIOP_USE_PARDISO)
    add_test(NAME NlpSparse7_1 COMMAND ${RUNCMD} "$<TARGET_FILE:nlpSparse_EX2.exe>" "500" "-selfcheck")
    add_test(NAME NlpSparse7_2 COMMAND ${RUNCMD} "$<TARGET_FILE:nlpSparse_EX2.exe>" "500" "-inertiafree" "-selfcheck")
  endif(HIOP_SPARSE)

  if(HIOP_USE_MPI)
    add_test(NAME NlpPriDec8_1 COMMAND ${RUNCMD} "$<TARGET_FILE:nlpPriDec_EX1.exe>" "-selfcheck")
    add_test(NAME NlpPriDec8_mpi COMMAND ${MPICMD} -n 2 "$<TARGET_FILE:nlpPriDec_EX1.exe>" "-selfcheck")
    if(HIOP_SPARSE)
      add_test(NAME NlpPriDec9_1 COMMAND ${RUNCMD} "$<TARGET_FILE:nlpPriDec_EX2.exe>" "-selfcheck")    
      add_test(NAME NlpPriDec9_mpi COMMAND ${MPICMD} -n 2 "$<TARGET_FILE:nlpPriDec_EX2.exe>" "-selfcheck")
    endif(HIOP_SPARSE)
  endif(HIOP_USE_MPI)

  if(HIOP_WITH_VALGRIND_TESTS)
    string(REPLACE ";" " " runcmd_str "${RUNCMD}")
    add_test(
      NAME NlpDenseCons1_5H_Valgrind
      COMMAND bash -c "${runcmd_str} ${HIOP_VALGRIND_CMD} $<TARGET_FILE:nlpDenseCons_EX1.exe> 500 1.0 -selfcheck"
      )
    add_test(
      NAME NlpMixedDenseSparse4_1_Valgrind
      COMMAND bash -c "${runcmd_str} ${HIOP_VALGRIND_CMD} $<TARGET_FILE:nlpMDS_EX1.exe> 400 100 0 -selfcheck"
      )
  endif()

  if(HIOP_BUILD_SHARED AND NOT HIOP_USE_GPU)
    add_test(NAME NlpMixedDenseSparseCinterface COMMAND ${RUNCMD} "$<TARGET_FILE:nlpMDS_cEx1.exe>")
  endif()

endif(HIOP_WITH_MAKETEST)

