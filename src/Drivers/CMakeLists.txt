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

add_executable(nlpDenseCons_ex1.exe ${CMAKE_CURRENT_SOURCE_DIR}/Dense/nlpDenseCons_ex1.cpp  ${CMAKE_CURRENT_SOURCE_DIR}/Dense/nlpDenseCons_ex1_driver.cpp)
target_link_libraries(nlpDenseCons_ex1.exe HiOp::HiOp)

add_executable(nlpDenseCons_ex2.exe  ${CMAKE_CURRENT_SOURCE_DIR}/Dense/nlpDenseCons_ex2.cpp  ${CMAKE_CURRENT_SOURCE_DIR}/Dense/nlpDenseCons_ex2_driver.cpp)
target_link_libraries(nlpDenseCons_ex2.exe HiOp::HiOp)

add_executable(nlpDenseCons_ex3.exe  ${CMAKE_CURRENT_SOURCE_DIR}/Dense/nlpDenseCons_ex3_driver.cpp)
target_link_libraries(nlpDenseCons_ex3.exe HiOp::HiOp)

add_executable(nlpMDS_ex4.exe  ${CMAKE_CURRENT_SOURCE_DIR}/MDS/nlpMDS_ex4_driver.cpp)
target_link_libraries(nlpMDS_ex4.exe HiOp::HiOp)

if(HIOP_USE_RAJA)
  if(HIOP_USE_GPU AND HIOP_USE_CUDA)
    set_source_files_properties(
      ${CMAKE_CURRENT_SOURCE_DIR}/MDS/nlpMDS_raja_ex4.cpp 
      ${CMAKE_CURRENT_SOURCE_DIR}/MDS/nlpMDS_ex4_raja_driver.cpp 
      PROPERTIES LANGUAGE CUDA
    )
  endif()
  add_executable(nlpMDS_ex4_raja.exe  ${CMAKE_CURRENT_SOURCE_DIR}/MDS/nlpMDS_ex4_raja_driver.cpp  ${CMAKE_CURRENT_SOURCE_DIR}/MDS/nlpMDS_raja_ex4.cpp)
  target_link_libraries(nlpMDS_ex4_raja.exe HiOp::HiOp)
endif()

add_executable(nlpMDS_ex5.exe  ${CMAKE_CURRENT_SOURCE_DIR}/MDS/nlpMDS_ex5_driver.cpp)
target_link_libraries(nlpMDS_ex5.exe HiOp::HiOp)

if(HIOP_USE_MPI)
  add_executable( hpc_multisolves.exe ${CMAKE_CURRENT_SOURCE_DIR}/MDS/hpc_multisolves.cpp)
  target_link_libraries(hpc_multisolves.exe HiOp::HiOp)
endif()

if(HIOP_BUILD_SHARED)
  add_executable(nlpMDS_cex4.exe ${CMAKE_CURRENT_SOURCE_DIR}/MDS/nlpMDS_ex4.c)
  target_link_libraries(nlpMDS_cex4.exe HiOp::HiOp)
endif()

if(HIOP_SPARSE)
    add_executable(nlpSparse_ex6.exe ${CMAKE_CURRENT_SOURCE_DIR}/Sparse/nlpSparse_ex6.cpp ${CMAKE_CURRENT_SOURCE_DIR}/Sparse/nlpSparse_ex6_driver.cpp)
    target_link_libraries(nlpSparse_ex6.exe HiOp::HiOp)

    add_executable(nlpSparse_ex7.exe ${CMAKE_CURRENT_SOURCE_DIR}/Sparse/nlpSparse_ex7.cpp ${CMAKE_CURRENT_SOURCE_DIR}/Sparse/nlpSparse_ex7_driver.cpp)
    target_link_libraries(nlpSparse_ex7.exe HiOp::HiOp)

    add_executable(nlpSparse_ex10.exe nlpSparse_ex10.cpp nlpSparse_ex10_driver.cpp)
    target_link_libraries(nlpSparse_ex10.exe HiOp::HiOp)
endif()

if(HIOP_USE_MPI)
  add_executable(nlpPriDec_ex8.exe ${CMAKE_CURRENT_SOURCE_DIR}/PriDec/nlpPriDec_ex8.cpp  ${CMAKE_CURRENT_SOURCE_DIR}/PriDec/nlpPriDec_ex8_driver.cpp)
  target_link_libraries(nlpPriDec_ex8.exe HiOp::HiOp)
endif()

if(HIOP_USE_MPI AND HIOP_SPARSE)
  add_executable(nlpPriDec_ex9.exe ${CMAKE_CURRENT_SOURCE_DIR}/PriDec/nlpPriDec_ex9_driver.cpp  ${CMAKE_CURRENT_SOURCE_DIR}/PriDec/nlpPriDec_ex9.cpp ${CMAKE_CURRENT_SOURCE_DIR}/Sparse/nlpSparse_ex6.cpp)
  target_link_libraries(nlpPriDec_ex9.exe HiOp::HiOp)
  add_executable(nlpPriDec_ex9_sparse.exe ${CMAKE_CURRENT_SOURCE_DIR}/PriDec/nlpPriDec_ex9_sparse_driver.cpp ${CMAKE_CURRENT_SOURCE_DIR}/PriDec/nlpPriDec_ex9_sparse.cpp ${CMAKE_CURRENT_SOURCE_DIR}/Sparse/nlpSparse_ex6.cpp)
  target_link_libraries(nlpPriDec_ex9_sparse.exe HiOp::HiOp)
  add_executable(nlpSparse_ex9.exe ${CMAKE_CURRENT_SOURCE_DIR}/Sparse/nlpSparse_ex9_driver.cpp)
  target_link_libraries(nlpSparse_ex9.exe HiOp::HiOp)

  if(HIOP_USE_RAJA)
    if(HIOP_USE_CUDA)
      set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/PriDec/nlpPriDec_ex9_sparse_raja.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/PriDec/nlpPriDec_ex9_sparse_raja_driver.cpp 
        PROPERTIES LANGUAGE CUDA)
    endif()
    add_executable(nlpPriDec_ex9_sparse_raja.exe ${CMAKE_CURRENT_SOURCE_DIR}/PriDec/nlpPriDec_ex9_sparse_raja_driver.cpp ${CMAKE_CURRENT_SOURCE_DIR}/PriDec/nlpPriDec_ex9_sparse_raja.cpp ${CMAKE_CURRENT_SOURCE_DIR}/Sparse/nlpSparse_ex6.cpp)
    target_link_libraries(nlpPriDec_ex9_sparse_raja.exe HiOp::HiOp)
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
  add_test(NAME NlpDenseCons1_5H  COMMAND ${RUNCMD} "$<TARGET_FILE:nlpDenseCons_ex1.exe>"  "500" "1.0" "-selfcheck")
  add_test(NAME NlpDenseCons1_5K  COMMAND ${RUNCMD} "$<TARGET_FILE:nlpDenseCons_ex1.exe>" "5000" "1.0" "-selfcheck")
  add_test(NAME NlpDenseCons1_50K COMMAND ${RUNCMD} "$<TARGET_FILE:nlpDenseCons_ex1.exe>" "50000" "1.0" "-selfcheck")
  if(HIOP_USE_MPI)
    add_test(NAME NlpDenseCons1_50K_mpi COMMAND ${MPICMD} -n 2 "$<TARGET_FILE:nlpDenseCons_ex1.exe>" "50000" "1.0" "-selfcheck")
  endif(HIOP_USE_MPI)
  add_test(NAME NlpDenseCons2_5H    COMMAND  ${RUNCMD} "$<TARGET_FILE:nlpDenseCons_ex2.exe>"   "500" "-selfcheck")
  add_test(NAME NlpDenseCons2_5K    COMMAND  ${RUNCMD} "$<TARGET_FILE:nlpDenseCons_ex2.exe>"  "5000" "-selfcheck")
  add_test(NAME NlpDenseCons2_UN_5K COMMAND  ${RUNCMD} "$<TARGET_FILE:nlpDenseCons_ex2.exe>"  "5000" "-unconstrained" "-selfcheck")
  add_test(NAME NlpDenseCons3_5H    COMMAND  ${RUNCMD} "$<TARGET_FILE:nlpDenseCons_ex3.exe>"   "500" "-selfcheck")
  add_test(NAME NlpDenseCons3_5K    COMMAND  ${RUNCMD} "$<TARGET_FILE:nlpDenseCons_ex3.exe>"  "5000" "-selfcheck")
  add_test(NAME NlpDenseCons3_50K   COMMAND  ${RUNCMD} "$<TARGET_FILE:nlpDenseCons_ex3.exe>" "50000" "-selfcheck")
  if(HIOP_USE_MPI)
    add_test(NAME NlpDenseCons3_50K_mpi COMMAND ${MPICMD} -n 2 "$<TARGET_FILE:nlpDenseCons_ex3.exe>" "50000" "-selfcheck")
  endif(HIOP_USE_MPI)

  set(STRIP_TABLE_CMD "awk '/Problem Summary/ { while (!match($0, /termination/)){ if(match($0, /^[ ]+[0-9]/)) { print $0; } getline } }'")
  add_test(NAME NlpMixedDenseSparse4_1 COMMAND ${RUNCMD} bash -c "$<TARGET_FILE:nlpMDS_ex4.exe> 400 100 0 -selfcheck \
    | ${STRIP_TABLE_CMD} \
    | tee ${HIOP_CTEST_OUTPUT_DIR}/mds4_1.out")
  add_test(NAME NlpMixedDenseSparse4_2 COMMAND ${RUNCMD} bash -c "$<TARGET_FILE:nlpMDS_ex4.exe> 400 100 1 -selfcheck \
    | ${STRIP_TABLE_CMD} \
    | tee ${HIOP_CTEST_OUTPUT_DIR}/mds4_2.out")
  
  add_test(NAME NlpMixedDenseSparse4_3 COMMAND ${RUNCMD} "$<TARGET_FILE:nlpMDS_ex4.exe>" "400" "100" "0" "-empty_sp_row" "-selfcheck")

  if(HIOP_USE_RAJA)
    add_test(NAME NlpMixedDenseSparseRaja4_1 COMMAND ${RUNCMD} bash -c "$<TARGET_FILE:nlpMDS_ex4_raja.exe> 400 100 0 -selfcheck \
      | ${STRIP_TABLE_CMD} \
      | tee ${HIOP_CTEST_OUTPUT_DIR}/mds4_raja_1.out")
    add_test(NAME NlpMixedDenseSparseRaja4_2 COMMAND ${RUNCMD} bash -c "$<TARGET_FILE:nlpMDS_ex4_raja.exe> 400 100 1 -selfcheck \
      | ${STRIP_TABLE_CMD} \
      | tee ${HIOP_CTEST_OUTPUT_DIR}/mds4_raja_2.out")
    add_test(NAME NlpMixedDenseSparseRaja4_3 COMMAND ${RUNCMD} bash -c "$<TARGET_FILE:nlpMDS_ex4_raja.exe> 400 100 0 -empty_sp_row -selfcheck")
    
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
          | ${PROJECT_SOURCE_DIR}/tests/testEx4CompareIterations.awk")
      endforeach()
    endif(HIOP_DEEPCHECKS)
  endif()

  add_test(NAME NlpMixedDenseSparse5_1 COMMAND ${RUNCMD} "$<TARGET_FILE:nlpMDS_ex5.exe>" "400" "100" "-selfcheck")
  
  if(HIOP_SPARSE)
    add_test(NAME NlpSparse6_1 COMMAND ${RUNCMD} "$<TARGET_FILE:nlpSparse_ex6.exe>" "500" "-selfcheck")
    add_test(NAME NlpSparse6_2 COMMAND ${RUNCMD} "$<TARGET_FILE:nlpSparse_ex6.exe>" "500" "-fr" "-selfcheck")
    if(HIOP_USE_PARDISO)
      add_test(NAME NlpSparse6_3 COMMAND ${RUNCMD} "$<TARGET_FILE:nlpSparse_ex6.exe>" "500" "-pardiso" "-selfcheck")
    endif(HIOP_USE_PARDISO)
    add_test(NAME NlpSparse7_1 COMMAND ${RUNCMD} "$<TARGET_FILE:nlpSparse_ex7.exe>" "500" "-selfcheck")
    add_test(NAME NlpSparse7_2 COMMAND ${RUNCMD} "$<TARGET_FILE:nlpSparse_ex7.exe>" "500" "-inertiafree" "-selfcheck")
  endif(HIOP_SPARSE)

  if(HIOP_USE_MPI)
    add_test(NAME NlpPriDec8_1 COMMAND ${RUNCMD} "$<TARGET_FILE:nlpPriDec_ex8.exe>" "-selfcheck")
    add_test(NAME NlpPriDec8_mpi COMMAND ${MPICMD} -n 2 "$<TARGET_FILE:nlpPriDec_ex8.exe>" "-selfcheck")
    if(HIOP_SPARSE)
      add_test(NAME NlpPriDec9_1 COMMAND ${RUNCMD} "$<TARGET_FILE:nlpPriDec_ex9.exe>" "-selfcheck")    
      add_test(NAME NlpPriDec9_mpi COMMAND ${MPICMD} -n 2 "$<TARGET_FILE:nlpPriDec_ex9.exe>" "-selfcheck")
    endif(HIOP_SPARSE)
  endif(HIOP_USE_MPI)

  if(HIOP_WITH_VALGRIND_TESTS)
    string(REPLACE ";" " " runcmd_str "${RUNCMD}")
    add_test(
      NAME NlpDenseCons1_5H_Valgrind
      COMMAND bash -c "${runcmd_str} ${HIOP_VALGRIND_CMD} $<TARGET_FILE:nlpDenseCons_ex1.exe> 500 1.0 -selfcheck"
      )
    add_test(
      NAME NlpMixedDenseSparse4_1_Valgrind
      COMMAND bash -c "${runcmd_str} ${HIOP_VALGRIND_CMD} $<TARGET_FILE:nlpMDS_ex4.exe> 400 100 0 -selfcheck"
      )
  endif()

  if(HIOP_BUILD_SHARED AND NOT HIOP_USE_GPU)
    add_test(NAME NlpMixedDenseSparseCinterface COMMAND ${RUNCMD} "$<TARGET_FILE:nlpMDS_cex4.exe>")
  endif()

endif(HIOP_WITH_MAKETEST)

