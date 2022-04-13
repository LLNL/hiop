add_executable(nlpMDS_EX1.exe nlpMDS_EX1_driver.cpp)
target_link_libraries(nlpMDS_EX1.exe HiOp::HiOp)

if(HIOP_USE_RAJA)
  if(HIOP_USE_GPU AND HIOP_USE_CUDA)
    set_source_files_properties(
      nlpMDS_raja_EX1.cpp 
      nlpMDS_EX1_raja_driver.cpp 
      PROPERTIES LANGUAGE CUDA
    )
  endif()
  add_executable(nlpMDS_EX1_raja.exe  nlpMDS_EX1_raja_driver.cpp  nlpMDS_raja_EX1.cpp)
  target_link_libraries(nlpMDS_EX1_raja.exe HiOp::HiOp)
endif()

add_executable(nlpMDS_EX2.exe  nlpMDS_EX2_driver.cpp)
target_link_libraries(nlpMDS_EX2.exe HiOp::HiOp)

if(HIOP_USE_MPI)
  add_executable( hpc_multisolves.exe hpc_multisolves.cpp)
  target_link_libraries(hpc_multisolves.exe HiOp::HiOp)
endif()

if(HIOP_BUILD_SHARED)
  add_executable(nlpMDS_cEx1.exe nlpMDS_EX1.c)
  target_link_libraries(nlpMDS_cEx1.exe HiOp::HiOp)
endif()

##########################################################
# CMake Tests
##########################################################

set(STRIP_TABLE_CMD "awk '/Problem Summary/ { while (!match($0, /termination/)){ if(match($0, /^[ ]+[0-9]/)) { print $0; } getline } }'")
add_test(NAME NlpMixedDenseSparse1_1 COMMAND ${RUNCMD} bash -c "$<TARGET_FILE:nlpMDS_EX1.exe> 400 100 0 -selfcheck \
  | ${STRIP_TABLE_CMD} \
  | tee ${HIOP_CTEST_OUTPUT_DIR}/mds4_1.out")
add_test(NAME NlpMixedDenseSparse1_2 COMMAND ${RUNCMD} bash -c "$<TARGET_FILE:nlpMDS_EX1.exe> 400 100 1 -selfcheck \
  | ${STRIP_TABLE_CMD} \
  | tee ${HIOP_CTEST_OUTPUT_DIR}/mds4_2.out")

add_test(NAME NlpMixedDenseSparse1_3 COMMAND ${RUNCMD} "$<TARGET_FILE:nlpMDS_EX1.exe>" "400" "100" "0" "-empty_sp_row" "-selfcheck")

if(HIOP_USE_RAJA)
  add_test(NAME NlpMixedDenseSparseRaja1_1 COMMAND ${RUNCMD} bash -c "$<TARGET_FILE:nlpMDS_EX1_raja.exe> 400 100 0 -selfcheck \
    | ${STRIP_TABLE_CMD} \
    | tee ${HIOP_CTEST_OUTPUT_DIR}/mds4_raja_1.out")
  add_test(NAME NlpMixedDenseSparseRaja1_2 COMMAND ${RUNCMD} bash -c "$<TARGET_FILE:nlpMDS_EX1_raja.exe> 400 100 1 -selfcheck \
    | ${STRIP_TABLE_CMD} \
    | tee ${HIOP_CTEST_OUTPUT_DIR}/mds4_raja_2.out")
  add_test(NAME NlpMixedDenseSparseRaja1_3 COMMAND ${RUNCMD} bash -c "$<TARGET_FILE:nlpMDS_EX1_raja.exe> 400 100 0 -empty_sp_row -selfcheck")
  
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
        | ${PROJECT_SOURCE_DIR}/tests/testMDS1CompareIterations.awk")
    endforeach()
  endif(HIOP_DEEPCHECKS)
endif()

add_test(NAME NlpMixedDenseSparse2_1 COMMAND ${RUNCMD} "$<TARGET_FILE:nlpMDS_EX2.exe>" "400" "100" "-selfcheck")


if(HIOP_WITH_VALGRIND_TESTS)
  string(REPLACE ";" " " runcmd_str "${RUNCMD}")
  add_test(
    NAME NlpMixedDenseSparse1_1_Valgrind
    COMMAND bash -c "${runcmd_str} ${HIOP_VALGRIND_CMD} $<TARGET_FILE:nlpMDS_EX1.exe> 400 100 0 -selfcheck"
    )
endif()

if(HIOP_BUILD_SHARED AND NOT HIOP_USE_GPU)
  add_test(NAME NlpMixedDenseSparseCinterface COMMAND ${RUNCMD} "$<TARGET_FILE:nlpMDS_cEx1.exe>")
endif()
