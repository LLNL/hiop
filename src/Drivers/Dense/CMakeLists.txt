add_executable(nlpDenseCons_EX1.exe nlpDenseCons_EX1.cpp nlpDenseCons_EX1_driver.cpp)
target_link_libraries(nlpDenseCons_EX1.exe HiOp::HiOp)

add_executable(nlpDenseCons_EX2.exe nlpDenseCons_EX2.cpp nlpDenseCons_EX2_driver.cpp)
target_link_libraries(nlpDenseCons_EX2.exe HiOp::HiOp)

add_executable(nlpDenseCons_EX3.exe nlpDenseCons_EX3_driver.cpp)
target_link_libraries(nlpDenseCons_EX3.exe HiOp::HiOp)

##########################################################
# CMake Tests
##########################################################
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

if(HIOP_WITH_VALGRIND_TESTS)
  string(REPLACE ";" " " runcmd_str "${RUNCMD}")
  add_test(
    NAME NlpDenseCons1_5H_Valgrind
    COMMAND bash -c "${runcmd_str} ${HIOP_VALGRIND_CMD} $<TARGET_FILE:nlpDenseCons_EX1.exe> 500 1.0 -selfcheck"
    )
endif()
