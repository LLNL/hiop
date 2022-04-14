add_executable(NlpDenseConsEx1.exe NlpDenseConsEx1.cpp NlpDenseConsEx1Driver.cpp)
target_link_libraries(NlpDenseConsEx1.exe HiOp::HiOp)

add_executable(NlpDenseConsEx2.exe NlpDenseConsEx2.cpp NlpDenseConsEx2Driver.cpp)
target_link_libraries(NlpDenseConsEx2.exe HiOp::HiOp)

add_executable(NlpDenseConsEx3.exe NlpDenseConsEx3Driver.cpp)
target_link_libraries(NlpDenseConsEx3.exe HiOp::HiOp)

##########################################################
# CMake Tests
##########################################################
add_test(NAME NlpDenseCons1_5H  COMMAND ${RUNCMD} "$<TARGET_FILE:NlpDenseConsEx1.exe>"  "500" "1.0" "-selfcheck")
add_test(NAME NlpDenseCons1_5K  COMMAND ${RUNCMD} "$<TARGET_FILE:NlpDenseConsEx1.exe>" "5000" "1.0" "-selfcheck")
add_test(NAME NlpDenseCons1_50K COMMAND ${RUNCMD} "$<TARGET_FILE:NlpDenseConsEx1.exe>" "50000" "1.0" "-selfcheck")
if(HIOP_USE_MPI)
  add_test(NAME NlpDenseCons1_50K_mpi COMMAND ${MPICMD} -n 2 "$<TARGET_FILE:NlpDenseConsEx1.exe>" "50000" "1.0" "-selfcheck")
endif(HIOP_USE_MPI)
add_test(NAME NlpDenseCons2_5H    COMMAND  ${RUNCMD} "$<TARGET_FILE:NlpDenseConsEx2.exe>"   "500" "-selfcheck")
add_test(NAME NlpDenseCons2_5K    COMMAND  ${RUNCMD} "$<TARGET_FILE:NlpDenseConsEx2.exe>"  "5000" "-selfcheck")
add_test(NAME NlpDenseCons2_UN_5K COMMAND  ${RUNCMD} "$<TARGET_FILE:NlpDenseConsEx2.exe>"  "5000" "-unconstrained" "-selfcheck")
add_test(NAME NlpDenseCons3_5H    COMMAND  ${RUNCMD} "$<TARGET_FILE:NlpDenseConsEx3.exe>"   "500" "-selfcheck")
add_test(NAME NlpDenseCons3_5K    COMMAND  ${RUNCMD} "$<TARGET_FILE:NlpDenseConsEx3.exe>"  "5000" "-selfcheck")
add_test(NAME NlpDenseCons3_50K   COMMAND  ${RUNCMD} "$<TARGET_FILE:NlpDenseConsEx3.exe>" "50000" "-selfcheck")
if(HIOP_USE_MPI)
  add_test(NAME NlpDenseCons3_50K_mpi COMMAND ${MPICMD} -n 2 "$<TARGET_FILE:NlpDenseConsEx3.exe>" "50000" "-selfcheck")
endif(HIOP_USE_MPI)

if(HIOP_WITH_VALGRIND_TESTS)
  string(REPLACE ";" " " runcmd_str "${RUNCMD}")
  add_test(
    NAME NlpDenseCons1_5H_Valgrind
    COMMAND bash -c "${runcmd_str} ${HIOP_VALGRIND_CMD} $<TARGET_FILE:NlpDenseConsEx1.exe> 500 1.0 -selfcheck"
    )
endif()