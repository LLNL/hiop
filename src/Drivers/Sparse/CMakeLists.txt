add_executable(NlpSparseEx1.exe NlpSparseEx1.cpp NlpSparseEx1Driver.cpp)
target_link_libraries(NlpSparseEx1.exe HiOp::HiOp)

add_executable(NlpSparseEx2.exe NlpSparseEx2.cpp NlpSparseEx2Driver.cpp)
target_link_libraries(NlpSparseEx2.exe HiOp::HiOp)

add_executable(NlpSparseEx3.exe NlpSparseEx3.cpp NlpSparseEx3Driver.cpp)
target_link_libraries(NlpSparseEx3.exe HiOp::HiOp)

##########################################################
# CMake Tests
##########################################################
add_test(NAME NlpSparse1_1 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx1.exe>" "500" "-selfcheck")
add_test(NAME NlpSparse1_2 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx1.exe>" "500" "-fr" "-selfcheck")
if(HIOP_USE_PARDISO)
  add_test(NAME NlpSparse1_3 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx1.exe>" "500" "-pardiso" "-selfcheck")
endif(HIOP_USE_PARDISO)
add_test(NAME NlpSparse2_1 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx2.exe>" "500" "-selfcheck")
add_test(NAME NlpSparse2_2 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx2.exe>" "500" "-inertiafree" "-selfcheck")
add_test(NAME NlpSparse3_1 COMMAND ${RUNCMD} "$<TARGET_FILE:NlpSparseEx3.exe>" "500" "-selfcheck")