add_executable(nlpSparse_EX1.exe nlpSparse_EX1.cpp nlpSparse_EX1_driver.cpp)
target_link_libraries(nlpSparse_EX1.exe HiOp::HiOp)

add_executable(nlpSparse_EX2.exe nlpSparse_EX2.cpp nlpSparse_EX2_driver.cpp)
target_link_libraries(nlpSparse_EX2.exe HiOp::HiOp)

add_executable(nlpSparse_EX3.exe nlpSparse_EX3.cpp nlpSparse_EX3_driver.cpp)
target_link_libraries(nlpSparse_EX3.exe HiOp::HiOp)

##########################################################
# CMake Tests
##########################################################
add_test(NAME NlpSparse1_1 COMMAND ${RUNCMD} "$<TARGET_FILE:nlpSparse_EX1.exe>" "500" "-selfcheck")
add_test(NAME NlpSparse1_2 COMMAND ${RUNCMD} "$<TARGET_FILE:nlpSparse_EX1.exe>" "500" "-fr" "-selfcheck")
if(HIOP_USE_PARDISO)
  add_test(NAME NlpSparse1_3 COMMAND ${RUNCMD} "$<TARGET_FILE:nlpSparse_EX1.exe>" "500" "-pardiso" "-selfcheck")
endif(HIOP_USE_PARDISO)
add_test(NAME NlpSparse2_1 COMMAND ${RUNCMD} "$<TARGET_FILE:nlpSparse_EX2.exe>" "500" "-selfcheck")
add_test(NAME NlpSparse2_2 COMMAND ${RUNCMD} "$<TARGET_FILE:nlpSparse_EX2.exe>" "500" "-inertiafree" "-selfcheck")
add_test(NAME NlpSparse3_1 COMMAND ${RUNCMD} "$<TARGET_FILE:nlpSparse_EX3.exe>" "500" "-selfcheck")
