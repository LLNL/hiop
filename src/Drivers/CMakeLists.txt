#add_executable(linalgTest.exe linalgTest.cpp)
#target_link_libraries(linalgTest.exe hiop ${HIOP_MATH_LIBRARIES})

add_executable(nlpDenseCons_ex1.exe nlpDenseCons_ex1.cpp nlpDenseCons_ex1_driver.cpp)
target_link_libraries(nlpDenseCons_ex1.exe hiop_math hiopLinAlg hiopOptimization hiopUtils)

add_executable(nlpDenseCons_ex2.exe nlpDenseCons_ex2.cpp nlpDenseCons_ex2_driver.cpp)
target_link_libraries(nlpDenseCons_ex2.exe hiop_math hiopLinAlg hiopOptimization hiopUtils)

add_executable(nlpDenseCons_ex3.exe nlpDenseCons_ex3_driver.cpp)
target_link_libraries(nlpDenseCons_ex3.exe hiop_math hiopLinAlg hiopOptimization hiopUtils)

add_executable(nlpMDS_ex4.exe nlpMDS_ex4_driver.cpp)
target_link_libraries(nlpMDS_ex4.exe hiop_math hiopLinAlg hiopOptimization hiopUtils)

if(HIOP_BUILD_SHARED)
  add_executable(nlpMDS_cex4.exe nlpMDS_ex4.c)
  target_link_libraries(nlpMDS_cex4.exe hiop_shared ${HIOP_MATH_LIBRARIES})
endif()

add_executable(nlpMDS_ex5.exe nlpMDS_ex5_driver.cpp)
target_link_libraries(nlpMDS_ex5.exe hiop ${HIOP_MATH_LIBRARIES})

#add_executable(hpc_benchmark.exe hpc_benchmark.cpp)
#target_link_libraries(hpc_benchmark.exe ${HIOP_MATH_LIBRARIES})
