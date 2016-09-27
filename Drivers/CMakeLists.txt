add_executable(linalgTest.exe linalgTest.cpp)
target_link_libraries(linalgTest.exe  hiopOptimization hiopLinAlg ${LAPACK_LIBRARIES})

add_executable(nlpDenseCons_ex1.exe nlpDenseCons_ex1.cpp)
target_link_libraries(nlpDenseCons_ex1.exe hiopOptimization hiopLinAlg ${LAPACK_LIBRARIES})

add_executable(nlpDenseCons_ex2.exe nlpDenseCons_ex2.cpp)
target_link_libraries(nlpDenseCons_ex2.exe hiopOptimization hiopLinAlg ${LAPACK_LIBRARIES})
