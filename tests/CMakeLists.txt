
# Build vector test
add_executable(testVector testVector.cpp LinAlg/vectorTestsPar.cpp)
target_link_libraries(testVector PRIVATE hiop)

# Build matrix test
add_executable(testMatrix testMatrix.cpp LinAlg/matrixTestsDense.cpp)
target_link_libraries(testMatrix PRIVATE hiop)

# Build sparse matrix test
add_executable(testMatrixSparse testMatrixSparse.cpp LinAlg/matrixTestsSparseTriplet.cpp)
target_link_libraries(testMatrixSparse PRIVATE hiop)
