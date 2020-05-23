
# Build vector test
add_executable(testVector testVector.cpp LinAlg/vectorTestsPar.cpp)
target_link_libraries(testVector PRIVATE hiop ${HIOP_MATH_LIBRARIES})

# Build matrix test
add_executable(testMatrix testMatrix.cpp LinAlg/matrixTestsDense.cpp)

target_link_libraries(testMatrix
    PRIVATE
        hiop_warnings
        hiop_options)

target_link_libraries(testMatrix PRIVATE hiop ${HIOP_MATH_LIBRARIES})
