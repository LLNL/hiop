
# Build vector test
add_executable(testVector testVector.cpp LinAlg/vectorTestsPar.cpp)
target_link_libraries(testVector hiop ${HIOP_MATH_LIBRARIES})
set_target_properties(testVector PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    )
add_compile_options(-Wall -Wextra -pedantic -Werror)

# Build matrix test
add_executable(testMatrix testMatrix.cpp LinAlg/matrixTestsDense.cpp)
target_link_libraries(testMatrix hiop ${HIOP_MATH_LIBRARIES})
