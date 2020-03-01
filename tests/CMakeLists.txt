

add_executable(testVector testVector.cpp vectorTestsPar.cpp)
target_link_libraries(testVector hiop ${HIOP_MATH_LIBRARIES})
