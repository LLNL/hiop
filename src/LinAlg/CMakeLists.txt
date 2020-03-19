add_library(hiopLinAlg OBJECT hiopVector.cpp hiopMatrix.cpp hiopMatrixSparseTripletStorage.cpp hiopMatrixSparseTriplet.cpp hiopMatrixComplexSparseTriplet.cpp)

target_link_libraries(hiopLinAlg
    PUBLIC
        hiop_warnings_1
        hiop_options)

add_executable(test_hiopLinAlgComplex.exe test_hiopLinalgComplex.cpp)
target_link_libraries(test_hiopLinAlgComplex.exe
    PRIVATE
        hiopLinAlg
        ${HIOP_MATH_LIBRARIES})

target_link_libraries(test_hiopLinAlgComplex.exe
    PRIVATE
        hiop_warnings_3
        hiop_options)
