add_library(hiopLinAlg OBJECT hiopVector.cpp hiopMatrix.cpp hiopMatrixComplexDense.cpp hiopMatrixSparseTripletStorage.cpp hiopMatrixSparseTriplet.cpp hiopMatrixComplexSparseTriplet.cpp)

if(MA86_LIB)
  add_library(hiopLinAlgZ OBJECT hiopLinSolverMA86Z.cpp)

  add_executable(test_hiopLinAlgComplex.exe test_hiopLinalgComplex.cpp)
  target_link_libraries(test_hiopLinAlgComplex.exe hiopLinAlg hiopLinAlgZ ${HIOP_MATH_LIBRARIES})
endif(MA86_LIB)

