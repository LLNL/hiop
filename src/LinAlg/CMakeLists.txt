add_library(hiopLinAlg OBJECT hiopVector.cpp hiopMatrix.cpp hiopMatrixComplexDense.cpp hiopMatrixSparseTripletStorage.cpp hiopMatrixSparseTriplet.cpp hiopMatrixComplexSparseTriplet.cpp)

if(UMFPACK_LIB)
  add_library(hiopLinAlgZ OBJECT hiopLinSolverUMFPACKZ.cpp)

  add_executable(test_hiopLinAlgComplex.exe test_hiopLinalgComplex.cpp)
  target_link_libraries(test_hiopLinAlgComplex.exe hiopLinAlg hiopLinAlgZ ${UMFPACK_LIB} ${METIS_LIB} ${HIOP_MATH_LIBRARIES})
endif(UMFPACK_LIB)

