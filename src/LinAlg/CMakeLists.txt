add_library(hiopLinAlg 
    OBJECT
        hiopVectorPar.cpp
        hiopMatrixDenseRowMajor.cpp
        hiopMatrixComplexDense.cpp
        hiopMatrixSparseTripletStorage.cpp
        hiopMatrixSparseTriplet.cpp
        hiopMatrixComplexSparseTriplet.cpp)
target_link_libraries(hiopLinAlg PUBLIC hiopOptimization hiop_math)

if(HIOP_WITH_KRON_REDUCTION)
  add_library(hiopLinAlgZ OBJECT hiopLinSolverUMFPACKZ.cpp)
  target_link_libraries(hiopLinAlgZ PUBLIC hiop_math)
  target_sources(hiopLinAlg PUBLIC $<TARGET_OBJECTS:hiopLinAlgZ>)

  add_executable(test_hiopLinAlgComplex.exe test_hiopLinalgComplex.cpp)
  target_link_libraries(test_hiopLinAlgComplex.exe PRIVATE hiopLinAlg hiop_math)
endif(HIOP_WITH_KRON_REDUCTION)
