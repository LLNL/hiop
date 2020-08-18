set(hiopLinAlg_SRC
  hiopVectorPar.cpp
  hiopMatrixDenseRowMajor.cpp
  hiopLinSolver.cpp
  hiopLinAlgFactory.cpp
  hiopMatrixComplexDense.cpp
  hiopMatrixSparseTripletStorage.cpp
  hiopMatrixSparseTriplet.cpp
  hiopMatrixComplexSparseTriplet.cpp
)

if(HIOP_USE_GPU)
  set(hiopLinAlg_SRC ${hiopLinAlg_SRC} hiopLinSolverIndefDenseMagma.cpp)
endif()

if(HIOP_WITH_KRON_REDUCTION)
  set(hiopLinAlg_SRC ${hiopLinAlg_SRC} hiopLinSolverUMFPACKZ.cpp)
endif()

add_library(hiopLinAlg OBJECT ${hiopLinAlg_SRC})
target_link_libraries(hiopLinAlg PUBLIC hiop_math)

if(HIOP_WITH_KRON_REDUCTION)
  add_executable(test_hiopLinAlgComplex.exe test_hiopLinalgComplex.cpp)
  target_link_libraries(test_hiopLinAlgComplex.exe PRIVATE hiop)
endif(HIOP_WITH_KRON_REDUCTION)
