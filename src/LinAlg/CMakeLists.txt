set(hiopLinAlg_SRC
  hiopVectorPar.cpp
  hiopVectorIntSeq.cpp
  hiopMatrixDenseRowMajor.cpp
  hiopLinSolver.cpp
  hiopLinAlgFactory.cpp
  hiopMatrixComplexDense.cpp
  hiopMatrixSparseTripletStorage.cpp
  hiopMatrixSparseTriplet.cpp
  hiopMatrixComplexSparseTriplet.cpp
)

if(HIOP_SPARSE)
    set(hiopLinAlg_SRC ${hiopLinAlg_SRC} hiopLinSolverIndefSparseMA57.cpp)
endif()

if(HIOP_USE_GPU)
  set(hiopLinAlg_SRC ${hiopLinAlg_SRC} hiopLinSolverIndefDenseMagma.cpp)
  foreach(f hiopVectorRajaPar.cpp 
            hiopVectorIntRaja.cpp 
            hiopMatrixRajaDense.cpp 
            hiopMatrixRajaSparseTriplet.cpp)
    set_source_files_properties(${f} PROPERTIES LANGUAGE CUDA)
  endforeach()
endif()

if(HIOP_USE_UMPIRE)
  set(hiopLinAlg_SRC ${hiopLinAlg_SRC}
    hiopVectorRajaPar.cpp 
    hiopVectorIntRaja.cpp
    hiopMatrixRajaDense.cpp 
    hiopMatrixRajaSparseTriplet.cpp)
endif()

if(HIOP_WITH_KRON_REDUCTION)
  set(hiopLinAlg_SRC ${hiopLinAlg_SRC} hiopLinSolverUMFPACKZ.cpp)
endif()

add_library(hiopLinAlg OBJECT ${hiopLinAlg_SRC})
target_link_libraries(hiopLinAlg PUBLIC hiop_math)

if(HIOP_USE_UMPIRE)
  target_link_libraries(hiopLinAlg PUBLIC umpire RAJA OpenMP::OpenMP_CXX)
endif()

if(HIOP_WITH_KRON_REDUCTION)
  add_executable(test_hiopLinAlgComplex.exe test_hiopLinalgComplex.cpp)
  target_link_libraries(test_hiopLinAlgComplex.exe PRIVATE hiop)
endif(HIOP_WITH_KRON_REDUCTION)
