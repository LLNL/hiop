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
  set_source_files_properties(hiopVectorRajaPar.cpp PROPERTIES LANGUAGE CUDA)
endif()

if(HIOP_USE_UMPIRE)
  set(LinAlg_SRC ${LinAlg_SRC} hiopVectorRajaPar.cpp)
  include_directories(${UMPIRE_INCLUDE_DIRS} ${camp_INSTALL_PREFIX}/include)
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
