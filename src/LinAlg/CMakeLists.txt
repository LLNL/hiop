
# Set linear algebra common source files
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

# Add interfaces for sparse linear solvers when enabled
if(HIOP_SPARSE)
    if(HIOP_USE_COINHSL)
      set(hiopLinAlg_SRC ${hiopLinAlg_SRC} hiopLinSolverIndefSparseMA57.cpp)
    endif(HIOP_USE_COINHSL)      
    if(HIOP_USE_STRUMPACK)
      set(hiopLinAlg_SRC ${hiopLinAlg_SRC} hiopLinSolverSparseSTRUMPACK.cpp)
    endif(HIOP_USE_STRUMPACK)
endif()

# Add RAJA/Umpire sources when enabled
if(HIOP_USE_UMPIRE)
  set(hiopLinAlg_SRC ${hiopLinAlg_SRC}
    hiopVectorRajaPar.cpp
    hiopVectorIntRaja.cpp
    hiopMatrixRajaDense.cpp
    hiopMatrixRajaSparseTriplet.cpp)
endif()

# If GPU support is enabled add Magma interface (CUDA version)
# Treat RAJA sources as CUDA (temporary, need more flexible solutions)
if(HIOP_USE_GPU)
  set(hiopLinAlg_SRC ${hiopLinAlg_SRC} hiopLinSolverIndefDenseMagma.cpp)
  set_source_files_properties(hiopVectorRajaPar.cpp PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(hiopVectorIntRaja.cpp PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(hiopMatrixRajaDense.cpp PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(hiopMatrixRajaSparseTriplet.cpp PROPERTIES LANGUAGE CUDA)
endif()

# Add interface to UMFPACK when Kron reduction is enabled
if(HIOP_WITH_KRON_REDUCTION)
  set(hiopLinAlg_SRC ${hiopLinAlg_SRC} hiopLinSolverUMFPACKZ.cpp)
endif()

# Build and link linear algebra target
add_library(hiopLinAlg OBJECT ${hiopLinAlg_SRC})
target_link_libraries(hiopLinAlg PUBLIC hiop_math)

# Link linear algebra target to Umpire & RAJA when enabled
if(HIOP_USE_UMPIRE)
  target_link_libraries(hiopLinAlg PUBLIC umpire RAJA OpenMP::OpenMP_CXX)
endif()

# Build Kron reduction app
if(HIOP_WITH_KRON_REDUCTION)
  add_executable(test_hiopLinAlgComplex.exe test_hiopLinalgComplex.cpp)
  target_link_libraries(test_hiopLinAlgComplex.exe PRIVATE hiop)
endif(HIOP_WITH_KRON_REDUCTION)
