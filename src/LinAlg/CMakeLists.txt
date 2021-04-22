
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
if(HIOP_USE_RAJA)
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

# Create legacy library
add_library(hiopLinAlg OBJECT ${hiopLinAlg_SRC})
target_link_libraries(hiopLinAlg PUBLIC hiop_math)
if(HIOP_USE_RAJA)
  target_link_libraries(hiopLinAlg PUBLIC umpire RAJA OpenMP::OpenMP_CXX)
endif()
# ---

set(hiopLinAlg_HEADERS
  hiopLinAlgFactory.hpp
  hiopLinSolver.hpp
  hiopLinSolverIndefDenseLapack.hpp
  hiopLinSolverIndefDenseMagma.hpp
  hiopLinSolverIndefSparseMA57.hpp
  hiopLinSolverMA86Z.hpp
  hiopLinSolverSparseSTRUMPACK.hpp
  hiopLinSolverUMFPACKZ.hpp
  hiopMatrix.hpp
  hiopMatrixComplexDense.hpp
  hiopMatrixComplexSparseTriplet.hpp
  hiopMatrixDense.hpp
  hiopMatrixDenseRowMajor.hpp
  hiopMatrixMDS.hpp
  hiopMatrixRajaDense.hpp
  hiopMatrixRajaSparseTriplet.hpp
  hiopMatrixSparse.hpp
  hiopMatrixSparseTriplet.hpp
  hiopMatrixSparseTripletStorage.hpp
  hiopVector.hpp
  hiopVectorInt.hpp
  hiopVectorIntRaja.hpp
  hiopVectorIntSeq.hpp
  hiopVectorPar.hpp
  hiopVectorRajaPar.hpp
  hiop_blasdefs.hpp
  )

hiop_add_component(
  SOURCES ${hiopLinAlg_SRC}
  INTERFACE_HEADERS ${hiopLinAlg_HEADERS}
  LINK_LIBRARIES PRIVATE umpire RAJA OpenMP::OpenMP_CXX
  )

# Build Kron reduction app
if(HIOP_WITH_KRON_REDUCTION)
  add_executable(test_hiopLinAlgComplex.exe test_hiopLinalgComplex.cpp)
  target_link_libraries(test_hiopLinAlgComplex.exe PRIVATE HiOp::HiOp)
endif(HIOP_WITH_KRON_REDUCTION)
