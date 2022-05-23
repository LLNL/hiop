
# Set headers to be installed as part of the hiop interface
set(hiopLinAlg_INTERFACE_HEADERS
  hiop_blasdefs.hpp
  hiopLinAlgFactory.hpp
  hiopLinSolver.hpp
  hiopLinSolverSymDenseLapack.hpp
  hiopLinSolverSymDenseMagma.hpp
  hiopLinSolverSymSparseMA57.hpp
  hiopLinSolverMA86Z.hpp
  hiopLinSolverSparseGinkgo.hpp
  hiopLinSolverSparseSTRUMPACK.hpp
  hiopLinSolverSparsePARDISO.hpp
  hiopLinSolverSparseCUSOLVER.hpp
  hiopLinSolverUMFPACKZ.hpp
  hiopLinSolverCholCuSparse.hpp
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
  hiopMatrixSparseCSR.hpp
  hiopMatrixSparseCSRSeq.hpp
  hiopMatrixSparseCsrCuda.hpp
  hiopVector.hpp
  hiopVectorInt.hpp
  hiopVectorIntRaja.hpp
  hiopVectorIntSeq.hpp
  hiopVectorPar.hpp
  hiopVectorRajaPar.hpp
  hiopLinearOperator.hpp
  hiopKrylovSolver.hpp
  )

# Set linear algebra common source files
set(hiopLinAlg_SRC
  hiopVectorPar.cpp
  hiopVectorIntSeq.cpp
  hiopMatrixDenseRowMajor.cpp
  hiopLinSolver.cpp
  hiopLinAlgFactory.cpp
  hiopMatrixMDS.cpp
  hiopMatrixComplexDense.cpp
  hiopMatrixSparseTripletStorage.cpp
  hiopMatrixSparseTriplet.cpp
  hiopMatrixComplexSparseTriplet.cpp
  hiopMatrixSparseCSRSeq.cpp
  hiopLinearOperator.cpp
  hiopKrylovSolver.cpp
)

set(hiopLinAlg_RAJA_SRC
  hiopVectorRajaPar.cpp
  hiopVectorIntRaja.cpp
  hiopMatrixRajaDense.cpp
  hiopMatrixRajaSparseTriplet.cpp
)

set(hiopLinAlg_GPU_SRC
  hiopLinSolverSymDenseMagma.cpp
)

set(hiopLinAlg_MA57_SRC
  hiopLinSolverSymSparseMA57.cpp
)

set(hiopLinAlg_STRUMPACK_SRC
  hiopLinSolverSparseSTRUMPACK.cpp
)

set(hiopLinAlg_PARDISO_SRC
  hiopLinSolverSparsePARDISO.cpp
)

set(hiopLinAlg_CUSOLVER_LU_SRC
  hiopLinSolverSparseCUSOLVER.cpp
)

set(hiopLinAlg_CUSOLVER_CHOL_SRC
  hiopLinSolverCholCuSparse.cpp
)

set(hiopLinAlg_CUDASPARSEKERNELS_SRC
  MatrixSparseCsrCudaKernels.cu
)

set(hiopLinAlg_CUDASPARSE_SRC
  hiopMatrixSparseCsrCuda.cpp
)

set(hiopLinAlg_KRON_REDUCTION_SRC
  hiopLinSolverUMFPACKZ.cpp
)

set(hiopLinAlg_Ginkgo_SRC
  hiopLinSolverSparseGinkgo.cpp
)

# Add interfaces for sparse linear solvers when enabled
if(HIOP_SPARSE)
    if(HIOP_USE_COINHSL)
      list(APPEND hiopLinAlg_SRC ${hiopLinAlg_MA57_SRC})
    endif(HIOP_USE_COINHSL)      
    if(HIOP_USE_STRUMPACK)
      list(APPEND hiopLinAlg_SRC ${hiopLinAlg_STRUMPACK_SRC})
    endif(HIOP_USE_STRUMPACK)
    if(HIOP_USE_PARDISO)
      list(APPEND hiopLinAlg_SRC ${hiopLinAlg_PARDISO_SRC})
    endif(HIOP_USE_PARDISO)
    if(HIOP_USE_CUSOLVER_LU)
      list(APPEND hiopLinAlg_SRC ${hiopLinAlg_CUSOLVER_LU_SRC})
      set_source_files_properties(${hiopLinAlg_CUSOLVER_LU_SRC} PROPERTIES LANGUAGE CUDA)
    endif(HIOP_USE_CUSOLVER_LU)
    if(HIOP_USE_CUDA)
      list(APPEND hiopLinAlg_SRC ${hiopLinAlg_CUSOLVER_CHOL_SRC})
      set_source_files_properties(${hiopLinAlg_CUSOLVER_CHOL_SRC} PROPERTIES LANGUAGE CUDA)
      list(APPEND hiopLinAlg_SRC ${hiopLinAlg_CUDASPARSE_SRC})
      set_source_files_properties(${hiopLinAlg_CUDASPARSEKERNELS_SRC} PROPERTIES LANGUAGE CUDA)
      list(APPEND hiopLinAlg_SRC ${hiopLinAlg_CUDASPARSEKERNELS_SRC})
    endif(HIOP_USE_CUDA)
    
    if(HIOP_USE_GINKGO)
      list(APPEND hiopLinAlg_SRC ${hiopLinAlg_Ginkgo_SRC})
  endif(HIOP_USE_GINKGO)
endif()

# Add RAJA/Umpire sources when enabled
if(HIOP_USE_RAJA)
  list(APPEND hiopLinAlg_INTERFACE_HEADERS hiop_raja_defs.hpp)
  list(APPEND hiopLinAlg_SRC ${hiopLinAlg_RAJA_SRC})
endif()

# If GPU support is enabled add Magma interface (CUDA version)
# Treat RAJA sources as CUDA (temporary, need more flexible solutions)
if(HIOP_USE_GPU)
  list(APPEND hiopLinAlg_SRC ${hiopLinAlg_GPU_SRC})
  if(HIOP_USE_CUDA)
    set_source_files_properties(${hiopLinAlg_RAJA_SRC} ${hiopLinAlg_CUSOLVER_SRC} PROPERTIES LANGUAGE CUDA)
  endif()
endif()

# Add interface to UMFPACK when Kron reduction is enabled
if(HIOP_WITH_KRON_REDUCTION)
  list(APPEND hiopLinAlg_SRC ${hiopLinAlg_KRON_REDUCTION_SRC})
endif()

# Build Kron reduction app
if(HIOP_WITH_KRON_REDUCTION)
  add_executable(test_hiopLinAlgComplex.exe test_hiopLinalgComplex.cpp)
  target_link_libraries(test_hiopLinAlgComplex.exe PRIVATE HiOp::HiOp)
endif(HIOP_WITH_KRON_REDUCTION)

install(
  FILES ${hiopLinAlg_INTERFACE_HEADERS}
  DESTINATION include
  )

add_library(hiopLinAlg OBJECT ${hiopLinAlg_SRC})
target_link_libraries(hiopLinAlg PRIVATE hiop_tpl)
