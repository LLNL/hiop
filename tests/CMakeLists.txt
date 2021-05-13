# Set sources for vector tests
set(testVector_SRC testVector.cpp LinAlg/vectorTestsPar.cpp LinAlg/vectorTestsIntSeq.cpp)

# Set sources for dense matrix tests
set(testMatrix_SRC testMatrixDense.cpp LinAlg/matrixTestsDenseRowMajor.cpp)

# Set sources for sparse matrix tests
set(testMatrixSparse_SRC testMatrixSparse.cpp LinAlg/matrixTestsSparseTriplet.cpp)

# Set sources for symmetric sparse matrix tests
set(testMatrixSymSparse_SRC testMatrixSymSparse.cpp LinAlg/matrixTestsSymSparseTriplet.cpp)

# Check if using RAJA and Umpire and add RAJA sources
if(HIOP_USE_RAJA)
  set(testVector_SRC ${testVector_SRC} LinAlg/vectorTestsRajaPar.cpp LinAlg/vectorTestsIntRaja.cpp)
  set(testMatrix_SRC ${testMatrix_SRC} LinAlg/matrixTestsRajaDense.cpp)
  set(testMatrixSparse_SRC ${testMatrixSparse_SRC} LinAlg/matrixTestsRajaSparseTriplet.cpp)
  set(testMatrixSymSparse_SRC ${testMatrixSymSparse_SRC} LinAlg/matrixTestsRajaSymSparseTriplet.cpp)
endif()

if(HIOP_USE_GPU)
  set_source_files_properties(testVector.cpp LinAlg/vectorTestsRajaPar.cpp PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(testVector.cpp LinAlg/vectorTestsIntRaja.cpp PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(testMatrixDense.cpp LinAlg/matrixTestsRajaDense.cpp PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(testMatrixSparse.cpp LinAlg/matrixTestsRajaSparseTriplet.cpp PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(testMatrixSymSparse.cpp LinAlg/matrixTestsRajaSymSparseTriplet.cpp PROPERTIES LANGUAGE CUDA)
endif()

# Build vector test
add_executable(testVector ${testVector_SRC})
target_link_libraries(testVector PRIVATE HiOp::HiOp)

# Build matrix test
add_executable(testMatrixDense ${testMatrix_SRC})
target_link_libraries(testMatrixDense PRIVATE HiOp::HiOp)

# Build sparse matrix test
add_executable(testMatrixSparse ${testMatrixSparse_SRC})
target_link_libraries(testMatrixSparse PRIVATE HiOp::HiOp)

# Build symmetric sparse matrix test
add_executable(testMatrixSymSparse ${testMatrixSymSparse_SRC})
target_link_libraries(testMatrixSymSparse PRIVATE HiOp::HiOp)

if(HIOP_USE_RAJA)
  target_link_libraries(testVector PRIVATE umpire RAJA OpenMP::OpenMP_CXX)
  target_link_libraries(testMatrixDense PRIVATE umpire RAJA OpenMP::OpenMP_CXX)
  target_link_libraries(testMatrixSparse PRIVATE umpire RAJA OpenMP::OpenMP_CXX)
  target_link_libraries(testMatrixSymSparse PRIVATE umpire RAJA OpenMP::OpenMP_CXX)
endif()
