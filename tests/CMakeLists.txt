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

if(HIOP_USE_GPU AND HIOP_USE_CUDA)
  set(cuda_sources testVector.cpp LinAlg/vectorTestsRajaPar.cpp
      testVector.cpp LinAlg/vectorTestsIntRaja.cpp
      testMatrixDense.cpp LinAlg/matrixTestsRajaDense.cpp
      testMatrixSparse.cpp LinAlg/matrixTestsRajaSparseTriplet.cpp
      testMatrixSymSparse.cpp LinAlg/matrixTestsRajaSymSparseTriplet.cpp)
  set_source_files_properties(${cuda_sources} PROPERTIES LANGUAGE CUDA)
endif()

# Build vector test
add_executable(testVector ${testVector_SRC})
target_link_libraries(testVector PRIVATE hiop)

# Build matrix test
add_executable(testMatrixDense ${testMatrix_SRC})
target_link_libraries(testMatrixDense PRIVATE hiop)

# Build sparse matrix test
add_executable(testMatrixSparse ${testMatrixSparse_SRC})
target_link_libraries(testMatrixSparse PRIVATE hiop)

# Build symmetric sparse matrix test
add_executable(testMatrixSymSparse ${testMatrixSymSparse_SRC})
target_link_libraries(testMatrixSymSparse PRIVATE hiop)

set(test_libraries hiop)
if(HIOP_USE_RAJA)
  list(APPEND test_libraries umpire RAJA)
  if(NOT HIOP_USE_HIP)
    list(APPEND test_libraries OpenMP::OpenMP_CXX)
  endif()
endif()

foreach(target testVector testMatrixDense testMatrixSparse testMatrixSymSparse)
  target_link_libraries(${target} PRIVATE ${test_libraries})
endforeach()
