# Set sources for vector tests
set(testVector_SRC testVector.cpp LinAlg/vectorTestsPar.cpp)

# Set sources for dense matrix tests
set(testMatrix_SRC testMatrix.cpp LinAlg/matrixTestsDense.cpp)

# Check if using RAJA and Umpire and add RAJA sources
if(HIOP_USE_UMPIRE)
  set(testVector_SRC ${testVector_SRC} LinAlg/vectorTestsRajaPar.cpp)
  set(testMatrix_SRC ${testMatrix_SRC} LinAlg/matrixTestsRajaDense.cpp)
  include_directories(${UMPIRE_INCLUDE_DIRS} ${camp_INSTALL_PREFIX}/include)
endif()

if(HIOP_USE_GPU)
  set_source_files_properties(testVector.cpp LinAlg/vectorTestsRajaPar.cpp PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(testMatrix.cpp LinAlg/matrixTestsRajaDense.cpp PROPERTIES LANGUAGE CUDA)
endif()

# Build vector test
add_executable(testVector ${testVector_SRC})
target_link_libraries(testVector PRIVATE hiop)

# Build matrix test
add_executable(testMatrix ${testMatrix_SRC})
target_link_libraries(testMatrix PRIVATE hiop)

if(HIOP_USE_UMPIRE)
  target_link_libraries(testVector PRIVATE umpire RAJA OpenMP::OpenMP_CXX)
  target_link_libraries(testMatrix PRIVATE umpire RAJA OpenMP::OpenMP_CXX)
endif()

# Build sparse matrix test
add_executable(testMatrixSparse testMatrixSparse.cpp LinAlg/matrixTestsSparseTriplet.cpp)
target_link_libraries(testMatrixSparse PRIVATE hiop)
