# Set sources for vector tests
set(testVector_SRC testVector.cpp LinAlg/vectorTestsPar.cpp)

# Check if using RAJA and Umpire and add RAJA sources
if(HIOP_USE_UMPIRE)
  set(testVector_SRC ${testVector_SRC} LinAlg/vectorTestsRajaPar.cpp)
  include_directories(${UMPIRE_INCLUDE_DIRS} ${camp_INSTALL_PREFIX}/include)
endif()

if(HIOP_USE_GPU)
  set_source_files_properties(testVector.cpp LinAlg/vectorTestsRajaPar.cpp PROPERTIES LANGUAGE CUDA)
endif()

# Build vector test
add_executable(testVector ${testVector_SRC})
target_link_libraries(testVector PRIVATE hiop)

if(HIOP_USE_UMPIRE)
  target_link_libraries(testVector PRIVATE umpire RAJA OpenMP::OpenMP_CXX)
endif()

# Build matrix test
add_executable(testMatrix testMatrix.cpp LinAlg/matrixTestsDense.cpp)
target_link_libraries(testMatrix PRIVATE hiop)

# Build sparse matrix test
add_executable(testMatrixSparse testMatrixSparse.cpp LinAlg/matrixTestsSparseTriplet.cpp)
target_link_libraries(testMatrixSparse PRIVATE hiop)
