# Build ReSolve library

set(ReSolve_SRC
  #hiopLinSolverSparseCUSOLVER.cpp
  RefactorizationSolver.cpp
  MatrixCsr.cpp
  IterativeRefinement.cpp
  KrylovSolverKernels.cu
)
set(ReSolve_HEADERS
  RefactorizationSolver.hpp
  MatrixCsr.hpp
  IterativeRefinement.hpp
)

set_source_files_properties(${ReSolve_SRC} PROPERTIES LANGUAGE CUDA)

add_library(ReSolve STATIC ${ReSolve_SRC})
target_include_directories(ReSolve INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_link_libraries(ReSolve PRIVATE KLU hiop_cuda)
