# Build ReSolve library

set(ReSolve_SRC
  #hiopLinSolverSparseReSolve.cpp
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
# Eventually, we will:
# find_package(ReSolve REQUIRED)
# target_link_libraries(hiop_tpl INTERFACE ReSolve)
target_link_libraries(ReSolve PRIVATE KLU hiop_cuda)
