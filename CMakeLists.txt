cmake_minimum_required (VERSION 3.12)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if (POLICY CMP0074)
cmake_policy(SET CMP0074 NEW)
endif ()

project (hiop)
# enable Fortran for Fortran name mangling
enable_language(Fortran)

# Create header for Fortran name mangling
include(FortranCInterface)
FortranCInterface_HEADER(FortranCInterface.hpp MACRO_NAMESPACE "FC_")

option(HIOP_USE_MPI "Build with MPI support" ON)
option(HIOP_USE_GPU "Build with support for GPUs - Magma and cuda libraries" OFF)
option(HIOP_TEST_WITH_BSUB "Use `jsrun` instead of `mpirun` commands when running tests" OFF)
option(HIOP_USE_UMPIRE "Build with portable memory management library Umpire" ON)
option(HIOP_USE_RAJA   "Build with portability abstraction library RAJA" ON)
option(HIOP_DEEPCHECKS "Extra checks and asserts in the code with a high penalty on performance" OFF)
option(HIOP_WITH_KRON_REDUCTION "Build Kron Reduction code (requires UMFPACK)" OFF)
option(HIOP_DEVELOPER_MODE "Build with extended warnings and options" OFF)
#with testing drivers capable of 'selfchecking' (-selfcheck)
option(HIOP_WITH_MAKETEST "Enable 'make test'" ON)
option(HIOP_BUILD_SHARED "Build shared library" OFF)
option(HIOP_BUILD_STATIC "Build static library" ON)
option(HIOP_SPARSE "Build with sparse linear algebra" ON)

if(HIOP_BUILD_SHARED)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

if(HIOP_USE_GPU)
option(HIOP_USE_MAGMA "Use Magma linear algebra" ON)
endif()

add_library(hiop_math INTERFACE)
add_library(hiop_options INTERFACE)
add_library(hiop_warnings INTERFACE)

if(HIOP_USE_MPI)
find_package(MPI REQUIRED)
if(NOT DEFINED MPI_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
  set(CMAKE_C_COMPILER ${MPI_C_COMPILER})
endif(NOT DEFINED MPI_CXX_COMPILER)
include_directories(${MPI_CXX_ADDITIONAL_INCLUDE_DIRS} ${MPI_CXX_COMPILER_INCLUDE_DIRS})
target_link_libraries(hiop_math INTERFACE MPI::MPI_CXX)
endif(HIOP_USE_MPI)

# The binary dir is already a global include directory
configure_file(
"${CMAKE_SOURCE_DIR}/src/Interface/hiop_defs.hpp.in"
"${CMAKE_BINARY_DIR}/hiop_defs.hpp")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

target_compile_features(hiop_options
INTERFACE
cxx_std_11
cxx_alignas
cxx_alignof
cxx_attributes
cxx_auto_type
cxx_constexpr
cxx_defaulted_functions
cxx_deleted_functions
cxx_final
cxx_lambdas
cxx_noexcept
cxx_override
cxx_range_for
cxx_rvalue_references
cxx_static_assert
cxx_strong_enums
cxx_trailing_return_types
cxx_unicode_literals
cxx_user_literals
cxx_variadic_macros
)

target_compile_options(hiop_warnings
INTERFACE
-Wall
-Wextra
-Wshadow            # warn the user if a variable
                    # declaration shadows one from a
                    # parent context
-Wnull-dereference
-Wdouble-promotion  # Warn on implicit conversion from
                    # float to double
)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
message(STATUS "Found extra GNU compiler warnings")
target_compile_options(hiop_warnings
  INTERFACE
  -Wmisleading-indentation    # Warn on indented blocks
                              # that are not really blocks
  -Wduplicated-cond           # Warn on if/else chain with
                              # duplicated conditions
  -Wduplicated-branches       # Warn on if/else chains with
                              # duplicated code
  -Wuseless-cast              # Warn when casting to the same type
  )
endif()

if(HIOP_DEVELOPER_MODE)
target_link_libraries(hiop_math INTERFACE hiop_options hiop_warnings)
endif()

if(HIOP_USE_GPU)
include(CheckLanguage)
enable_language(CUDA)
check_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(HIOP_NVCC_ARCH)
  set(CMAKE_CUDA_FLAGS "-arch=${HIOP_NVCC_ARCH}")
  message(STATUS "Using CUDA arch ${HIOP_NVCC_ARCH}")
else()
  set(CMAKE_CUDA_FLAGS "-arch=sm_35")
  message(STATUS "Using CUDA arch sm_35")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

set(HIOP_MAGMA_DIR CACHE PATH "Path to Magma directory")

include(FindHiopCudaLibraries)
target_link_libraries(hiop_math INTERFACE hiop_cuda)
endif(HIOP_USE_GPU)

if(HIOP_USE_RAJA)
# Look for CMake configuration file in RAJA installation
find_package(RAJA CONFIG
  PATHS ${RAJA_DIR} ${RAJA_DIR}/share/raja/cmake
  REQUIRED)
endif()

if(HIOP_USE_UMPIRE)
# Look for CMake configuration file in Umpire installation
find_package(umpire CONFIG
  PATHS ${umpire_DIR} ${umpire_DIR}/share/umpire/cmake
  REQUIRED)
endif()

if(HIOP_WITH_KRON_REDUCTION)
set(HIOP_UMFPACK_DIR CACHE PATH "Path to UMFPACK directory")
include(FindUMFPACK)
target_link_libraries(hiop_math INTERFACE UMFPACK)

# metis needed (5.x needed by umfpack)
set(HIOP_METIS_DIR CACHE PATH "Path to METIS directory")
include(FindMETIS)
target_link_libraries(hiop_math INTERFACE METIS)
endif(HIOP_WITH_KRON_REDUCTION)

find_package(OpenMP)
target_link_libraries(hiop_math INTERFACE OpenMP::OpenMP_CXX)

if(NOT DEFINED BLAS_LIBRARIES)
find_package(BLAS REQUIRED)
target_link_libraries(hiop_math INTERFACE ${BLAS_LIBRARIES})
message(STATUS "Found BLAS libraries: ${BLAS_LIBRARIES}")
endif(NOT DEFINED BLAS_LIBRARIES)


set(LAPACK_LIBRARIES "-lgfortran;/usr/lib64/liblapack.so;-lpthread;-lm;-ldl")

if(NOT DEFINED LAPACK_LIBRARIES)
# in case the toolchain defines them
find_package(LAPACK REQUIRED)
endif(NOT DEFINED LAPACK_LIBRARIES)
target_link_libraries(hiop_math INTERFACE ${LAPACK_LIBRARIES})
message(STATUS "Using LAPACK libraries: ${LAPACK_LIBRARIES}")


if(HIOP_USE_UMPIRE)
  target_link_libraries(hiop_math INTERFACE umpire RAJA)
endif()

#
# extended precision lapack based on xblas testing example
#
#set(LAPACK_LIBRARIES "-lgfortran;/export/home/petra1/work/installs/xblas-1.0.248/libxblas.a;/home/petra1/work/installs/lapack-3.7.0/libblas.a;/home/petra1/work/installs/lapack-3.7.0/liblapack.a;/home/petra1/work/installs/lapack-3.7.0/liblapack.a;/home/petra1/work/installs/lapack-3.7.0/libblas.a;/export/home/petra1/work/installs/xblas-1.0.248/libxblas.a")


# metis needed (5.x needed by umfpack)
if(NOT HIOP_METIS4_DIR)
  set(HIOP_METIS4_DIR $ENV{HIOP_METIS4_DIR} )
endif(NOT HIOP_METIS4_DIR)

find_library(METIS4_LIBRARY NAME libmetis.a HINTS ${HIOP_METIS4_DIR}/lib PATHS ${HIOP_METIS4_DIR}/lib)
find_path(HIOP_METIS4_INCLUDE_DIR NAME metis.h HINTS ${HIOP_METIS4_DIR}/include)
if(HIOP_METIS4_INCLUDE_DIR AND METIS4_LIBRARY)
  include_directories(${HIOP_METIS4_INCLUDE_DIR})
  message(STATUS "Found METIS_4 libraries: ${METIS4_LIBRARY}")
endif(HIOP_METIS4_INCLUDE_DIR AND METIS4_LIBRARY)

if( NOT HIOP_MA57_DIR)
#  set(HIOP_MA57_DIR CACHE PATH "Path to MA57 directory")
  set(HIOP_MA57_DIR $ENV{HIOP_MA57_DIR} )
endif(NOT HIOP_MA57_DIR)

find_library(MA57_LIBRARY libma57.a PATH ${HIOP_MA57_DIR}/lib)
if(MA57_LIBRARY)
  message(STATUS "Found MA57 libraries: ${MA57_LIBRARY}")
endif(MA57_LIBRARY)

if(MA57_LIBRARY AND METIS4_LIBRARY)
  target_link_libraries(hiop_math INTERFACE ${MA57_LIBRARY} ${METIS4_LIBRARY} -lgfortran)
  set(HIOP_SPARSE ON CACHE BOOL "Build with sparse linear algebra" FORCE)
  message(STATUS "Build with sparse linear algebra" )
endif(MA57_LIBRARY AND METIS4_LIBRARY)







# include build directory for Fortran name mangling header
include_directories(${CMAKE_BINARY_DIR})

include_directories(src/Interface)
include_directories(src/Optimization)
include_directories(src/LinAlg)
include_directories(src/Utils)
#!!!this include needs to ALWAYS be the last!!!
include_directories(src/_Internals)

add_subdirectory(src/Interface)
add_subdirectory(src/Optimization)
add_subdirectory(src/LinAlg)
add_subdirectory(src/Drivers)
add_subdirectory(src/Utils)

##########################################################
#  INSTALLATION
##########################################################
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  # set a name for the build type to make the output of the 'make install' look nice
  set(CMAKE_BUILD_TYPE "default-build")
endif("${CMAKE_BUILD_TYPE}" STREQUAL "")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/_dist-${CMAKE_BUILD_TYPE}")
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

if(HIOP_BUILD_STATIC)
  add_library(hiop STATIC
    $<TARGET_OBJECTS:hiopOptimization>
    $<TARGET_OBJECTS:hiopLinAlg>
    $<TARGET_OBJECTS:hiopUtils>)
  if(HIOP_WITH_KRON_REDUCTION)
    target_sources(hiop PUBLIC $<TARGET_OBJECTS:hiopKronRed>)
  endif(HIOP_WITH_KRON_REDUCTION)
  target_link_libraries(hiop PUBLIC hiop_math)
  install(TARGETS hiop DESTINATION lib)
endif()

if(HIOP_BUILD_SHARED)
  add_library(hiop_shared SHARED $<TARGET_OBJECTS:hiopOptimization>
    $<TARGET_OBJECTS:hiopLinAlg>
    $<TARGET_OBJECTS:hiopUtils>
    $<TARGET_OBJECTS:chiopInterface>)
  set_target_properties(hiop_shared PROPERTIES OUTPUT_NAME hiop)
  if(HIOP_WITH_KRON_REDUCTION)
    target_sources(hiop_shared PUBLIC $<TARGET_OBJECTS:hiopKronRed>)
  endif(HIOP_WITH_KRON_REDUCTION)
  target_link_libraries(hiop_shared PUBLIC hiop_math)
  install(TARGETS hiop_shared DESTINATION lib)
endif()

add_subdirectory(tests)

install(FILES src/Interface/hiopInterface.hpp
  src/Interface/chiopInterface.hpp
  src/Interface/hiopInterface.h
  src/Optimization/hiopNlpFormulation.hpp
  src/Optimization/hiopNlpTransforms.hpp
  src/Optimization/hiopAlgFilterIPM.hpp
  src/Optimization/hiopIterate.hpp
  src/Optimization/hiopResidual.hpp
  src/Optimization/hiopKKTLinSys.hpp
  src/Optimization/hiopPDPerturbation.hpp
  src/Optimization/hiopLogBarProblem.hpp
  src/Optimization/hiopFilter.hpp
  src/Optimization/hiopHessianLowRank.hpp
  src/Optimization/hiopDualsUpdater.hpp
  src/LinAlg/hiopVector.hpp
  src/LinAlg/hiopVectorPar.hpp
  src/LinAlg/hiopVectorInt.hpp
  src/LinAlg/hiopVectorIntSeq.hpp
  src/LinAlg/hiopMatrix.hpp
  src/LinAlg/hiopMatrixDenseRowMajor.hpp
  src/LinAlg/hiopMatrixDense.hpp
  src/LinAlg/hiopMatrixMDS.hpp
  src/LinAlg/hiopMatrixSparse.hpp
  src/LinAlg/hiopMatrixSparseTriplet.hpp
  src/LinAlg/hiopMatrixSparseTripletStorage.hpp
  src/LinAlg/hiopMatrixMDS.hpp
  src/LinAlg/hiopMatrixComplexSparseTriplet.hpp
  src/LinAlg/hiopMatrixComplexDense.hpp
  src/LinAlg/hiopLinSolver.hpp
  src/LinAlg/hiopLinSolverIndefDenseLapack.hpp
  src/LinAlg/hiopLinSolverUMFPACKZ.hpp
  src/LinAlg/hiopLinSolverIndefSparseMA57.hpp
  src/LinAlg/hiopLinAlgFactory.hpp
  src/Utils/hiopRunStats.hpp
  src/Utils/hiopLogger.hpp
  src/Utils/hiopCSR_IO.hpp
  src/Utils/hiopTimer.hpp
  src/Utils/hiopOptions.hpp
  src/Utils/hiopKronReduction.hpp
  src/Utils/hiopMPI.hpp
  src/Utils/hiopCppStdUtils.hpp
  src/LinAlg/hiop_blasdefs.hpp
  src/Drivers/IpoptAdapter.hpp
  ${CMAKE_BINARY_DIR}/hiop_defs.hpp
  DESTINATION include)

if(HIOP_USE_RAJA)
  install(FILES
    src/LinAlg/hiopVectorIntRaja.hpp
    src/LinAlg/hiopVectorRajaPar.hpp
    src/LinAlg/hiopMatrixRajaDense.hpp
    src/LinAlg/hiopMatrixRajaSparseTriplet.hpp
    DESTINATION include)
endif()

set(STRIP_TABLE_CMD "awk '/Problem Summary/ { while (!match($0, /termination/)){ if(match($0, /^[ ]+[0-9]/)) { print $0; } getline } }'")
# If running in a BSUB allocaation, use jrun commands and make sure we are
# requesting the correct resources.
if(HIOP_TEST_WITH_BSUB)
  set(RUNCMD "jsrun -n 1")
  set(MPICMD "jsrun")
  if(HIOP_USE_GPU)
    set(MPICMD "${MPICMD} -g 1")
    set(RUNCMD "${RUNCMD} -g 1")
  endif()
else()
  set(RUNCMD "") # No special command is needed to run this program
  set(MPICMD "mpirun")
endif()

##########################################################
# CMake Tests
##########################################################
if (HIOP_WITH_MAKETEST)
  enable_testing()
  add_test(NAME VectorTest        COMMAND bash -c "${RUNCMD} $<TARGET_FILE:testVector>")
  if(HIOP_USE_MPI)
    add_test(NAME VectorTest_mpi COMMAND bash -c "${MPICMD} -n 2 $<TARGET_FILE:testVector>")
  endif(HIOP_USE_MPI)
  add_test(NAME MatrixTest        COMMAND bash -c "${RUNCMD} $<TARGET_FILE:testMatrix>")
  if(HIOP_USE_MPI)
    add_test(NAME MatrixTest_mpi COMMAND bash -c "${MPICMD} -n 2 $<TARGET_FILE:testMatrix>")
  endif(HIOP_USE_MPI)
  add_test(NAME SparseMatrixTest  COMMAND bash -c "${RUNCMD} $<TARGET_FILE:testMatrixSparse>")
  add_test(NAME SymmetricSparseMatrixTest COMMAND bash -c "${RUNCMD} $<TARGET_FILE:testMatrixSymSparse>")
  add_test(NAME NlpDenseCons1_5H  COMMAND bash -c "${RUNCMD} $<TARGET_FILE:nlpDenseCons_ex1.exe>   500 1.0 -selfcheck")
  add_test(NAME NlpDenseCons1_5K  COMMAND bash -c "${RUNCMD} $<TARGET_FILE:nlpDenseCons_ex1.exe>  5000 1.0 -selfcheck")
  add_test(NAME NlpDenseCons1_50K COMMAND bash -c "${RUNCMD} $<TARGET_FILE:nlpDenseCons_ex1.exe> 50000 1.0 -selfcheck")
  if(HIOP_USE_MPI)
    add_test(NAME NlpDenseCons1_50K_mpi COMMAND bash -c "${MPICMD} -n 2 $<TARGET_FILE:nlpDenseCons_ex1.exe> 50000 1.0 -selfcheck")
  endif(HIOP_USE_MPI)
  add_test(NAME NlpDenseCons2_5H COMMAND  bash -c "${RUNCMD} $<TARGET_FILE:nlpDenseCons_ex2.exe>   500 -selfcheck")
  add_test(NAME NlpDenseCons2_5K COMMAND  bash -c "${RUNCMD} $<TARGET_FILE:nlpDenseCons_ex2.exe>  5000 -selfcheck")
  add_test(NAME NlpDenseCons3_5H  COMMAND bash -c "${RUNCMD} $<TARGET_FILE:nlpDenseCons_ex3.exe>   500 -selfcheck")
  add_test(NAME NlpDenseCons3_5K  COMMAND bash -c "${RUNCMD} $<TARGET_FILE:nlpDenseCons_ex3.exe>  5000 -selfcheck")
  add_test(NAME NlpDenseCons3_50K COMMAND bash -c "${RUNCMD} $<TARGET_FILE:nlpDenseCons_ex3.exe> 50000 -selfcheck")
  if(HIOP_USE_MPI)
    add_test(NAME NlpDenseCons3_50K_mpi COMMAND bash -c "${MPICMD} -n 2 $<TARGET_FILE:nlpDenseCons_ex3.exe> 50000 -selfcheck")
  endif(HIOP_USE_MPI)

  add_test(NAME NlpMixedDenseSparse4_1 COMMAND bash -c "${RUNCMD} $<TARGET_FILE:nlpMDS_ex4.exe> 400 100 0 -selfcheck \
    | ${STRIP_TABLE_CMD} \
    | tee ${PROJECT_BINARY_DIR}/mds4_1.out")
  add_test(NAME NlpMixedDenseSparse4_2 COMMAND bash -c "${RUNCMD} $<TARGET_FILE:nlpMDS_ex4.exe> 400 100 1 -selfcheck \
    | ${STRIP_TABLE_CMD} \
    | tee ${PROJECT_BINARY_DIR}/mds4_2.out")
  if(HIOP_USE_RAJA)
    add_test(NAME NlpMixedDenseSparseRaja4_1 COMMAND bash -c "${RUNCMD} $<TARGET_FILE:nlpMDS_ex4_raja.exe> 400 100 0 -selfcheck \
      | ${STRIP_TABLE_CMD} \
      | tee ${PROJECT_BINARY_DIR}/mds4_raja_1.out")
    add_test(NAME NlpMixedDenseSparseRaja4_2 COMMAND bash -c "${RUNCMD} $<TARGET_FILE:nlpMDS_ex4_raja.exe> 400 100 1 -selfcheck \
      | ${STRIP_TABLE_CMD} \
      | tee ${PROJECT_BINARY_DIR}/mds4_raja_2.out")

    if(HIOP_DEEPCHECKS)
      foreach(iter 1 2)
        add_test(
          NAME "CompareExample4_NumIterations_${iter}" 
          COMMAND bash -c "\
          if [[ $(wc -l ${PROJECT_BINARY_DIR}/mds4_${iter}.out|cut -f1 -d' ') == $(wc -l ${PROJECT_BINARY_DIR}/mds4_raja_${iter}.out|cut -f1 -d' ') ]]
            then
              echo 'Output tables have the same number of iterations.'
              exit 0
            else
              echo 'Output tables have a different number of iterations!'
              exit 1
            fi")
        add_test(
          NAME "CompareExample4_ElementWise_${iter}"
          COMMAND bash -c "\
            join ${PROJECT_BINARY_DIR}/mds4_${iter}.out ${PROJECT_BINARY_DIR}/mds4_raja_${iter}.out \
            | ${PROJECT_SOURCE_DIR}/tests/testEx4CompareIterations.awk")
      endforeach()
    endif(HIOP_DEEPCHECKS)
  endif()

  add_test(NAME NlpMixedDenseSparse5_1 COMMAND bash -c "${RUNCMD} $<TARGET_FILE:nlpMDS_ex5.exe> 400 100 -selfcheck")

  if(HIOP_BUILD_SHARED AND NOT HIOP_USE_GPU)
    add_test(NAME NlpMixedDenseSparseCinterface COMMAND bash -c "${RUNCMD} $<TARGET_FILE:nlpMDS_cex4.exe>")
  endif()
endif(HIOP_WITH_MAKETEST)
