cmake_minimum_required (VERSION 3.8)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

cmake_policy(SET CMP0074 NEW)

project (hiop)
# enable Fortran for Fortran name mangling
enable_language(Fortran)

# Create header for Fortran name mangling
include(FortranCInterface)
FortranCInterface_HEADER(FortranCInterface.hpp MACRO_NAMESPACE "FC_")

option(HIOP_USE_MPI "Build with MPI support" ON)
option(HIOP_USE_GPU "Build with support for GPUs - Magma and cuda libraries" OFF)
option(HIOP_DEEPCHECKS "Extra checks and asserts in the code with a high penalty on performance" ON)
option(HIOP_WITH_KRON_REDUCTION "Build Kron Reduction code (requires MA86)" OFF)
option(HIOP_DEVELOPER_MODE "Build with extended warnings and options" OFF)
#with testing drivers capable of 'selfchecking' (-selfcheck)
option(HIOP_WITH_MAKETEST "Enable 'make test'" ON)
option(HIOP_BUILD_SHARED "Build shared library" OFF)
option(HIOP_BUILD_STATIC "Build static library" ON)

set(HIOP_BUILD_SHARED CACHE BOOL "Build shared library")
set(HIOP_BUILD_STATIC CACHE BOOL "Build static library")

if(HIOP_BUILD_SHARED)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

if(HIOP_USE_GPU)
  set(HIOP_USE_MAGMA ON)
else()
  set(HIOP_USE_MAGMA OFF)
endif()

if(HIOP_USE_MPI)
  if(NOT DEFINED MPI_CXX_COMPILER)
    find_package(MPI REQUIRED)
    set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
    set(CMAKE_C_COMPILER ${MPI_C_COMPILER})
  endif(NOT DEFINED MPI_CXX_COMPILER)
endif(HIOP_USE_MPI)

# The binary dir is already a global include directory
configure_file(
  "${CMAKE_SOURCE_DIR}/src/Interface/hiop_defs.hpp.in"
  "${CMAKE_BINARY_DIR}/hiop_defs.hpp")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

add_library(hiop_math INTERFACE)
add_library(hiop_options INTERFACE)
add_library(hiop_warnings INTERFACE)

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

if(NOT DEFINED LAPACK_LIBRARIES)
  # in case the toolchain defines them
  find_package(LAPACK REQUIRED)
  target_link_libraries(hiop_math INTERFACE ${LAPACK_LIBRARIES})
  message(STATUS "Found LAPACK libraries: ${LAPACK_LIBRARIES}")
endif(NOT DEFINED LAPACK_LIBRARIES)

#
# extended precision lapack based on xblas testing example
#
#set(LAPACK_LIBRARIES "-lgfortran;/export/home/petra1/work/installs/xblas-1.0.248/libxblas.a;/home/petra1/work/installs/lapack-3.7.0/libblas.a;/home/petra1/work/installs/lapack-3.7.0/liblapack.a;/home/petra1/work/installs/lapack-3.7.0/liblapack.a;/home/petra1/work/installs/lapack-3.7.0/libblas.a;/export/home/petra1/work/installs/xblas-1.0.248/libxblas.a")

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
  add_library(hiop STATIC $<TARGET_OBJECTS:hiopOptimization>
    $<TARGET_OBJECTS:hiopLinAlg>
    $<TARGET_OBJECTS:hiopUtils>)
  target_link_libraries(hiop PUBLIC hiop_math)
  if(HIOP_WITH_KRON_REDUCTION)
    target_link_libraries(hiop PUBLIC hiopKronRed hiopLinAlgZ)
  endif(HIOP_WITH_KRON_REDUCTION)
  install(TARGETS hiop DESTINATION lib)
endif()

if(HIOP_BUILD_SHARED)
  add_library(hiop_shared SHARED $<TARGET_OBJECTS:hiopOptimization>
    $<TARGET_OBJECTS:hiopLinAlg>
    $<TARGET_OBJECTS:hiopUtils>
    $<TARGET_OBJECTS:chiopInterface>)
  set_target_properties(hiop_shared PROPERTIES OUTPUT_NAME hiop)
  target_link_libraries(hiop_shared PUBLIC hiop_math)
  if(HIOP_WITH_KRON_REDUCTION)
    target_link_libraries(hiop_shared PUBLIC hiopKronRed hiopLinAlgZ)
  endif(HIOP_WITH_KRON_REDUCTION)
  install(TARGETS hiop_shared DESTINATION lib)
endif()

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
  src/LinAlg/hiopMatrix.hpp
  src/LinAlg/hiopMatrixMDS.hpp
  src/LinAlg/hiopMatrixSparseTriplet.hpp
  src/LinAlg/hiopMatrixSparseTripletStorage.hpp
  src/LinAlg/hiopMatrixMDS.hpp
  src/LinAlg/hiopMatrixComplexSparseTriplet.hpp
  src/LinAlg/hiopMatrixComplexDense.hpp
  src/Utils/hiopRunStats.hpp
  src/Utils/hiopLogger.hpp
  src/Utils/hiopCSR_IO.hpp
  src/Utils/hiopTimer.hpp
  src/Utils/hiopOptions.hpp
  src/Utils/hiopKronReduction.hpp
  src/LinAlg/hiop_blasdefs.hpp
  src/Drivers/IpoptAdapter.hpp
  ${CMAKE_BINARY_DIR}/hiop_defs.hpp
  DESTINATION include)

##########################################################
# CMake Tests
##########################################################
if (HIOP_WITH_MAKETEST)
  enable_testing()
  add_test(NAME NlpDenseCons1_5H  COMMAND $<TARGET_FILE:nlpDenseCons_ex1.exe>   500 1.0 -selfcheck)
  add_test(NAME NlpDenseCons1_5K  COMMAND $<TARGET_FILE:nlpDenseCons_ex1.exe>  5000 1.0 -selfcheck)
  add_test(NAME NlpDenseCons1_50K COMMAND $<TARGET_FILE:nlpDenseCons_ex1.exe> 50000 1.0 -selfcheck)
  if(HIOP_USE_MPI)
    add_test(NAME NlpDenseCons1_50K_mpi COMMAND mpirun -np 2 $<TARGET_FILE:nlpDenseCons_ex1.exe> 50000 1.0 -selfcheck)
  endif(HIOP_USE_MPI)
  add_test(NAME NlpDenseCons2_5H COMMAND $<TARGET_FILE:nlpDenseCons_ex2.exe>   500 -selfcheck)
  add_test(NAME NlpDenseCons2_5K COMMAND $<TARGET_FILE:nlpDenseCons_ex2.exe>  5000 -selfcheck)
  add_test(NAME NlpDenseCons3_5H  COMMAND $<TARGET_FILE:nlpDenseCons_ex3.exe>   500 -selfcheck)
  add_test(NAME NlpDenseCons3_5K  COMMAND $<TARGET_FILE:nlpDenseCons_ex3.exe>  5000 -selfcheck)
  add_test(NAME NlpDenseCons3_50K COMMAND $<TARGET_FILE:nlpDenseCons_ex3.exe> 50000 -selfcheck)
  if(HIOP_USE_MPI)
    add_test(NAME NlpDenseCons3_50K_mpi COMMAND mpirun -np 2 $<TARGET_FILE:nlpDenseCons_ex3.exe> 50000 -selfcheck)
  endif(HIOP_USE_MPI)
  add_test(NAME NlpMixedDenseSparse4_1 COMMAND $<TARGET_FILE:nlpMDS_ex4.exe> 400 100 0 -selfcheck)
  add_test(NAME NlpMixedDenseSparse4_2 COMMAND $<TARGET_FILE:nlpMDS_ex4.exe> 400 100 1 -selfcheck)
  add_test(NAME NlpMixedDenseSparse5_1 COMMAND $<TARGET_FILE:nlpMDS_ex5.exe> 400 100 -selfcheck)
  if(HIOP_BUILD_SHARED)
    add_test(NAME NlpMixedDenseSparseCinterface COMMAND $<TARGET_FILE:nlpMDS_cex4.exe>)
  endif()
endif(HIOP_WITH_MAKETEST)
