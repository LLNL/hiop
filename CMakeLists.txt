cmake_minimum_required (VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project (hiop)
# enable Fortran for Fortran name mangling
enable_language(Fortran)

# Create header for Fortran name mangling
include(FortranCInterface)
FortranCInterface_HEADER(FortranCInterface.hpp MACRO_NAMESPACE "FC_")

# file(WRITE "src/Interface/hiop_defs.hpp" "")

option(HIOP_USE_MPI "Build with MPI support" ON)
option(HIOP_USE_GPU "Build with support for GPUs - Magma and cuda libraries" OFF)
option(HIOP_DEEPCHECKS "Extra checks and asserts in the code with a high penalty on performance" ON)
option(HIOP_WITH_KRON_REDUCTION "Build Kron Reduction code (requires MA86)" OFF)

#with testing drivers capable of 'selfchecking' (-selfcheck)
option(HIOP_WITH_MAKETEST "Enable 'make test'" ON)

if(HIOP_USE_MPI)
  if(NOT DEFINED MPI_CXX_COMPILER)
    find_package(MPI REQUIRED)
    set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
  endif(NOT DEFINED MPI_CXX_COMPILER)
endif(HIOP_USE_MPI)

# The binary dir is already a global include directory
configure_file(
  "${CMAKE_SOURCE_DIR}/src/Interface/hiop_defs.hpp.in"
  "${CMAKE_BINARY_DIR}/hiop_defs.hpp")

add_library(hiop_MAGMA INTERFACE)
add_library(hiop_MATH INTERFACE)
add_library(hiop_CUDA INTERFACE)
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
        -Wpedantic          # Warn on nonstandard C++
        -Wunused
        -Wconversion        # Warn on lossy conversions
        -Wnull-dereference
        -Wdouble-promotion  # Warn on implicit conversion from
                            # float to double
)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    message("Using extra GNU-specific compiler warnings...")
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

target_link_libraries(hiop_MATH INTERFACE hiop_options hiop_options)

if(HIOP_USE_GPU)
  include(CheckLanguage)
  enable_language(CUDA)
  check_language(CUDA)
  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()

  if(HIOP_NVCC_ARCH)
      set(CMAKE_CUDA_FLAGS "-arch=${HIOP_NVCC_ARCH} --expt-extended-lambda")
      message(STATUS "Using CUDA arch ${HIOP_NVCC_ARCH}")
  else()
      set(CMAKE_CUDA_FLAGS "-arch=sm_35 --expt-extended-lambda")
      message(STATUS "Using CUDA arch sm_35")
  endif()

  # for now we rely on MAGMA for GPUs computations
  target_link_libraries(hiop_MAGMA INTERFACE "-lmagma")
  if(HIOP_MAGMA_DIR)
    target_link_libraries(hiop_MAGMA INTERFACE "-L${HIOP_MAGMA_DIR}/lib")
    target_include_directories(hiop_MAGMA INTERFACE "${HIOP_MAGMA_DIR}/include")
  endif(HIOP_MAGMA_DIR)

  if(HIOP_CUDA_LIB_DIR)
    target_link_libraries(hiop_CUDA INTERFACE "-L${HIOP_CUDA_LIB_DIR}")
  else()
    target_link_libraries(hiop_CUDA INTERFACE "-L${CUDA_TOOLKIT_ROOT_DIR}")
  endif(HIOP_CUDA_LIB_DIR)

  # on some systems RHEL Nvidia toolkit puts cuBlas in different locations
  # user can setup/overwrite the lib directory
  if(HIOP_CUBLAS_LIB_DIR)
    target_link_libraries(hiop_CUDA INTERFACE "-L${HIOP_CUBLAS_LIB_DIR}")
  endif(HIOP_CUBLAS_LIB_DIR)

  if(HIOP_CUDA_INCLUDE_DIR)
    target_include_directories(hiop_CUDA INTERFACE "${HIOP_CUDA_INCLUDE_DIR}")
  else()
    target_include_directories(hiop_CUDA INTERFACE "${CUDA_INCLUDE_DIR}")
  endif(HIOP_CUDA_INCLUDE_DIR)

  target_link_libraries(hiop_MATH INTERFACE hiop_MAGMA hiop_CUDA)
  message("HiOp support for GPUs is on")
  get_target_property(cuda_includes hiop_CUDA INTERFACE_INCLUDE_DIRECTORIES)
  get_target_property(magma_includes hiop_MAGMA INTERFACE_INCLUDE_DIRECTORIES)
  message("CUDA include directories: ${cuda_includes} ${magma_includes}")
  get_target_property(cuda_libraries hiop_CUDA INTERFACE_LINK_LIBRARIES)
  get_target_property(magma_libraries hiop_MAGMA INTERFACE_LINK_LIBRARIES)
  message("CUDA linked libraries: ${cuda_libraries} ${magma_libraries}")
endif(HIOP_USE_GPU)

#set(MA86_DIR "/home/petra1/work/installs/hsl_ma86-1.6.0")
if(HIOP_WITH_KRON_REDUCTION)
  find_library(UMFPACK_LIB NAMES umfpack HINTS ${UMFPACK_DIR} "${UMFPACK_DIR}/lib" )
  if(UMFPACK_LIB)
    message("Using UMFPACK lib '${UMFPACK_LIB}'")

    find_path(UMFPACK_INCL NAMES "umfpack.h" HINTS ${UMFPACK_DIR} "${UMFPACK_DIR}/include")
    if(UMFPACK_INCL)
      include_directories(${UMFPACK_INCL})
      message("Using UMFPACK include '${UMFPACK_INCL}'")
    else() 
      message(FATAL_ERROR "Could not find UMFPACK header needed by HiOp's kron reduction. Please cmake with '-DUMFPACK_DIR=/path/to/umfpack_dir'; currently, UMFPACK_DIR='${UMFPACK_DIR}'")
    endif(UMFPACK_INCL)
  else()
    message(FATAL_ERROR "Could not find UMFPACK needed by HiOp's kron reduction. Please cmake with '-DUMFPACK_DIR=/path/to/umfpack_dir'; currently, UMFPACK_DIR='${UMFPACK_DIR}'")
  endif(UMFPACK_LIB)

  # metis needed (5.x needed by umfpack)
  find_library(METIS_LIB NAMES metis HINTS ${METIS_DIR} "${METIS_DIR}/lib" "${METIS_DIR}/src/")
  if(METIS_LIB)
    message("Using METIS lib '${METIS_LIB}'")
  else()
    message(FATAL_ERROR "Could not find METIS lib needed by UMFPACK. Please cmake with '-DMETIS_DIR=/path/to/metis/lib'; currently, METIS_DIR='${METIS_DIR}'")
  endif(METIS_LIB)


  # find_library(MA86_LIB NAMES hsl_ma86 ma86 HINTS ${MA86_DIR} "${MA86_DIR}/lib" )
  # if(MA86_LIB)
  #   message("Using HSL MA86 lib '${MA86_LIB}'")

  #   find_path(MA86_INCL NAMES "hsl_ma86z.h" HINTS ${MA86_DIR} "${MA86_DIR}/include")
  #   if(MA86_INCL)
  #     include_directories(${MA86_INCL})
  #     message("Using HSL MA86 include '${MA86_INCL}'")
  #   else() 
  #     message(FATAL_ERROR "Could not find HSL MA86 headers needed by kron reduction. Please cmake with '-DMA86_DIR=/path/to/ma86_dir'; currently, MA86_DIR='${MA86_DIR}'")
  #   endif(MA86_INCL)
  # else()
  #   message(FATAL_ERROR "Could not find HSL MA86 needed by kron reduction. Please cmake with '-DMA86_DIR=/path/to/ma86_dir'; currently, MA86_DIR='${MA86_DIR}'")
  # endif(MA86_LIB)

  # # metis needed (4.x works)
  # find_library(METIS_LIB NAMES metis HINTS ${METIS_DIR} "${METIS_DIR}/lib" "${METIS_DIR}/src/")
  # if(METIS_LIB)
  #   message("Using METIS lib '${METIS_LIB}'")
  # else()
  #   message(FATAL_ERROR "Could not find METIS lib needed by MA86. Please cmake with '-DMETIS_DIR=/path/to/metis/lib'; currently, METIS_DIR='${METIS_DIR}'")
  # endif(METIS_LIB)
endif(HIOP_WITH_KRON_REDUCTION)

find_package(OpenMP)
target_link_libraries(hiop_MATH INTERFACE OpenMP::OpenMP_CXX)

if(NOT DEFINED LAPACK_LIBRARIES)
  # in case the toolchain defines them
  find_package(LAPACK REQUIRED)
  target_link_libraries(hiop_MATH INTERFACE "${LAPACK_LIBRARIES}")
endif(NOT DEFINED LAPACK_LIBRARIES)

message("Using LAPACK libraries: ${LAPACK_LIBRARIES}")

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

add_subdirectory(src/Optimization)
add_subdirectory(src/LinAlg)
add_subdirectory(src/Drivers)
add_subdirectory(src/Utils)

##########################################################
#  INSTALATION
##########################################################
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
# set a name for the build type to make the output of the 'make install' look nice
  set(CMAKE_BUILD_TYPE "default-build")
endif("${CMAKE_BUILD_TYPE}" STREQUAL "")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/_dist-${CMAKE_BUILD_TYPE}")
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

add_library(hiop STATIC $<TARGET_OBJECTS:hiopOptimization>
                        $<TARGET_OBJECTS:hiopLinAlg>
			$<TARGET_OBJECTS:hiopUtils>)

target_link_libraries(hiop PUBLIC hiop_MATH)
if(HIOP_WITH_KRON_REDUCTION)
  target_link_libraries(hiop PUBLIC hiopKronRed hiopLinAlgZ)
endif(HIOP_WITH_KRON_REDUCTION)

install(TARGETS hiop DESTINATION lib)
install(FILES src/Interface/hiopInterface.hpp
	      src/Optimization/hiopNlpFormulation.hpp
	      src/Optimization/hiopNlpTransforms.hpp
	      src/Optimization/hiopAlgFilterIPM.hpp
	      src/Optimization/hiopIterate.hpp
	      src/Optimization/hiopResidual.hpp
              src/Optimization/hiopKKTLinSys.hpp
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
  add_test(NAME NlpMixedDenseSparse_1 COMMAND $<TARGET_FILE:nlpMDS_ex4.exe> 400 100 0 -selfcheck)
  add_test(NAME NlpMixedDenseSparse_2 COMMAND $<TARGET_FILE:nlpMDS_ex4.exe> 400 100 1 -selfcheck)
endif(HIOP_WITH_MAKETEST)
