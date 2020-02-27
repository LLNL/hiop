cmake_minimum_required (VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project (hiop)
# enable Fortran for Fortran name mangling
enable_language(Fortran)

# Create header for Fortran name mangling
include(FortranCInterface)
FortranCInterface_HEADER(FortranCInterface.hpp MACRO_NAMESPACE "FC_")

file(WRITE "src/Interface/hiop_defs.hpp" "")


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
  file(APPEND "src/Interface/hiop_defs.hpp" "#define HIOP_USE_MPI\n")
else()
  file(APPEND "src/Interface/hiop_defs.hpp" "#ifdef HIOP_USE_MPI\n#undef HIOP_USE_MPI\n#endif\n")
endif(HIOP_USE_MPI)

if(HIOP_USE_GPU)
  # for now we rely on MAGMA for GPUs computations
  set(MAGMA_LIBRARIES "-lmagma")
  if(HIOP_MAGMA_DIR)
    set(MAGMA_LIBRARIES "-L${HIOP_MAGMA_DIR}/lib ${MAGMA_LIBRARIES}")
    set(MAGMA_INCLUDE "${HIOP_MAGMA_DIR}/include")
  endif(HIOP_MAGMA_DIR)

  set(HIOP_CUDA_LIBRARIES "-lculibos -lcublas -lcublasLt -lnvblas -lcusparse -lcudart -lcudadevrt")
  find_package(CUDA)
  
  if(HIOP_CUDA_LIB_DIR)
    set(HIOP_CUDA_LIBRARIES "-L${HIOP_CUDA_LIB_DIR} ${HIOP_CUDA_LIBRARIES}")
  else()
    set(HIOP_CUDA_LIBRARIES "-L${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${HIOP_CUDA_LIBRARIES}")
  endif(HIOP_CUDA_LIB_DIR)

  # on some systems RHEL Nvidia toolkit puts cuBlas in different locations
  # user can setup/overwrite the lib directory
  if(HIOP_CUBLAS_LIB_DIR)
    set(HIOP_CUDA_LIBRARIES "-L${HIOP_CUBLAS_LIB_DIR} ${HIOP_CUDA_LIBRARIES}")
  endif(HIOP_CUBLAS_LIB_DIR)

  if(HIOP_CUDA_INCLUDE_DIR)
    set(HIOP_CUDA_INCLUDE "${HIOP_CUDA_INCLUDE_DIR}")
  else()
    set(HIOP_CUDA_INCLUDE "${CUDA_INCLUDE_DIRS}")
  endif(HIOP_CUDA_INCLUDE_DIR)

  message("HiOp support for GPUs is on")
  message("HiOp GPU libs: '${MAGMA_LIBRARIES} ${HIOP_CUDA_LIBRARIES}'") 
  message("HiOp GPU includes: '${MAGMA_INCLUDE} ${HIOP_CUDA_INCLUDE}'") 
  include_directories(${MAGMA_INCLUDE} ${HIOP_CUDA_INCLUDE})
  set(HIOP_MATH_LIBRARIES "${MAGMA_LIBRARIES} ${HIOP_CUDA_LIBRARIES} ${HIOP_MATH_LIBRARIES}")
endif(HIOP_USE_GPU)

if(HIOP_USE_GPU)
  file(APPEND "src/Interface/hiop_defs.hpp" "#define HIOP_USE_MAGMA\n")
else()
  file(APPEND "src/Interface/hiop_defs.hpp" "#ifdef HIOP_USE_MAGMA\n#undef HIOP_USE_MAGMA\n#endif\n")
endif(HIOP_USE_GPU)

if(HIOP_DEEPCHECKS)
  file(APPEND "src/Interface/hiop_defs.hpp" "#define HIOP_DEEPCHECKS\n")
else()
  file(APPEND "src/Interface/hiop_defs.hpp" "#ifdef HIOP_DEEPCHECKS\n#undef HIOP_DEEPCHECKS\n#endif\n")
endif(HIOP_DEEPCHECKS)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#set(MA86_DIR "/home/petra1/work/installs/hsl_ma86-1.6.0")
if(HIOP_WITH_KRON_REDUCTION)
  find_library(MA86_LIB NAMES hsl_ma86 ma86 HINTS ${MA86_DIR} "${MA86_DIR}/lib" )
  if(MA86_LIB)
    message("Using HSL MA86 lib '${MA86_LIB}'")

    find_path(MA86_INCL NAMES "hsl_ma86z.h" HINTS ${MA86_DIR} "${MA86_DIR}/include")
    if(MA86_INCL)
      include_directories(${MA86_INCL})
      message("Using HSL MA86 include '${MA86_INCL}'")
    else() 
      message(FATAL_ERROR "Could not find HSL MA86 headers needed by kron reduction. Please cmake with '-DMA86_DIR=/path/to/ma86_dir'; currently, MA86_DIR='${MA86_DIR}'")
    endif(MA86_INCL)
  else()
    message(FATAL_ERROR "Could not find HSL MA86 needed by kron reduction. Please cmake with '-DMA86_DIR=/path/to/ma86_dir'; currently, MA86_DIR='${MA86_DIR}'")
  endif(MA86_LIB)

  # metis needed (4.x works)
  find_library(METIS_LIB NAMES metis HINTS ${METIS_DIR} "${METIS_DIR}/lib" "${METIS_DIR}/src/")
  if(METIS_LIB)
    message("Using METIS lib '${MA86_LIB}'")
  else()
    message(FATAL_ERROR "Could not find METIS lib needed by MA86. Please cmake with '-DMETIS_DIR=/path/to/metis/lib'; currently, METIS_DIR='${METIS_DIR}'")
  endif(METIS_LIB)
endif(HIOP_WITH_KRON_REDUCTION)

find_package(OpenMP)

if(NOT DEFINED LAPACK_LIBRARIES)
  # in case the toolchain defines them
  find_package(LAPACK REQUIRED)
  set(HIOP_MATH_LIBRARIES "${LAPACK_LIBRARIES} ${HIOP_MATH_LIBRARIES}")
endif(NOT DEFINED LAPACK_LIBRARIES)
string(STRIP "${HIOP_MATH_LIBRARIES}" HIOP_MATH_LIBRARIES)

#
# extended precision lapack based on xblas testing example
#
#set(LAPACK_LIBRARIES "-lgfortran;/export/home/petra1/work/installs/xblas-1.0.248/libxblas.a;/home/petra1/work/installs/lapack-3.7.0/libblas.a;/home/petra1/work/installs/lapack-3.7.0/liblapack.a;/home/petra1/work/installs/lapack-3.7.0/liblapack.a;/home/petra1/work/installs/lapack-3.7.0/libblas.a;/export/home/petra1/work/installs/xblas-1.0.248/libxblas.a")

message("HiOp will use LAPACK/BLAS libs: ${LAPACK_LIBRARIES}")
message("HiOp: the so-called math libraries are: '${HIOP_MATH_LIBRARIES}'")

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
	      src/Optimization/hiopLogBarProblem.hpp
	      src/Optimization/hiopFilter.hpp
	      src/Optimization/hiopHessianLowRank.hpp
	      src/Optimization/hiopDualsUpdater.hpp
	      src/LinAlg/hiopVector.hpp
	      src/LinAlg/hiopMatrix.hpp
	      src/LinAlg/hiopMatrixSparseTriplet.hpp
	      src/LinAlg/hiopMatrixSparseTripletStorage.hpp
              src/LinAlg/hiopMatrixComplexSparseTriplet.hpp
	      src/LinAlg/hiopMatrixComplexDense.hpp
	      src/Utils/hiopRunStats.hpp
	      src/Utils/hiopLogger.hpp
	      src/Utils/hiopCSR_IO.hpp
	      src/Utils/hiopTimer.hpp
	      src/Utils/hiopOptions.hpp
              src/Utils/hiopKronReduction.hpp
	      src/Interface/hiop_defs.hpp
	      src/LinAlg/hiop_blasdefs.hpp
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
endif(HIOP_WITH_MAKETEST)
