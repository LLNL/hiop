cmake_minimum_required (VERSION 2.8)
project (hippopde)
# enable Fortran for Fortran name mangling
enable_language(Fortran)

# Create header for Fortran name mangling
include(FortranCInterface)
FortranCInterface_HEADER(FortranCInterface.hpp MACRO_NAMESPACE "FC_")

option(WITH_MPI "Build with MPI support" ON)
option(DEEP_CHECKING "Extra checks and asserts in the code with a high penalty on performance" ON)

#with testing drivers capable of 'selfchecking' (-selfcheck)
option(WITH_MAKETEST "Enable 'make test'" ON)

if(WITH_MPI)
  if(NOT DEFINED MPI_CXX_COMPILER)
    find_package(MPI REQUIRED)
    set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
  endif(NOT DEFINED MPI_CXX_COMPILER)
  add_definitions(-DWITH_MPI)
endif(WITH_MPI)

if(DEEP_CHECKING)
    add_definitions(-DDEEP_CHECKING)
endif(DEEP_CHECKING)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fpermissive")

find_package(OpenMP)

if(NOT DEFINED LAPACK_LIBRARIES)
  # in case the toolchain defines them
  find_package(LAPACK REQUIRED)
endif(NOT DEFINED LAPACK_LIBRARIES)
message("LAPACK/BLAS libs: ${LAPACK_LIBRARIES}")

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

#add_subdirectory(src)

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

install(TARGETS hiop DESTINATION lib)
install(FILES src/Interface/hiopInterface.hpp
	      src/Optimization/hiopNlpFormulation.hpp
	      src/Optimization/hiopAlgFilterIPM.hpp
	      src/Optimization/hiopIterate.hpp
	      src/Optimization/hiopResidual.hpp
	      src/Optimization/hiopLogBarProblem.hpp
	      src/Optimization/hiopFilter.hpp
	      src/Optimization/hiopHessianLowRank.hpp
	      src/Optimization/hiopDualsUpdater.hpp
	      src/LinAlg/hiopVector.hpp
	      src/LinAlg/hiopMatrix.hpp
	      src/Utils/hiopRunStats.hpp
	      src/Utils/hiopLogger.hpp
	      src/Utils/hiopTimer.hpp
	      src/Utils/hiopOptions.hpp
        DESTINATION include)

##########################################################
# CMake Tests
##########################################################
if (WITH_MAKETEST)
  enable_testing()
  add_test(NAME NlpDenseCons1_5H COMMAND $<TARGET_FILE:nlpDenseCons_ex1.exe>   500 1.0 -selfcheck)
  add_test(NAME NlpDenseCons1_5K COMMAND $<TARGET_FILE:nlpDenseCons_ex1.exe>  5000 1.0 -selfcheck)
  add_test(NAME NlpDenseCons2_5H COMMAND $<TARGET_FILE:nlpDenseCons_ex2.exe>   500 -selfcheck)
  add_test(NAME NlpDenseCons2_5K COMMAND $<TARGET_FILE:nlpDenseCons_ex2.exe>  5000 -selfcheck)
  add_test(NAME NlpDenseCons2_50K COMMAND $<TARGET_FILE:nlpDenseCons_ex2.exe> 50000 -selfcheck)
  if(WITH_MPI)
    add_test(NAME NlpDenseCons2_50K_mpi COMMAND mpirun -np 2 $<TARGET_FILE:nlpDenseCons_ex2.exe> 50000 -selfcheck)
  endif(WITH_MPI)
endif(WITH_MAKETEST)
