cmake_minimum_required (VERSION 2.8)
project (hippopde)
# enable Fortran for Fortran name mangling
enable_language(Fortran)

# Create header for Fortran name mangling
include(FortranCInterface)
FortranCInterface_HEADER(FortranCInterface.hpp MACRO_NAMESPACE "FC_")

file(WRITE "src/Interface/hiop_defs.hpp" "")

option(HIOP_USE_MPI "Build with MPI support" ON)
option(HIOP_DEEPCHECKS "Extra checks and asserts in the code with a high penalty on performance" ON)

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

if(HIOP_DEEPCHECKS)
  file(APPEND "src/Interface/hiop_defs.hpp" "#define HIOP_DEEPCHECKS\n")
else()
  file(APPEND "src/Interface/hiop_defs.hpp" "#ifdef HIOP_DEEPCHECKS\n#undef HIOP_DEEPCHECKS\n#endif\n")
endif(HIOP_DEEPCHECKS)

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
	      src/Interface/hiop_defs.hpp
        DESTINATION include)

##########################################################
# CMake Tests
##########################################################
if (HIOP_WITH_MAKETEST)
  enable_testing()
  add_test(NAME NlpDenseCons1_5H COMMAND $<TARGET_FILE:nlpDenseCons_ex1.exe>   500 1.0 -selfcheck)
  add_test(NAME NlpDenseCons1_5K COMMAND $<TARGET_FILE:nlpDenseCons_ex1.exe>  5000 1.0 -selfcheck)
  add_test(NAME NlpDenseCons2_5H COMMAND $<TARGET_FILE:nlpDenseCons_ex2.exe>   500 -selfcheck)
  add_test(NAME NlpDenseCons2_5K COMMAND $<TARGET_FILE:nlpDenseCons_ex2.exe>  5000 -selfcheck)
  add_test(NAME NlpDenseCons2_50K COMMAND $<TARGET_FILE:nlpDenseCons_ex2.exe> 50000 -selfcheck)
  if(HIOP_USE_MPI)
    add_test(NAME NlpDenseCons2_50K_mpi COMMAND mpirun -np 2 $<TARGET_FILE:nlpDenseCons_ex2.exe> 50000 -selfcheck)
  endif(HIOP_USE_MPI)
endif(HIOP_WITH_MAKETEST)
