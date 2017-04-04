cmake_minimum_required (VERSION 2.8.11)
project (hippopde)

option(WITH_MPI "Build with MPI support" ON)
option(DEEP_CHECKING "Extra checks and asserts in the code with a high penalty on performance" ON)

#with testing drivers capable of 'selfchecking' (-selfcheck)
option(WITH_MAKETEST "Enable 'make test'" ON)

if(WITH_MPI)
    set(CMAKE_CXX_COMPILER "mpicxx")
    add_definitions(-DWITH_MPI)
endif(WITH_MPI)

if(DEEP_CHECKING)
    add_definitions(-DDEEP_CHECKING)
endif(DEEP_CHECKING)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fpermissive")

find_package(OpenMP)
find_package(LAPACK REQUIRED)
#message(STATUS " LAPACK_LIBRARIES:  ${LAPACK_LIBRARIES}")
#message(STATUS " CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES:  ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}")
#set(MATH_LIBS ${LAPACK_LIBRARIES}) #-ldl -l${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES} ${OpenMP_CXX_FLAGS})


include_directories(Interface)
include_directories(Optimization)
include_directories(LinAlg)
include_directories(Utils)

add_subdirectory(Optimization)
add_subdirectory(LinAlg)
add_subdirectory(Drivers)
add_subdirectory(Utils)


add_library(hiop STATIC $<TARGET_OBJECTS:hiopOptimization>
                        $<TARGET_OBJECTS:hiopLinAlg>
			$<TARGET_OBJECTS:hiopUtils>)

##########################################################
# CMake Tests
##########################################################
if (WITH_MAKETEST)
  enable_testing()
  add_test(NAME NlpDenseCons2_5H COMMAND $<TARGET_FILE:nlpDenseCons_ex2.exe>   500 -selfcheck)
  add_test(NAME NlpDenseCons2_5K COMMAND $<TARGET_FILE:nlpDenseCons_ex2.exe>  5000 -selfcheck)
  add_test(NAME NlpDenseCons2_50K COMMAND $<TARGET_FILE:nlpDenseCons_ex2.exe> 50000 -selfcheck)
  if(WITH_MPI)
    add_test(NAME NlpDenseCons2_50K_mpi COMMAND mpirun -np 2 $<TARGET_FILE:nlpDenseCons_ex2.exe> 50000 -selfcheck)
  endif(WITH_MPI) 
endif(WITH_MAKETEST)
