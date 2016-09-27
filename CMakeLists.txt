cmake_minimum_required (VERSION 2.8.11)
project (hippopde)

option(WITH_MPI "Build with MPI support" ON)
option(DEEP_CHECKING "Extra checks and asserts in the code with a high penalty on speed" ON)

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

add_subdirectory(Optimization)
add_subdirectory(LinAlg)
add_subdirectory(Drivers)


