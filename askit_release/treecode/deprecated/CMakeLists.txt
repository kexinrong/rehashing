cmake_minimum_required(VERSION 2.6)
project(lafmm_treecode CXX)

option(DEBUG "Compile with debugging information" ON)

find_package(MPI REQUIRED)
include_directories(MPI_INCLUDE_PATH)

set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpic++)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS})
# include dir for mkl
# TODO: handle this correctly
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMKL_ILP64 -openmp -DTIXML_USE_STL -I/share/apps/intel/Compiler/12.1/mkl/include -fPIC -Wall -mkl=parallel")

#set(CMAKE_CXX_LINKER_FLAGS "${CMAKE_CXX_LINKER_FLAGS} -mkl=parallel")

if(DEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif(DEBUG)

# for mkl
# TODO: make this not suck -- do it portably
link_directories(/share/apps/intel/Compiler/12.1/mkl/lib/intel64)


# for Bo's tree code
#set(KNN_DIR /h1/march/knn/src)
set(KNN_DIR /h2/bo/git.ronaldo/knn/src)

find_library(
    KNN_LIB knn
    HINTS "${KNN_DIR}/build"
)

message(STATUS "dir: ${KNN_DIR}/build")

if(${KNN_LIB} STREQUAL "KNN_LIB-NOTFOUND")
    message(FATAL_ERROR "Couldn't find the 'knn' library")
endif()

message(STATUS "Found knn at:  ${KNN_LIB} " )

find_library(
    RROT_LIB rrot
    HINTS "${KNN_DIR}/build"
)

find_library(
    CMD_LIB cmd
    HINTS "${KNN_DIR}/build"
)

link_directories(${KNN_DIR}/build)




include_directories(${KNN_DIR}/generator)
include_directories(${KNN_DIR}/include)
include_directories(${KNN_DIR}/include/binTree)
include_directories(${KNN_DIR}/include/direct_knn)
include_directories(${KNN_DIR}/include/repartition)
include_directories(${KNN_DIR}/include/lsh)
include_directories(${KNN_DIR}/include/parallelIO)
include_directories(${KNN_DIR}/include/stTree)


include_directories(${KNN_DIR}/external/cmd/)

add_subdirectory(id)
add_subdirectory(kernel)
add_subdirectory(fkstree)
add_subdirectory(driver)




