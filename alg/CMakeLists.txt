cmake_minimum_required(VERSION 3.5)

set(EIGEN3_DIR "lib/eigen-git-mirror/cmake")
find_package (Eigen3 3.3 REQUIRED)

set(Boost_USE_MULTITHREADED ON)
find_package( Boost REQUIRED )
include_directories( ${Boost_INCLUDE_DIRS} )

include_directories(../utils)
add_library (alg multiHBE.cpp multiHBE.h naiveKDE.h naiveKDE.cpp RS.h RS.cpp BaseLSH.cpp BaseLSH.h HashBucket.h HashTable.h MoMEstimator.h)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/modules)
find_package(MKL)
if(MKL_FOUND)
    include_directories(${MKL_INCLUDE_DIRS})
    target_link_libraries(alg ${MKL_LIBRARIES})
endif()

target_link_libraries(alg Eigen3::Eigen)

