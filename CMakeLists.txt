cmake_minimum_required(VERSION 3.5)
project(hbe)

set(CMAKE_CXX_STANDARD 14)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fopenmp -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

include_directories(utils)
add_subdirectory(alg)
add_subdirectory(data)

#add_executable(hbe SyntheticFindHBE.cpp)
#add_executable(hbe SyntheticFindRandom.cpp SyntheticFindRandom.h)
add_executable(hbe experiments/SyntheticFixAcc.cpp experiments/SyntheticFixAcc.cpp)
target_link_libraries(hbe alg data)
