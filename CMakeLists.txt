cmake_minimum_required(VERSION 3.5)
project(hbe)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(utils)
add_subdirectory(alg)
add_subdirectory(data)

add_executable(hbe SyntheticDensity.cpp SyntheticDensity.h)
target_link_libraries(hbe alg data)
