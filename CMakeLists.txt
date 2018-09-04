cmake_minimum_required(VERSION 3.5)
project(hbe)

set(CMAKE_CXX_STANDARD 14)

include_directories(utils)
add_subdirectory(alg)

add_executable(hbe library.cpp library.h)
target_link_libraries(hbe alg)
