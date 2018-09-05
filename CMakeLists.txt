cmake_minimum_required(VERSION 3.5)
project(hbe)

set(CMAKE_CXX_STANDARD 11)

include_directories(utils)
add_subdirectory(alg)

add_executable(hbe library.cpp library.h utils/CSVparser.h)
target_link_libraries(hbe alg)
