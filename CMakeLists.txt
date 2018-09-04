cmake_minimum_required(VERSION 3.5)
project(hbe)

set(CMAKE_CXX_STANDARD 14)
set(EIGEN3_DIR "lib/eigen-git-mirror/cmake")
find_package (Eigen3 3.3 REQUIRED)

add_executable(hbe library.cpp library.h kernel/kernel.h kernel/expkernel.h utils/utils.cpp utils/utils.h alg/naiveKDE.cpp alg/naiveKDE.h)
target_link_libraries(hbe Eigen3::Eigen)