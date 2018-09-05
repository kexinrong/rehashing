cmake_minimum_required(VERSION 3.5)

set(EIGEN3_DIR "lib/eigen-git-mirror/cmake")
find_package (Eigen3 3.3 REQUIRED)

include_directories(../utils)
add_library (data GenericInstance.h GenericInstance.cpp SyntheticData.cpp SyntheticData.h)
target_link_libraries(data Eigen3::Eigen)

