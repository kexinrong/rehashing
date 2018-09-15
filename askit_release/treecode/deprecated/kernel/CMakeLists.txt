add_library(gaussian_kernel
  gaussian_kernel.hpp
  gaussian_kernel.cpp
)
target_link_libraries(gaussian_kernel
  knn # needed for pairwise distances computation
)

add_executable(test_gaussian_kernel
  test_gaussian_kernel.cpp
)
target_link_libraries(test_gaussian_kernel
  gaussian_kernel
)