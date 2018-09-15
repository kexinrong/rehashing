
add_library(treecode_driver
  treecode_driver.hpp
  treecode_driver.cpp
)
target_link_libraries(treecode_driver
  gaussian_kernel
  id
  fkstree
)

add_executable(test_treecode
  test_main.cpp
)
target_link_libraries(test_treecode
  treecode_driver
)


add_executable(treecode_shared_mem
  treecode_shared_main.cpp
)
target_link_libraries(treecode_shared_mem
  treecode_driver
)