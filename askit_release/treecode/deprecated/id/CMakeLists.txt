
add_library(id
  id.hpp
  id.cpp
)
#target_link_libraries(id
#)
# don't link mkl here, it will be linked by whatever uses id

add_executable(id_test
  test_id.hpp
  test_id.cpp
  id_test_main.cpp
)
target_link_libraries(id_test
  mkl_intel_ilp64
  mkl_core
  mkl_intel_thread
  pthread
  imf
  m
  id
)


