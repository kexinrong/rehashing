
add_library(fkstree
  fks_ompNode.h
  fks_ompNode.cpp
  fksTree.h
  fksTree.cpp
)
target_link_libraries(fkstree
  cmd
  knn
  rrot
)

add_executable(test_fkstree
  test.cpp
)
target_link_libraries(test_fkstree
  fkstree
)


