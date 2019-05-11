#EXEC=test_debug.exe 
EXEC=test_hypertree.exe test_ht_arbitraryN.exe test_mtree.exe test_coll_tree.exe ex1a2a.exe test_find_knn_cmd.exe


#LOC_SRC=mpitree.cpp groupdistribute.cpp
#LOC_OBJ=$(LOC_SRC:.cpp=.o)


#LOC_SRC=mpitree.cpp groupdistribute.cpp
#LOC_OBJ=$(LOC_SRC:.cpp=.o)

#include $(KNN_DIR)/makefile.in.files/makefile.in
include $(KNN_DIR)/makefile.inc

all: $(EXEC)

%.exe: %.o 
	$(CXX)  $< -o $ $(LOC_OBJ) $@ $(LDFLAGS) 

#.PHONY: clean

clean: 
		@-$(RM) *.o ${EXEC} *.out *.d


