# the file below is the file in which you define src files and include files
#include $(KNN_DIR)/makefile.in.files/makefile.in
include $(KNN_DIR)/makefile.inc

all: external rrot lib

.PHONY: docs tags lib test clean external rrot

tags : 
	-$(TAGS) */*.cpp */*.cxx include/*

lib:
	$(MAKE) -C $(KNN_DIR)/build/ all
	$(MAKE) -C $(KNN_DIR)/parallelIO
external:
	$(MAKE) -C $(KNN_DIR)/external/cmd
	/bin/mv $(KNN_DIR)/external/cmd/*.a $(KNN_DIR)/build

rrot:
	$(MAKE) -C $(KNN_DIR)/external/rrot
	/bin/mv $(KNN_DIR)/external/rrot/id_lib.a $(KNN_DIR)/build/librrot.a

clean:
	$(MAKE) -C $(KNN_DIR)/build clean
	$(MAKE) -C $(KNN_DIR)/parallelIO clean
	$(RM) */*.o


