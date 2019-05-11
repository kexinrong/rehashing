ifndef KNN_DIR
$(info see ./buildme file, modify it and run ./builme_PLATFORM)
$(error make stopped because KNN_DIR environment is undefined)

endif

ifndef MPICXX
$(error make stopped because MPICXX is undefined)
endif

ifndef MPICC
$(error make stopped because MPICC is undefined)
endif

export KNN_DIR
export MPICXX
export KS_DIR
export USE_KS
export USE_MKL
export USE_GNU
export USE_INTEL
export PLATFORM

# this file needs to be included by all subdirectories
include makefile.in.files/makefile.$(PLATFORM)


all: src lib

.PHONY: src help clean lib

src:
	$(MAKE) -C src all

lib:
	$(MAKE) -C build all

clean:
	$(MAKE) -C src clean
	$(RM) */*.o


