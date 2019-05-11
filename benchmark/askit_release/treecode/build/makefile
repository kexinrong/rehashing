include ../makefile.in.files/makefile.$(PLATFORM)

ASKIT_SRC=../src/askit_utils.cpp \
		  ../src/id.cpp \
		  ../src/fksTree.cpp \
		  ../src/fks_ompNode.cpp \
		  ../src/gaussian_kernel.cpp \
		  ../src/polynomial_kernel.cpp \
		  ../src/laplace_kernel.cpp 

ASKIT_LIB_OBJ = $(ASKIT_SRC:.cpp=.o)

ARFILES = libaskit.a
all: $(ARFILES)

$(ARFILES): $(ASKIT_LIB_OBJ)
	$(AR) $(ARFLAGS) ./$(ARFILES) $(ASKIT_LIB_OBJ)
	$(RANLIB) ./$(ARFILES)

clean:
	-$(RM) *.a

