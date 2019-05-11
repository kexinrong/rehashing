include ../makefile.in.files/makefile.$(PLATFORM)

TEST_ASKIT := test_askit_main.exe
TEST_ASKIT_OBJS := fks_ompNode.o fksTree.o id.o askit_utils.o gaussian_kernel.o laplace_kernel.o polynomial_kernel.o test_askit_main.o test_id.o

# TEST_DIST_ASKIT := test_distributed_main.exe
# TEST_DIST_ASKIT_OBJS := fks_ompNode.o fksTree.o id.o askit_utils.o gaussian_kernel.o laplace_kernel.o polynomial_kernel.o test_distributed_main.o

DIST_MAIN := askit_distributed_main.exe
DIST_MAIN_OBJS := fks_ompNode.o fksTree.o gaussian_kernel.o laplace_kernel.o polynomial_kernel.o id.o askit_distributed_main.o askit_utils.o 

KDE_MAIN := askit_kde_main.exe
KDE_MAIN_OBJS := fks_ompNode.o fksTree.o gaussian_kernel.o id.o askit_kde_main.o askit_utils.o 

ALL_OBJS := $(TEST_ASKIT_OBJS) $(DIST_MAIN_OBJS) $(KDE_MAIN_OBJS) # $(TEST_DIST_ASKIT_OBJS) 

TESTS := $(TEST_ASKIT) # $(TEST_DIST_ASKIT)

EXECUTABLES := $(DIST_MAIN) $(KDE_MAIN)

# Need this to recognize the dependencies on the template definitions
# this allows us to rebuild correctly after only editing a template file
TEMPLATE_INCLUDES := askit_alg.hpp askit_alg_impl.hpp askit_utils.hpp askit_utils_impl.hpp kernel_inputs.hpp id.hpp id_impl.hpp

KNN_INCLUDE := -I${KNN_DIR}/generator/ \
			   -I$(KNN_DIR)/include/ \
	           -I$(KNN_DIR)/include/binTree/ \
	           -I$(KNN_DIR)/include/direct_knn/ \
	           -I$(KNN_DIR)/include/repartition/ \
	           -I$(KNN_DIR)/include/parallelIO/ \
			   -I${KNN_DIR}/include/stTree/
KNN_LIB := -L$(KNN_DIR)/build -lknn -lrrot

CMD_INCLUDE := -I${KNN_DIR}/external/cmd/
CMD_LIB := -L${KNN_DIR}/build -lcmd

ifeq ($(USE_KS), true)
	KS_INCLUDE := -I${KS_DIR}/include/
	KS_LIB := ${KS_DIR}/lib/libks.a
	CPPFLAGS += -DUSE_KS=1
endif

ALL_INCLUDE += -I./ $(CMD_INCLUDE) $(KNN_INCLUDE) $(BLAS_INCLUDE) $(KS_INCLUDE)
LD_FLAGS += -L./ $(PLATFORM_LD_FLAGS) $(CMD_LIB) $(KNN_LIB) $(BLAS_LIBS) $(KS_LIB)

$(info * Using CPPFLAGS=${CPPFLAGS})
$(info * Using CFLAGS=${CFLAGS})
$(info * Using KNN_DIR=${KNN_DIR})
$(info * Using KS_DIR=${KS_DIR})
$(info * Using BLAS_LIBS=${BLAS_LIBS})
$(info * Using BLAS_INCLUDE=${BLAS_INCLUDE})
$(info * Using INCLUDE=${ALL_INCLUDE})
$(info * Using TEMPLATE_INCLUDES=${TEMPLATE_INCLUDES})
$(info * Using LD_FLAGS=${LD_FLAGS})
$(info -------------------------------------)

.PHONY: all

all: $(TESTS) $(EXECUTABLES)

tests: $(TESTS)

.PHONY: clean

$(TEST_ASKIT) : $(TEST_ASKIT_OBJS) $(TEMPLATE_INCLUDES)
	$(MPICXX) $(CPPFLAGS) $(CFLAGS) -o $@ $(TEST_ASKIT_OBJS) $(LD_FLAGS)

# $(TEST_DIST_ASKIT) : $(TEST_DIST_ASKIT_OBJS) $(TEMPLATE_INCLUDES)
# 	$(MPICXX) $(CPPFLAGS) $(CFLAGS) -o $@ $(TEST_DIST_ASKIT_OBJS) $(LD_FLAGS)

%.o: %.cpp
		$(MPICXX) -c $(CPPFLAGS) $(CFLAGS) $< -o $@ $(ALL_INCLUDE)

$(DIST_MAIN): $(DIST_MAIN_OBJS) $(TEMPLATE_INCLUDES)
	$(MPICXX) $(CPPFLAGS) $(CFLAGS) -o $@ $(DIST_MAIN_OBJS) $(LD_FLAGS)
	
$(KDE_MAIN): $(KDE_MAIN_OBJS) $(TEMPLATE_INCLUDES)
	$(MPICXX) $(CPPFLAGS) $(CFLAGS) -o $@ $(KDE_MAIN_OBJS) $(LD_FLAGS)
	

clean:
	rm -rf $(TESTS) $(EXECUTABLES) $(ALL_OBJS)

