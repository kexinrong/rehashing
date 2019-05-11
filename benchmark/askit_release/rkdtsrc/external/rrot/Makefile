ARCH=ar
ARCHFLAGS=cr
RANLIB=ranlib

###############################################################################


BIN_DIR=$(KNN_DIR)/external/rrot/bin
SRC_DIR=$(KNN_DIR)/external/rrot/src
TMP_DIR=$(KNN_DIR)/external/rrot/tmp
TEST_DIR=$(KNN_DIR)/external/rrot/test
RROT_DIR=$(KNN_DIR)/external/rrot
VPATH=$(SRC_DIR):$(BIN_DIR):$(TMP_DIR)

#CXXFLAGS += -openmp -fPIC -O3 -xSSE4.2 -ip -unroll-aggressive -no-prec-div -simd -opt-prefetch -use-intel-optimized-headers
FFLAGS += -fPIC 

ALL_ID_OBJS=id_rand.o id_rtrans.o 

all:$(ALL_ID_OBJS) id_lib.a #id_rtrans_test
	@echo "rrot done"
#	@echo
#	@echo "Now run test: randomly rotate 10,000 10D vectors"
#	@echo "  cd tmp; export OMP_NUM_THREADS=4; ./id_rtrans_test 10 10000"

id_lib.a:
	cd $(BIN_DIR); $(ARCH) $(ARCHFLAGS) ../$@ $(ALL_ID_OBJS)
	$(RANLIB) $@
.f.o:
	$(FC) $(FFLAGS) -c $< -o $(BIN_DIR)/$@

id_rtrans_test:
	cd $(TEST_DIR); $(CXX) $@.cc $(RROT_DIR)/id_lib.a -o $(TEM_DIR)$@

clean:
	rm -f *.a
