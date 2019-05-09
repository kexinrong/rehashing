## ARCHITECTURE DEPENDENT ---- PLEASE EDIT THESE THREE LINES (for example if you use gcc -openmp needs to be -fopenmp)

COMPFLAGS=-qopenmp -fPIC -O3
COMPOPTFLAGS = -xSSE4.2 -ip -unroll-aggressive -no-prec-div -simd -qopt-prefetch
BLAS_LIBS = -mkl=parallel -qopenmp
export MKLROOT=/opt/intel/mkl
export LDFLAGS=-L/opt/intel/mkl/lib/intel64
#CPPFLAGS = -fopenmp -fPIC -fno-exceptions -w -O3 -finline-functions -funroll-loops
#BLAS_INCLUDE = -DMKL_ILP64 -m64 -I${MKLROOT}/include
#BLAS_LIBS = -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl
#--------------- SHOULD NOT BE NECESSARY TO  EDIT BELOW THIS LINE 


RANLIB = ranlib
DOX = doxygen
TAGS = etags
ARFLAGS = -r
CC = ${MPICC}
CXX= ${MPICXX}
FC = ${MPIF90}
CXXFLAGS += ${COMPFLAGS} ${COMPOPTFLAGS}
FFLAGS= ${COMPFLAGS}
KNN_TESTING ?= no

# THESE ARE FOR THE COMPILE; don not change below this line
#/* ************************************************** */
# define source and include files

KNN_INCLUDE =\
 -I$(KNN_DIR)/include \
 -I$(KNN_DIR)/include/parallelIO  -I$(KNN_DIR)/include/direct_knn \
 -I$(KNN_DIR)/include/repartition \
 -I$(KNN_DIR)/include/pcltree -I$(KNN_DIR)/include/metricTree \
 -I$(KNN_DIR)/include/lsh -I$(KNN_DIR)/include/binTree \
 -I$(KNN_DIR)/include/stTree \
 -I$(KNN_DIR)/include/parTree -I$(KNN_DIR)/include/oldtree \
 -I$(KNN_DIR)/generator \
 -I$(KNN_DIR)/include/forest

KNN_SRC=\
    $(KNN_DIR)/direct_knn/direct_knn.cpp \
	$(KNN_DIR)/parallelIO/parallelIO.cpp \
	$(KNN_DIR)/direct_knn/knnreduce.cpp\
	$(KNN_DIR)/lsh/lsh.cpp \
	$(KNN_DIR)/lsh/dist_lsh.cpp \
	$(KNN_DIR)/repartition/repartition.cpp \
	$(KNN_DIR)/repartition/file_io.cpp \
	$(KNN_DIR)/repartition/clustering.cpp \
	$(KNN_DIR)/generator/generator.cpp $(KNN_DIR)/generator/hyperSphereSurface.cpp \
	$(KNN_DIR)/generator/normal.cpp $(KNN_DIR)/generator/uniform.cpp \
	$(KNN_DIR)/generator/sphere.cpp $(KNN_DIR)/generator/mixOfGaussian.cpp \
	$(KNN_DIR)/generator/gaussian.cpp $(KNN_DIR)/generator/embedding.cpp \
	$(KNN_DIR)/pcltree/groupdistribute.cpp \
	$(KNN_DIR)/binTree/binTree.cpp \
	$(KNN_DIR)/binTree/distributeToLeaf.cpp \
	$(KNN_DIR)/binTree/binQuery.cpp \
	$(KNN_DIR)/binTree/eval.cpp \
	$(KNN_DIR)/binTree/rotation.cpp \
	$(KNN_DIR)/binTree/verbose.cpp \
	$(KNN_DIR)/stTree/stTree.cpp \
	$(KNN_DIR)/stTree/srkdt.cpp \
	$(KNN_DIR)/stTree/stTreeSearch.cpp \
	$(KNN_DIR)/parTree/parTree.cpp \
	$(KNN_DIR)/parTree/parTreeQuery.cpp \
	$(KNN_DIR)/pcltree/mpitree.cpp \
	$(KNN_DIR)/pcltree/queryK.cpp \
	$(KNN_DIR)/pcltree/queryR.cpp \
	$(KNN_DIR)/oldtree/oldTree.cpp \
	$(KNN_DIR)/oldtree/oldQuery.cpp \
	$(KNN_DIR)/oldtree/distributeToLeaf_ot.cpp \
	$(KNN_DIR)/oldtree/gatherTree_ot.cpp \
	$(KNN_DIR)/binTree/gatherTree.cpp \
	$(KNN_DIR)/forest/forest.cpp

KNN_EXTERNAL_INCLUDE =  -I$(KNN_DIR)/external/cmd

#/* ************************************************** */

ALL_INCLUDE = $(KNN_INCLUDE) $(KNN_EXTERNAL_INCLUDE) $(BLAS_INCLUDE)
KNN_LIBS= -L$(KNN_DIR)/build -lknn -lcmd -lrrot
ALL_LIBS= $(KNN_LIBS) $(BLAS_LIBS)	

# main flags for compilers
CXXFLAGS+=$(ALL_INCLUDE) 
CPPFLAGS+=$(CXXFLAGS)
LDFLAGS+= $(ALL_LIBS) $(BLAS_LIBS)



