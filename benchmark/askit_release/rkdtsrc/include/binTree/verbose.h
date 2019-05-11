#ifndef _VERBOSE_H__
#define _VERBOSE_H__
// various debug options


// stree

#define STTREE_LOAD_BALANCE_VERBOSE 0

#define STTREE_TIMING_VERBOSE 0


// super charing

#define SC_DEBUG_VERBOSE 0		// super charging debug

#define SC_STAGE_VERBOSE 0

#define SC_TIMING_VERBOSE 0


// rkdt
#define RKDT_ITER_VERBOSE 0

#define RKDT_MERGE_VERBOSE 0

#define RKDT_FINAL_KNN_OUTPUT_VERBOSE 0


// parallel tree
#define PART_DEBUG_VERBOSE 0


// performance report
#define ALL_TO_ALL_VERBOSE 0

#define STAGE_OUTPUT_VERBOSE 0

#define STAGE_DTREE_OUTPUT_VERBOSE 0

#define STAGE_STREE_OUTPUT_VERBOSE 0

#define LOAD_BALANCE_VERBOSE 0

#define COMM_TIMING_VERBOSE 1	// communication timing (all-to-all and comm splitting)

#define OVERALL_TREE_TIMING_VERBOSE 1

#define DETAIL_VERBOSE 0

#define TREE_DEBUG_VERBOSE 0

#define PCL_DEBUG_VERBOSE 0

// extern global timing variable
extern double Tree_Const_T_;

extern double Tree_Search_T_;

extern double STree_Const_T_;

extern double Direct_Kernel_T_;

extern double STree_Search_T_;

extern double STree_Direct_Kernel_T_;

extern double Repartition_Tree_Build_T_;

extern double Repartition_Query_T_;

extern double Repartition_T_;

extern double Comm_Split_T_;

extern double MPI_Collective_T_;

extern double MPI_Collective_Const_T_;

extern double MPI_Collective_Query_T_;

extern double COMPUTE_DIST_T_;

extern double MAX_HEAP_T_;

extern double MPI_Collective_KNN_T_;


#define MPI_CALL(CODE)			\
  {\
    double tic = omp_get_wtime();		\
    (CODE);  \
    double toc = omp_get_wtime()-tic;\
    MPI_Collective_T_ += toc;\
  }\


#endif
