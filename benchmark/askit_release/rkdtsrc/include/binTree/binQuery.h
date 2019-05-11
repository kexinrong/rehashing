#ifndef _BINQUERY_H_
#define _BINQUERY_H_

#include <mpi.h>
#include <cassert>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <utility>
#include <util.h>

#include "binQuery.h"

using namespace std;

namespace bintree {

	void queryR( pbinData inData, long rootNpoints, double dupFactor,
				pbinNode searchNode, double range,
				vector< pair<double, long> > *&neighbors);


	void queryR(  pbinData inData, long rootNpoints, double dupFactor,
				pbinNode searchNode,
				vector< pair<double, long> > *&neighbors,
				int *nvectors);


	void queryRK( pbinData inData, long rootNpoints, double dupFactor,
				    int k, pbinNode searchNode,
				    vector< pair<double, long> > *&neighbors,int *nvectors);

	void queryKSelectRs(pbinData redistQuery,
		              pbinData homeData, pbinNode searchNode,
					  int global_numof_query_points,
					  int k, double **R);

	void queryK( pbinData inData, long rootNpoints, double dupFactor,
				pbinNode searchNode, int k,
				vector<long> *queryIDs,
				vector< pair<double, long> > *kNN);

	void queryK_Greedy( pbinData inData, long rootNpoints,
							 pbinNode searchNode,
							 int k,
							 int traverse_type, int max_points, int max_tree_level,
							 vector<long> *queryIDs,
							 vector< pair<double, long> > *kNN);


	void queryK_Greedy_a2a( long rootNpoints,
							 pbinNode searchNode,
							 int k,
							 int max_points, int max_tree_level,
							 vector<long> *queryIDs,
							 vector< pair<double, long> > *kNN);


    // currently, I just add one function to quickly fix the bugs for fmm
    void queryK_Greedy_a2a( long rootNpoints,
							 pbinNode searchNode, int k,
							 int max_points, int max_tree_level,
                             vector<long> &inscan_numof_query_points_per_rank,
							 vector<long> *queryIDs,
							 vector< pair<double, long> > *kNN);



	pair<double, double> knnTreeSearch_RandomRotation_a2a(pbinData refData,
									  int k,
									  int numof_iterations,
									  treeParams params,
									  int flag_r, int flag_c,
									  vector<long> &queryIDs,
									  vector< pair<double, long> > * &kNN,
                                      double tol_hit = 100.0,
                                      double tol_err = 0.0);


	void knnTreeSearch_RandomRotation(pbinData refData,
									  pbinData queryData,
									  int k,
									  int numof_iterations,
									  treeParams params,
									  int flag_r, int flag_c,
									  vector<long> &queryIDs,
									  vector< pair<double, long> > * &kNN,
                                      double tol_hit = 100.0,
                                      double tol_err = 0.0);

	void knnTreeSearch_RandomSampling(pbinData refData,
									  pbinData queryData,
									  int k,
									  int numof_iterations,
									  treeParams params,
									  vector<long> &queryIDs,
									  vector< pair<double, long> > &kNN);


	void superCharging_p2p(	double *ref, int numof_ref_points, int dim, int k,
					vector< pair<double, long> > &all2all_kNN,
					// output
					vector< pair<double, long> > &sckNN,
					MPI_Comm comm);


	void superCharging(	double *ref, int numof_ref_points, int dim, int k,
					vector< pair<double, long> > &all2all_kNN,
					// output
					vector< pair<double, long> > &sckNN,
					MPI_Comm comm);


	void find_knn_single_query(double *ref, double *query, long *ref_ids,
							   int numof_ref_points, int dim, int k,
								// output
								pair<double, long> *result,
								// auxliary
								double *diff = NULL, pair<double, long> *dist = NULL);

	void getNeighborCoords(int numof_ref_points, double * ref, int dim,
		                   int numof_query_points, int k, vector< pair<double, long> > &all2all_kNN,
		                   // output
		                   double * neighborCoords,    // numof_ref_points * k * dim
		                   MPI_Comm comm);


}

#endif



