#ifndef _OLDQUERY_H_
#define _OLDQUERY_H_

#include <mpi.h>
#include <cassert>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <utility>
#include <util.h>


using namespace std;

namespace oldtree {

	void queryR( pbinData inData, long rootNpoints, double dupFactor,
				poldNode searchNode, double range,
				vector< pair<double, long> > *&neighbors);


	void queryR(  pbinData inData, long rootNpoints, double dupFactor,
				poldNode searchNode,
				vector< pair<double, long> > *&neighbors,
				int *nvectors);


	void queryRK( pbinData inData, long rootNpoints, double dupFactor, 
				int k, poldNode searchNode,
				vector< pair<double, long> > *&neighbors,int *nvectors);

	void queryKSelectRs(pbinData redistQuery,
		              pbinData homeData, poldNode searchNode,
					  int global_numof_query_points,
					  int k, double **R);

	void queryK( pbinData inData, long rootNpoints, double dupFactor,
				poldNode searchNode, int k, 
				vector<long> *queryIDs,
				vector< pair<double, long> > *kNN);


	void queryK_Greedy( pbinData inData, long rootNpoints, 
							 poldNode searchNode, 
							 int k, 
							 int traverse_type, int max_points, int max_tree_level,
							 vector<long> *queryIDs,
							 vector< pair<double, long> > *kNN);

	
	void queryK_Greedy_a2a( long rootNpoints, 
							 poldNode searchNode, 
							 int k, 
							 int max_points, int max_tree_level,
							 vector<long> *queryIDs,
							 vector< pair<double, long> > *kNN);
	
	void knnTreeSearch_RandomRotation_a2a(pbinData refData,
									  int k,
									  int numof_iterations,
									  treeParams params,
									  int flag_r, int flag_c,
									  vector<long> &queryIDs,
									  vector< pair<double, long> > * &kNN);
	

	void knnTreeSearch_RandomRotation(pbinData refData,
									  pbinData queryData,
									  int k,
									  int numof_iterations,
									  treeParams params,
									  int flag_r, int flag_c,
									  vector<long> &queryIDs,
									  vector< pair<double, long> > * &kNN);


	void knn_merge( vector< pair<double, long> > &A, 
					vector< pair<double, long> > &B,
					int n, 
					int k,
					vector< pair<double, long> > &result);


	void find_knn_single_query(double *ref, double *query, long *ref_ids,
							   int numof_ref_points, int dim, int k,
								// output
								pair<double, long> *result,
								// auxliary 
								double *diff = NULL, pair<double, long> *dist = NULL);
}

#endif












