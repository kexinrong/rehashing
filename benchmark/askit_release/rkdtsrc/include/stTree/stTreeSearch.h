#ifndef _STTREESEARCH_H__
#define _STTREESEARCH_H__

#include <vector>
#include <algorithm>

#include "binTree.h"

using namespace std;

void knn_merge( vector< pair<double, long> > &A, vector< pair<double, long> > &B, int n, int k,
				vector< pair<double, long> > &result);


void single_merge(const pair<double, long> *A, const pair<double, long> *B, int k,
                    pair<double, long> *C, vector< pair<double, long> > &auxVec);



// random rotation kd tree
void stTreeSearch_rkdt( double *ref, double *query, 
						long *refids, long *queryids,				
						int numof_ref_points, int numof_query_points, int dim,
						int max_points, int max_tree_level,
						int k, int numof_iterations,
						int flag_r,
						//output
						vector< pair<double, long> >* &kNN);


// random rotation kd tree, special for all-to-all case
void stTreeSearch_rkdt_a2a( double *ref, double *query,
						long *refids, long *queryids,
						int numof_ref_points, int numof_query_points, int dim,
						int max_points, int max_tree_level,
						int k, int numof_iterations,
						int flag_r,
						//output
						vector< pair<double, long> >* &kNN);


// random rotation kd tree, special for all-to-all case, memory efficient
void stTreeSearch_rkdt_a2a_me( pbinData refData, int k, int flag_r,
							   int max_points, int max_tree_level,
								//output
								vector< pair<double, long> >* &kNN);


// random rotation kd tree
void stTreeSearch_rkdt_me( pbinData refData, pbinData queryData,
						int k, int flag_r,
						int max_points, int max_tree_level,
						//output
						vector< pair<double, long> >* &kNN);

// random sampling metric tree
void stTreeSearch_rsmt( double *ref, double *query,
						long *refids, long *queryids,
						int numof_ref_points, int numof_query_points, int dim,
						int max_points, int max_tree_level,
						int k, int numof_iterations,
						//output
						vector< pair<double, long> > &kNN);


// new random rotation kd tree, save memory
void stTreeSearch_rkdt( binData *&refData, binData *queryData, int k,
						int max_points, int max_tree_level, int numof_iterations, int flag_r, bool flag_rotate_back,
						//output
						vector< pair<double, long> >* &kNN);


void find_knn_srkdt( binData *&refData, binData *queryData, int k,
					 int max_points, int max_tree_level, int numof_iterations,
					 vector< pair<double, long> >* &kNN);


void find_knn_srkdt( binData *&refData, binData *queryData, int k,
					 int max_points, int max_tree_level, int numof_iterations,
					 vector< pair<double, long> >* &kNN,
                     vector<long> &queryIDs,
                     vector<long> &sampleIDs, vector<double> &globalKdist, vector<long> &globalKid);


void find_knn_srkdt_a2a( binData *&refData, int k,
					 int max_points, int max_tree_level, int numof_iterations,
					 vector< pair<double, long> >* &kNN);


void find_knn_srkdt_a2a( binData *&refData, int k,
					 int max_points, int max_tree_level, int numof_iterations,
					 vector< pair<double, long> >* &kNN,
                     vector<long> &queryIDs,
                     vector<long> &sampleIDs, vector<double> &globalKdist, vector<long> &globalKid);







#endif
