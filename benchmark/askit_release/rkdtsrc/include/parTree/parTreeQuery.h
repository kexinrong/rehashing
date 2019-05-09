#ifndef _PARTREESEARCH_H__
#define _PARTREESEARCH_H__

#include <vector>
#include <algorithm>

#include "parTree.h"
#include "binTree.h"

using namespace std;

// if some A[i].first == B[j].first, always choose A[i], and remove B[j], the value of B might changed
void part_knn_merge( vector< pair<double, long> > &A, vector< pair<double, long> > &B, int n, int k,
					   vector< pair<double, long> > &result);


void part_query_greedy_a2a(pbinData refData, int k, int flag_r, 
						   int max_points, int max_tree_level,
						   vector<long>* queryIDs,
						   vector< pair<double, long> >*kNN, 
					       MPI_Comm comm);


void part_query_greedy(pbinData refData, pbinData queryData,
					   int k, int flag_r, int max_points, int max_tree_level,
					   vector<long>* queryIDs,
					   vector< pair<double, long> >*kNN, 
					   MPI_Comm comm);


void rkdt_a2a(pbinData refData, int k, int numof_iters,
			  int max_points, int max_tree_level, 
			  int flag_r, int flag_c,
			  // debug options
			  int verify_verbose,
			  // output
			  vector<long> &queryIDs,
			  vector< pair<double, long> >* &kNN,
			  MPI_Comm comm);


void rkdt(pbinData refData, pbinData queryData, int k, int numof_iters,
			  int max_points, int max_tree_level, 
			  int flag_r, int flag_c,
			  // debug options
			  int verify_verbose,
			  // output
			  vector<long> &queryIDs,
			  vector< pair<double, long> >* &kNN,
			  MPI_Comm comm);


#endif
