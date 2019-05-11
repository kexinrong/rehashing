#ifndef _EVAL_H__
#define _EVAL_H__

#include <stdlib.h>
#include <vector>

#include "binTree.h"
#include "binQuery.h"
#include "distributeToLeaf.h"

using namespace std;


void printBinTree(pbinNode in_node);

void saveBinTree(char * filename, pbinNode in_node);

void get_sample_info(double *ref, double *query, long *refids, long *queryids,
					int numof_ref_points, int numof_query_points,
					int dim, int k,
					// output
					vector<long> &sampleIDs,
					vector<double> &globalKdist,
					vector<long> &globalKid);

void verify(// input: direct search info
				vector<long> &sampleIDs,
				vector<double> &globalKdist,
				vector<long> &globalKid,
				// input: approx search info
				vector<long> &queryIDs,
				vector< pair<double, long> > &kNN,
				// output
				int &numof_missed_neighbors,
				double &hit_rate,
				double &relative_error);



void verify(// input: direct search info
				vector<long> &sampleIDs,
				vector<double> &globalKdist,
				vector<long> &globalKid,
				// input: approx search info
				vector<long> &queryIDs,
				vector< pair<double, long> > &kNN,
				// output
				int &numof_missed_neighbors,
				double &hit_rate,
				double &relative_error,
                double &glb_total_true_dist,
                double &glb_total_knn_dist);



void verify(// input: direct search info
                vector<long> &sampleIDs,
                vector<double> &globalKdist,
                vector<long> &globalKid,
            // input: approx search info
                vector<long> &queryIDs,
                vector< pair<double, long> > &kNN,
                MPI_Comm comm,
            // output
                int &numof_missed_neighbors,
                double &hit_rate,
                double &relative_error);



void evaluation_sample( double *ref, double *query,
					  long *refids, long *queryids,
					  int numof_ref_points, int numof_query_points,
					  int dim, int k,
					  vector<long> &queryIDs,
					  vector< pair<double, long> > &kNN);


void evaluation_full( double *ref, double *query,
					  long *refids, long *queryids,
					  int numof_ref_points, int numof_query_points,
					  int dim, int k,
					  vector<long> &outQueryIDs,
					  vector< pair<double, long> > &approxkNN);

#endif
