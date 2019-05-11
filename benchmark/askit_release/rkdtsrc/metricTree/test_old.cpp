#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <assert.h>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <map>
#include <vector>

#include "file_io.h"
#include "repartition.h"
#include "CmdLine.h"
#include "generator.h"
#include "clustering.h"
#include "direct_knn.h"
#include "metrictree.h"
#include "treeprint.h"
//#include "parUtils.h"

using namespace std;
using namespace knn;
using namespace knn::repartition;
using namespace Torch;


void printMat(double *X, int numof_points, int dim)
{
	for(int i = 0; i < numof_points; i++) {
		for(int j = 0; j < dim; j++)
			cout<<X[i*dim+j]<<" ";
		//cout<<endl;
	}
	cout<<endl;
}


int main( int argc, char **argv)
{
	CmdLine cmd;
	const char *pchHelp = "This is hopeless.";
	cmd.addInfo(pchHelp);
	int numof_ref_points;
	cmd.addICmdOption("-nr", &numof_ref_points, 5000, "number of reference data points generated per proc (5000)");
	int numof_query_points;
	cmd.addICmdOption("-nq", &numof_query_points, 5000, "number of query points per proc (5000)");
	int dim;
	cmd.addICmdOption("-d", &dim, 8, "dimensionality of points generated per proc (8)");
	int max_tree_level;
	cmd.addICmdOption("-mtl", &max_tree_level, 10, "maximum tree depth (10)");
	int max_points_per_node;
	cmd.addICmdOption("-mppn", &max_points_per_node, 5000, "maximum number of points per tree node (5000)");
	int min_comm_size_per_node;
	cmd.addICmdOption("-mcsptn", &min_comm_size_per_node, 1, "min comm size per tree node (1)");
	int cf;
	cmd.addICmdOption("-cf", &cf, 1, "Clustering factor (1)");
	int k;
	cmd.addICmdOption("-k", &k, 2, "k for k nearest neighbors");
	cmd.read(argc, argv);

	int rank, nproc;
	MPI_Comm comm = MPI_COMM_WORLD;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);

	MetricData ref;
	ref.X.resize(dim*numof_ref_points);  // points
	ref.gids.resize(numof_ref_points);       // global ids
	ref.dim = dim;

	MetricData query;
	query.X.resize(dim*numof_query_points);
	query.gids.resize(numof_query_points);
	query.dim = dim;
	
	if(rank == 0) cout<<"Distribution: Uniform"<<endl;
	generateUniform(numof_ref_points, dim, &(ref.X[0]), comm);
	generateUniform(numof_query_points, dim, &(query.X[0]), comm);
	for(int i = 0; i < numof_ref_points; i++)
		ref.gids[i] = rank*numof_ref_points+i;
	for(int i = 0; i < numof_query_points; i++)
		query.gids[i] = rank*numof_query_points+i;

	//cout<<"rank: (ref)"<<rank<<" ";
	//printMat(&(ref.X[0]), numof_ref_points, dim);
	//MPI_Barrier(comm);

	//cout<<"rank: (query)"<<rank<<" ";
	//printMat(&(query.X[0]), numof_query_points, dim);
	//MPI_Barrier(comm);


	//double *proj = new double [dim];
	//getProjection(&(ref.X[0]), numof_ref_points, dim, proj, comm);
	//delete [] proj;


	double start_t;
	// ========== build a tree ==========
	MetricNode root;
	root.options.pruning_verbose=true;
	int numof_kids = 2;
	root.Insert( NULL,
		 max_points_per_node,
		 max_tree_level,
		 min_comm_size_per_node,
		 comm,
		 &ref,
		 0);
	MPI_Barrier(MPI_COMM_WORLD);

	vector< pair<double, long> > MTkNN;
	vector<long> *queryOutIDs = new vector<long>();
	MTqueryK(&query, numof_query_points*nproc, 16.0, &root, k, queryOutIDs, &MTkNN);



	MPI_Finalize();


	return 0;
}
	


	
	


