#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <assert.h>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <fstream>

#include "file_io.h"
#include "repartition.h"
#include "CmdLine.h"
#include "generator.h"
#include "clustering.h"
#include "direct_knn.h"
#include "mpitree.h"
#include "treeprint.h"



using namespace std;
using namespace knn;
using namespace knn::repartition;
using namespace Torch;

struct TestOptions{
	int numof_points;
	int max_tree_level;
	int max_points_per_node;
	int min_comm_size_per_tree_node;
	int dim;
	double range;
	int num_query_per_process;
	int cf;
	int dbg_out;
        int k;
	int seedType;
};



void ParseInput(CmdLine &cmd, TestOptions &o, int argc, char **argv)
{
	const char *pchHelp = "This is hopeless.";
	cmd.addInfo(pchHelp);
	cmd.addICmdOption("-cp", &o.numof_points, 5000, "number of data points generated per proc (5000)");
	cmd.addICmdOption("-gd", &o.dim, 8, "dimensionality of points generated per proc (8)");
	cmd.addICmdOption("-s", &o.seedType, 1, "0:random seeds, 1:ostrovskey seeds (1)");
	cmd.addICmdOption("-k", &o.k, 8, "Number of near neighbors to find (8)");
	cmd.addICmdOption("-mtl", &o.max_tree_level, 4, "maximum tree depth");
	cmd.addICmdOption("-mppn", &o.max_points_per_node, 10, "maximum number of points per tree node");
	cmd.addICmdOption("-mcsptn", &o.min_comm_size_per_tree_node, 1, "min comm size per tree node");
	cmd.addICmdOption("-cf", &o.cf, 1, "Clustering factor (1)");
	cmd.addICmdOption("-dbg", &o.dbg_out, 0, "Enable debugging output (0)");
	cmd.read(argc, argv);
}


ostream& printPt( double* pt, int dim ){
	for( int i = 0; i < dim; i++ )
		cout << pt[i] << " ";

	return cout;
}

ostream& printPt( int* pt, int dim ){
	for( int i = 0; i < dim; i++ )
		cout << pt[i] << " ";

	return cout;
}

void printLeaves(MTNode &node) {
	if( node.kid == NULL ) {  //This is a leaf.
		int rank, size;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &size);
		
		cout << "Rank " << rank << "\n" <<
		"Dim = " << (*(node.data)).dim << "\n" <<
		" cluster center: (" << printPt(&(node.C[0]), (*(node.data)).dim) 
		<< ")\n"  <<
		" cluster radius: " << node.R[0] << "\n" << 
		" point ids: ";

		for(int i = 0; i < (*(node.data)).gids.size(); i++ )
			cout << (*(node.data)).gids[i] << " ";

		cout << "\n--------------------------------------------------------";
		cout << endl;

		return;
	} else {
		printLeaves(*(node.kid));
	}

}


int main( int argc, char **argv)
{
	CmdLine cmd;
	TestOptions o;
	ParseInput(cmd, o, argc, argv);

	int dim = o.dim;
	int numof_points = o.numof_points;
	int k = o.k;

	int        rank, nproc, mpi_namelen;
	char       mpi_name[MPI_MAX_PROCESSOR_NAME];
	double     start_time, end_time;
	MPI_Status status;
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Get_processor_name(mpi_name,&mpi_namelen);
	MPI_Barrier(MPI_COMM_WORLD);

	MTData data;
	data.X.resize(o.dim*o.numof_points);  // points
	data.gids.resize(o.numof_points);       // global ids
	data.dim = dim;

	MTData query;
	query.X.resize(o.dim*o.numof_points);
	query.gids.resize(o.numof_points);
	query.dim = dim;
	
/*
	int *tmp_ids = new int [numof_points];
	genPointInRandomLine(numof_points, dim, &data.X[0], tmp_ids, comm, true, rank*numof_points);
	for(int i = 0; i < numof_points*dim; i++)  data.X[i]*=1.0e9;
	PRINTSELF( cout << "Number of points " <<  numof_points << endl );
	for(int i = 0; i < numof_points; i++) {	data.gids[i] = long(tmp_ids[i]); } delete [] tmp_ids;
*/

	//Generate points uniformly distributed on the surface of the unit hypersphere  
        generateUnitHypersphere(o.numof_points, dim, &(data.X[0]), MPI_COMM_WORLD);
        #pragma omp parallel for
        for(int i = 0; i < o.numof_points; i++) data.gids[i] = i + numof_points*rank;

/*
	int *labels = new int[numof_points];
	generateMixOfUnitGaussian( o.numof_points, dim,
                                8, 16,
                                &(data.X[0]), labels, MPI_COMM_WORLD);
        #pragma omp parallel for
        for(int i = 0; i < o.numof_points; i++) data.gids[i] = i + numof_points*rank;
*/

        #pragma omp parallel for
	for(int i = 0; i < o.numof_points*dim; i++) 
		query.X[i] = data.X[i];

	#pragma omp parallel for
	for(int i = 0; i < o.numof_points; i++)
		query.gids[i] = data.gids[i];
	

	// ------------- plant a tree --------------------
	MTNode root;
        root.options.debug_verbose=o.dbg_out;
	root.options.pruning_verbose=true;
	int numof_kids = 2;
	double time = MPI_Wtime();
	root.Insert( NULL,
		 o.max_points_per_node,
		 o.max_tree_level,
		 o.min_comm_size_per_tree_node,
		 comm,
		 &data, 
		 o.seedType);
	MPI_Barrier(MPI_COMM_WORLD);
	if( rank == 0 ) cout << "construction: " << MPI_Wtime() - time << endl;	

	// ------------- query ----------------
	vector< pair<double, long> > *neighbors = NULL;
	double querytime = MPI_Wtime();
	vector< pair<double, long> > kNN;
	vector<long> *queryIDs;
        queryIDs = new vector<long>();
	queryK( &query, numof_points*nproc, 64.0 , &root, k, queryIDs,  &kNN);
        if(rank == 0) cout << "query: " << MPI_Wtime() - querytime << endl;
	
	MPI_Finalize();
	return 0;
}
	


	
	


