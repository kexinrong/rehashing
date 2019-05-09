#include<direct_knn.h>
#include<knnreduce.h>
#include<lsh.h>
#include<generator.h>
#include<vector>
#include<cassert>
#include<cmath>
#include<utility>
#include<omp.h>
#include<mpi.h>
#include<CmdLine.h>
#include <ompUtils.h>
#include <ctime>

#include "mpitree.h"
#include "clustering.h"
#include "repartition.h"
#include "binTree.h"
#include "binQuery.h"
#include "distributeToLeaf.h"
#include "parallelIO.h"
#include "eval.h"
#include "papi_perf.h"
#include "verbose.h"

using namespace Torch;
using namespace std;


int main(int argc, char **argv) {

	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if(rank == 0) {cout<<"type ex1a2a -help to see options"<<endl;}

	if(rank == 0) {cout<<"tree starts with "<<size<<" MPI ranks"<<endl;}	

  CmdLine cmd;
	const char *phelp = "Help";
	cmd.addInfo(phelp);

	char *ptrInputFile = NULL;
	cmd.addSCmdOption("-file", &ptrInputFile, "data.bin", "input binary file storing points");

	int dim;
	cmd.addICmdOption("-d", &dim, 4, "Dimension of the space (default = 4).");

	int intrinsicDim;
	cmd.addICmdOption("-id", &intrinsicDim, 2, "Intrinsic dimension of the embedding data (default = 2)");

	int rn;
	cmd.addICmdOption("-rn", &rn, 10000, "Number of referrence data points per process (default = 10000).");

	int k=4;
	cmd.addICmdOption("-k", &k, 4, "Number of nearest neighbors to find (3).");

	int tree_dbg_out;
	cmd.addICmdOption("-dbg", &tree_dbg_out, 0, "Enable tree debugging output (0)");

    
	int numiters;
	cmd.addICmdOption("-iter", &numiters, 5, "iterations of random rotation for greedy tree traverse (5)");
	
	int eval_verbose;
	cmd.addICmdOption("-eval", &eval_verbose, 0, "Evaluate results? (0). 1: check uisng logm sampled points");

	int max_points_per_node;
	cmd.addICmdOption("-mppn", &max_points_per_node, 2*k, "maximum number of points per tree node ( = 2*k)");


	int tree_time_out =0;
	int max_tree_level=15;
	int splitter_type=0;
	int min_comm_size_per_node =1;    
	int flops_verbose=0;
	int tree_rotation_type=0;
	int tree_coord_type =0;
	int super_charging_verbose = 0;
	int a2a_verbose = 1;
	int traverse_type = 0;
	int hypertree_verbose = 1;

	cmd.read(argc, argv);

	long numof_ref_points = rn;
	long numof_query_points = rn;

	treeParams params;
	params.hypertree = hypertree_verbose;
	params.splitter = splitter_type;
	params.debug_verbose = tree_dbg_out;
	params.timing_verbose = tree_time_out;
	params.pruning_verbose = 1;
	params.max_points_per_node = max_points_per_node;
	params.max_tree_level = max_tree_level;
	params.min_comm_size_per_node = min_comm_size_per_node;
	params.flops_verbose = flops_verbose;
	params.eval_verbose = eval_verbose;
	params.traverse_type = traverse_type;
	
	
	// ----------- parsing cmd complete! ------------------

	double *ref, *query;
	long *refids, *queryids;
	long nglobal, mglobal;
	long refid_offset, queryid_offset;

	double start_t, end_t;
	double max_t, min_t, avg_t;

	//	srand((unsigned)time(NULL)*rank);
	
	if(rank == 0) cout << "Distribution: "<<intrinsicDim<<"-d Unit Gaussian Embedding into "<<dim<<"-d space" << endl;
	ref = new double[numof_ref_points*dim];
	generateUniformEmbedding(numof_ref_points, dim, intrinsicDim, ref, MPI_COMM_WORLD);
	if(rank == 0) cout << "generate points done!" << endl;
	
	MPI_Allreduce( &numof_ref_points, &nglobal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
	MPI_Scan(&numof_ref_points, &refid_offset, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
	refid_offset -= numof_ref_points;
	refids = new long[numof_ref_points];
	for(int i = 0; i < numof_ref_points; i++) refids[i] = refid_offset + (long)i;

	numof_query_points = numof_ref_points;
	query = ref;
	queryids = refids;

	// knn tree search
	vector<long> sampleIDs;
	vector<double> globalKdist;
	vector<long> globalKid;
	if(eval_verbose && super_charging_verbose) {
		get_sample_info(ref, query, refids, queryids, numof_ref_points, numof_query_points, dim, k,
										sampleIDs, globalKdist, globalKid);
	}

	if(rank == 0) cout<<"get verify info done."<<endl;

	float mflops = -1.0, papi_real_time, papi_proc_time;
	long long flpins = 0;
	float max_mflops = -1.0, min_mflops = -1.0, avg_mflops = -1.0;

	if(rank == 0) cout<<endl<<"*************** total iters = "<<numiters<<" ***************** "<<endl;

	// 1. copy data
	binData refData;
	refData.X.resize(dim*numof_ref_points);
	refData.gids.resize(numof_ref_points);
	refData.dim = dim;
	refData.numof_points = numof_ref_points;
    #pragma omp parallel for
	for(int i = 0; i < numof_ref_points*dim; i++) refData.X[i] = ref[i];
    #pragma omp parallel for
	for(int i = 0; i < numof_ref_points; i++) refData.gids[i] = refids[i];
	delete[] ref;
	delete[] refids;


	// 2. search
	vector<long> queryIDs_rkdt;
	vector< pair<double, long> > *kNN_rkdt = new vector< pair<double, long> >();

	start_t = MPI_Wtime();

	bintree::knnTreeSearch_RandomRotation_a2a(&refData, k,
												numiters, params, tree_rotation_type, tree_coord_type,
												queryIDs_rkdt, kNN_rkdt);
	end_t = MPI_Wtime() - start_t;

	printf("\n\t RANK[%2d]: Search took %f seconds\n", rank,end_t);
	fflush(stdout);

	delete kNN_rkdt;

	MPI_Finalize();
	return 0;

}




