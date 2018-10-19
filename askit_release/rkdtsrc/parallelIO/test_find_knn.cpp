#include<mpi.h>
#include<direct_knn.h>
#include<knnreduce.h>
#include<lsh.h>
#include<generator.h>
#include<vector>
#include<cassert>
#include<cmath>
#include<utility>
#include<omp.h>
#include<CmdLine.h>
#include <ompUtils.h>
#include <ctime>
#include <climits>

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

void printknn(vector<long> &queryIDs_rkdt, vector< pair<double, long> > *kNN_rkdt, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int k = kNN_rkdt->size() / queryIDs_rkdt.size();
    for(int r = 0; r < size; r++) {
        if(rank == r) {
            for(int i = 0; i < queryIDs_rkdt.size(); i++) {
                cout<<"(rank "<<rank<<") "<<queryIDs_rkdt[i]<<": ";
                for(int j = 0; j < k; j++) {
                    cout<<"("<<(*kNN_rkdt)[i*k+j].second<<","<<(*kNN_rkdt)[i*k+j].first<<")  ";
                }
                cout<<endl;
            }
            cout.flush();
        }
        MPI_Barrier(comm);
    }
    if(rank == 0) cout<<endl;
}


int main(int argc, char **argv) {

	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	CmdLine cmd;
	const char *phelp = "Help";
    cmd.addInfo(phelp);

	char *ptrInputRefFile = NULL;
    cmd.addSCmdOption("-ref_file", &ptrInputRefFile, "ref.txt", "input file storing reference points (ascii)");

	char *ptrInputQueryFile = NULL;
    cmd.addSCmdOption("-query_file", &ptrInputQueryFile, "query.txt", "input file storing query points (ascii)");

    char *ptrOutputFile = NULL;
    cmd.addSCmdOption("-knn_file", &ptrOutputFile, "knn_results.txt", "output ascii file storing knn results");

    bool isBinary;
    cmd.addBCmdOption("-binary", &isBinary, false, "use binary file");

    bool bl_all2all;
    cmd.addBCmdOption("-search_all2all", &bl_all2all, false, "if query and reference points are the same, specify -search_all2all");

    long glb_nref;
	cmd.addLCmdOption("-glb_nref", &glb_nref, 1000, "global number of referrence data points to read (default = 1000).");

    long glb_nquery;
	cmd.addLCmdOption("-glb_nquery", &glb_nquery, 1000, "global number of referrence data points to read (default = 1000).");

	int dim;
	cmd.addICmdOption("-dim", &dim, -1, "Dimensionality of data (default = -1).");

	int k;
	cmd.addICmdOption("-k", &k, 2, "Number of nearest neighbors to find (2).");

	int max_tree_level;
	cmd.addICmdOption("-mtl", &max_tree_level, 10, "maximum tree depth (10)");

	int max_points_per_node;
	cmd.addICmdOption("-mppn", &max_points_per_node, -1, "maximum number of points per tree node ( = numof_ref_points)");

	int numiters;
	cmd.addICmdOption("-iter", &numiters, 5, "iterations of random rotation for greedy tree traverse (5)");

    bool bl_eval;
    cmd.addBCmdOption("-eval", &bl_eval, false, "evaluate results using log(nquery) sampled points");

    bool bl_disp;
    cmd.addBCmdOption("-disp", &bl_disp, false, "read knn and display it");

	cmd.read(argc, argv);

	if(rank == 0) cout<<"nproc = "<<size<<endl;

    long divd = glb_nref / size;
    long rem = glb_nref % size;
    long numof_ref_points = rank < rem ? (divd+1) : divd;

    divd = glb_nquery / size;
    rem = glb_nquery % size;
    long numof_query_points = rank < rem ? (divd+1) : divd;

    cout<<"rank "<<rank<<": numof_ref_points = "<<numof_ref_points<<endl;
    if(!bl_all2all) cout<<"rank "<<rank<<": numof_query_points = "<<numof_query_points<<endl;

    int splitter_type = 0;
	int tree_dbg_out = 0;
	int tree_time_out = 0;
	int min_comm_size_per_node = 1;
	int flops_verbose = 0;
	int tree_rotation_type = 1;
	int tree_coord_type = 0;
	int super_charging_verbose = 0;
	int hypertree_verbose = 1;

    int eval_verbose = bl_eval;
	if(max_points_per_node == -1) max_points_per_node = 2*k;

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

	// ----------- parsing cmd complete! ------------------

	double start_t, end_t;
	double max_t, min_t, avg_t;

	srand((unsigned)time(NULL)*rank);

    // ---------- read data ------------------
	double *ref = NULL, *query = NULL;
	long *refids = NULL, *queryids = NULL;
	long nglobal, mglobal;
	long refid_offset, queryid_offset;

	long glb_numof_ref_points = glb_nref;
    if(isBinary) {
        int dummy_numof_points;
        knn::mpi_binread(ptrInputRefFile, glb_numof_ref_points, dim, dummy_numof_points, ref, comm);
        numof_ref_points = dummy_numof_points;
    }
    else {
        knn::mpi_dlmread(ptrInputRefFile, glb_numof_ref_points, dim, ref, comm, false);
        numof_ref_points = glb_numof_ref_points;
    }
    if(rank == 0) cout << "read ref input: " << ptrInputRefFile << endl;

	MPI_Allreduce( &numof_ref_points, &nglobal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
	MPI_Scan(&numof_ref_points, &refid_offset, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
	refid_offset -= numof_ref_points;
	refids = new long[numof_ref_points];
	for(int i = 0; i < numof_ref_points; i++) refids[i] = refid_offset + (long)i;

    binData refData;
	refData.X.resize((long)dim*(long)numof_ref_points);
	refData.gids.resize(numof_ref_points);
	refData.dim = dim;
	refData.numof_points = numof_ref_points;
	#pragma omp parallel for
	for(long i = 0; i < (long)numof_ref_points*(long)dim; i++) refData.X[i] = ref[i];
	#pragma omp parallel for
	for(int i = 0; i < numof_ref_points; i++) refData.gids[i] = refids[i];
    delete[] ref;
	delete[] refids;

	binData queryData;
    if(!bl_all2all) {
        long glb_numof_query_points = glb_nquery;
        if(isBinary) {
            int dummy_numof_points;
            knn::mpi_binread(ptrInputQueryFile, glb_numof_query_points, dim,
                                dummy_numof_points, query, comm);
            numof_query_points = dummy_numof_points;
        }
        else {
            knn::mpi_dlmread(ptrInputQueryFile, glb_numof_query_points, dim, query, comm, false);
            numof_query_points = glb_numof_query_points;
        }
        if(rank == 0) cout << "read query input: " << ptrInputQueryFile << endl;

        MPI_Allreduce( &numof_query_points, &mglobal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
		MPI_Scan(&numof_query_points, &queryid_offset, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
		queryid_offset -= numof_query_points;
		queryids = new long[numof_query_points];
		for(int i = 0; i < numof_query_points; i++) queryids[i] = queryid_offset + (long)i;

		queryData.X.resize((long)dim*(long)numof_query_points);
		queryData.gids.resize(numof_query_points);
		queryData.dim = dim;
		queryData.numof_points = numof_query_points;
		#pragma omp parallel for
		for(long i = 0; i < (long)numof_query_points*(long)dim; i++) queryData.X[i] = query[i];
		#pragma omp parallel for
		for(int i = 0; i < numof_query_points; i++) queryData.gids[i] = queryids[i];
		delete[] query;
		delete[] queryids;
    }


    /*
    for(int r = 0; r < size; r++) {
        if(r == rank) {
            for(int i = 0; i < refData.numof_points; i++) {
                cout<<"("<<rank<<") "<<refData.gids[i]<<": ";
                for(int j = 0; j < refData.dim; j++) {
                    cout<<refData.X[i*refData.dim+j]<<" ";
                }
                cout<<endl;
            }
            cout.flush();
        }
        MPI_Barrier(comm);
    }*/


    MPI_Barrier(comm);
    if(rank == 0) cout<<endl<<"start find knn ... "<<endl;
    // . find knn
    vector<long> queryIDs_rkdt;
	vector< pair<double, long> > *kNN_rkdt = new vector< pair<double, long> >();

    clock_t t1 = clock();
    if(bl_all2all) {		// use all to all special case
		bintree::knnTreeSearch_RandomRotation_a2a(&refData, k,
								numiters, params, tree_rotation_type, tree_coord_type,
								queryIDs_rkdt, kNN_rkdt);
	}
    else {
		bintree::knnTreeSearch_RandomRotation(&refData, &queryData, k,
									numiters, params, tree_rotation_type, tree_coord_type,
									queryIDs_rkdt, kNN_rkdt);
	}
    clock_t t2 = clock();
    std::cout << "KNN took: " << (float)(t2 - t1)/CLOCKS_PER_SEC << std::endl;

    if(bl_disp) {
        if(rank == 0) cout<<"knn results by rkdt: "<<endl;
        printknn(queryIDs_rkdt, kNN_rkdt, comm);
    }


    if(isBinary) {
        knn::mpi_binwrite_knn( ptrOutputFile, queryIDs_rkdt, kNN_rkdt, comm );
    }
    else {
        knn::mpi_dlmwrite_knn( ptrOutputFile, queryIDs_rkdt, kNN_rkdt, comm );
    }

    if(bl_disp) {
        vector< pair<double, long> > *knn = new vector< pair<double, long> >();
        if(isBinary) {
            knn::binread_knn(ptrOutputFile, queryIDs_rkdt, k-1, knn);
        } else {
            knn::dlmread_knn(ptrOutputFile, queryIDs_rkdt, k-1, knn);
        }
        if(rank == 0) cout<<"read knn results read from binary: "<<endl;
        printknn(queryIDs_rkdt, knn, comm);
        delete knn;
    }

	delete kNN_rkdt;

	MPI_Finalize();

	return 0;

}


