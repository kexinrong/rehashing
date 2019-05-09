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
#include "stTree.h"
#include "stTreeSearch.h"
#include "verbose.h"

using namespace Torch;
using namespace std;


int main(int argc, char **argv) {

	CmdLine cmd;
	const char *phelp = "Help";
    cmd.addInfo(phelp);

	int gen;
	cmd.addICmdOption("-gen", &gen, 0, "Data generator (0). 0=uniform, 1=unit gaussian embedding");
    
	int dim;
	cmd.addICmdOption("-d", &dim, 4, "Dimension of the space (default = 4).");
	int intrinsicDim;
	cmd.addICmdOption("-id", &intrinsicDim, 2, "Intrinsic dimension of the embedding data (default = 2)");

	int rn;
	cmd.addICmdOption("-rn", &rn, 1000, "Number of referrence data points per process (default = 1000).");

	int qn;
	cmd.addICmdOption("-qn", &qn, 1000, "Number of query data points per process (default = 1000).");

	int k;
	cmd.addICmdOption("-k", &k, 2, "Number of nearest neighbors to find (2).");

	int tree_dbg_out;
	cmd.addICmdOption("-dbg", &tree_dbg_out, 0, "Enable tree debugging output (0)");

	int max_tree_level;
	cmd.addICmdOption("-mtl", &max_tree_level, 10, "maximum tree depth (10)");

	int max_points;
	cmd.addICmdOption("-mp", &max_points, -1, "maximum number of points per tree node ( = numof_ref_points)");

	int iter_tree;
	cmd.addICmdOption("-iter", &iter_tree, 5, "iterations of random rotation for greedy tree traverse (5)");
	int tree_rotate_type;
	cmd.addICmdOption("-fr", &tree_rotate_type, 1, "tree rotation type (1). 0:do not ratate, 1: only rotate at root, 2: rotate at every level");

	cmd.read(argc, argv);

    long numof_ref_points = rn;
    long numof_query_points = qn;
	if(max_points == -1) max_points = numof_ref_points;

	// ----------- parsing cmd complete! ------------------

	double start_t, end_t;
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	double *ref, *query;
	long *refids, *queryids;
	long nglobal, mglobal;
	long refid_offset, queryid_offset;
	ref = new double[numof_ref_points*dim];
	query = new double[numof_query_points*dim];

	srand((unsigned)time(NULL)*rank);

	switch(gen) {
		case 0:
        {
			cout << "Distribution: Uniform" << endl;
		    generateUniform(numof_ref_points, dim, ref, MPI_COMM_WORLD);
	        generateUniform(numof_query_points, dim, query, MPI_COMM_WORLD);
			break;
        }

		case 1:
		{
			if(rank == 0) cout << "Distribution: "<<intrinsicDim<<"-d Unit Gaussian Embedding into "<<dim<<"-d space" << endl;
			delete [] ref;
			int np = numof_ref_points + numof_query_points;
			ref = new double [np*dim];
			generateNormalEmbedding(np, dim, intrinsicDim, ref, MPI_COMM_WORLD);
			double *newref = new double [numof_ref_points*dim];
			#pragma omp parallel for
			for(int i = 0; i < dim*numof_ref_points; i++)
				newref[i] = ref[i];
			#pragma omp parallel for
			for(int i = 0; i < dim*numof_query_points; i++)
				query[i] = ref[dim*numof_ref_points+i];
			delete [] ref;
			ref = newref;

			break;
		}

		default:
		cerr << "Invalid generator selection" << endl;
		exit(1);
	}


	MPI_Allreduce( &numof_ref_points, &nglobal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce( &numof_query_points, &mglobal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
	MPI_Scan(&numof_ref_points, &refid_offset, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
	MPI_Scan(&numof_query_points, &queryid_offset, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
	refid_offset -= numof_ref_points;
	queryid_offset -= numof_query_points;
	refids = new long[numof_ref_points];
	queryids = new long[numof_query_points];
	for(int i = 0; i < numof_ref_points; i++) refids[i] = refid_offset + (long)i;
	for(int i = 0; i < numof_query_points; i++) queryids[i] = queryid_offset + (long)i;
    for(int i = 0; i < numof_query_points*dim; ++i) query[i] = ref[i];

    if(tree_dbg_out) {
        cout<<"====== ref ======"<<endl;
	    for(int i = 0; i < numof_ref_points; i++) {
		    cout<<refids[i]<<": ";
		    for(int j = 0; j < dim; j++)
			    cout<<ref[i*dim+j]<<" ";
		    cout<<endl;
	    }
        cout<<"====== query ======"<<endl;
	    for(int i = 0; i < numof_query_points; i++) {
		    cout<<queryids[i]<<": ";
	    for(int j = 0; j < dim; j++)
			    cout<<query[i*dim+j]<<" ";
		    cout<<endl;
	    }
    }

    pbinData queryData = new binData();
	queryData->X.resize(numof_query_points*dim);
	queryData->gids.resize(numof_query_points);
	queryData->dim = dim;
    queryData->numof_points = numof_query_points;
	#pragma omp parallel for
	for(int i = 0; i < numof_query_points; i++) {
        memcpy( &(queryData->X[i*dim]), query+i*dim, sizeof(double)*dim);
		queryData->gids[i] = queryids[i];
	}

	pbinData refData = new binData();
	refData->X.resize(numof_ref_points*dim);
	refData->gids.resize(numof_ref_points);
	refData->dim = dim;
    refData->numof_points = numof_ref_points;
	#pragma omp parallel for
	for(int i = 0; i < numof_ref_points; i++) {
        memcpy(&(refData->X[i*dim]), ref+i*dim, sizeof(double)*dim);
		refData->gids[i] = refids[i];
	}


	vector<long> sampleIDs;
	vector<double> globalKdist;
	vector<long> globalKid;
	get_sample_info(ref, query, refids, queryids, numof_ref_points, numof_query_points, dim, k,
					sampleIDs, globalKdist, globalKid);

    cout<<"sampleIDs.size = "<<sampleIDs.size()
        <<", gkdist.size = "<<globalKdist.size()
        <<", gkid.size = "<<globalKid.size()
        <<endl;


	vector< pair<double, long> > *knn_rkdt = NULL;
	double hit_rate = 0.0, relative_error = 0.0;
	int nmiss = 0;

    /*start_t = omp_get_wtime();
    stTreeSearch_rkdt( ref, query, refids, queryids, numof_ref_points, numof_query_points, dim,
					   max_points, max_tree_level, k, iter_tree, tree_rotate_type,
					   knn_rkdt);
    delete knn_rkdt;
	cout<<"old rkdt tree search (standard) time: "<<omp_get_wtime() - start_t<<endl;
    */

    /*start_t = omp_get_wtime();
	knn_rkdt = new vector< pair<double, long> >();
	stTreeSearch_rkdt(refData, queryData, k, max_points, max_tree_level, iter_tree, tree_rotate_type, true, knn_rkdt);
	cout<<"new rkdt tree search (standard) time: "<<omp_get_wtime() - start_t<<endl;
verify(sampleIDs, globalKdist, globalKid, queryData->gids, (*knn_rkdt), nmiss, hit_rate, relative_error);
	cout<<"- Hit Rate ("<<sampleIDs.size()<<" samples): "<< hit_rate << "%" << endl;
	cout<<"- Miss NN in error ("<<sampleIDs.size()<<" samples): "<< nmiss << endl;
	cout<<"- Relative Error ("<<sampleIDs.size()<<" samples): "<< relative_error << "%" << endl;
    delete knn_rkdt;
    */

    /*
    STree_Search_T_ = 0.0;
	start_t = omp_get_wtime();
	knn_rkdt = new vector< pair<double, long> >();
	find_knn_srkdt(refData, queryData, k, max_points, max_tree_level, iter_tree, knn_rkdt);

	cout<<"srkdt tree search (standard) time: "<<omp_get_wtime() - start_t<<endl;
	cout<<"direct search kernel: "<<STree_Search_T_<<endl;
	verify(sampleIDs, globalKdist, globalKid, queryData->gids, (*knn_rkdt), nmiss, hit_rate, relative_error);
	cout<<"- Hit Rate ("<<sampleIDs.size()<<" samples): "<< hit_rate << "%" << endl;
	cout<<"- Miss NN in error ("<<sampleIDs.size()<<" samples): "<< nmiss << endl;
	cout<<"- Relative Error ("<<sampleIDs.size()<<" samples): "<< relative_error << "%" << endl;
    */


	start_t = omp_get_wtime();
	knn_rkdt = new vector< pair<double, long> >();
	find_knn_srkdt_a2a(refData, k, max_points, max_tree_level, iter_tree, knn_rkdt);


    for(int i = 0; i < queryData->numof_points; i++) {
        cout<<queryData->gids[i]<<": ";
        for(int j = 0; j < k; j++) {
            cout<<"("<<(*knn_rkdt)[i*k+j].second<<", "<<(*knn_rkdt)[i*k+j].first<<") ";
        }
        cout<<endl;
    }


    delete knn_rkdt;

    delete[] ref;
	delete[] query;
	delete[] refids;
	delete[] queryids;
    delete refData;
    delete queryData;

    MPI_Finalize();

    return 0;
}



