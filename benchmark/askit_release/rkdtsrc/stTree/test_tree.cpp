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

using namespace Torch;
using namespace std;


int main(int argc, char **argv) {
   
	CmdLine cmd;
	const char *phelp = "Help";
    cmd.addInfo(phelp);
    
	int dim;
	cmd.addICmdOption("-d", &dim, 4, "Dimension of the space (default = 4).");
    
	int rn;
	cmd.addICmdOption("-rn", &rn, 1000, "Number of referrence data points per process (default = 1000).");
	
	int qn;
	cmd.addICmdOption("-qn", &qn, 1000, "Number of query data points per process (default = 1000).");
    
	int k;
	cmd.addICmdOption("-k", &k, 2, "Number of nearest neighbors to find (2).");
    
	int gen;
	cmd.addICmdOption("-gen", &gen, 0, "Data generator (0). 0=uniform, 1=hypersphere, 2=unit guassian, 3=mix of gaussians, 4=line");
    
	int tree_dbg_out;
	cmd.addICmdOption("-dbg", &tree_dbg_out, 0, "Enable tree debugging output (0)");
    
	int tree_time_out;
	cmd.addICmdOption("-time", &tree_time_out, 0, "Enable tree timing output (0)");
    
	int max_tree_level;
	cmd.addICmdOption("-mtl", &max_tree_level, 10, "maximum tree depth (10)");
    
	int max_points;
	cmd.addICmdOption("-mp", &max_points, -1, "maximum number of points per tree node ( = numof_ref_points)");
    
	int iter_tree;
	cmd.addICmdOption("-iter", &iter_tree, 5, "iterations of random rotation for greedy tree traverse (5)");
   	
	int tree_search_type;
	cmd.addICmdOption("-tsearch", &tree_search_type, 0, "tree search type (0). 0:random kd tree, 1: random kd tree (all2all), 2: both of them");
   	
	int tree_rotate_type;
	cmd.addICmdOption("-fr", &tree_rotate_type, 0, "tree rotation type (0). 0:do not ratate, 1: only rotate at root, 2: rotate at every level");
    
	cmd.read(argc, argv);
	
    long numof_ref_points = rn;
    long numof_query_points = qn;
	if(max_points == -1) max_points = numof_ref_points;

	// ----------- parsing cmd complete! ------------------

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

	double start_t, end_t;

	//srand((unsigned)time(NULL)*rank);
	srand(0);

	switch(gen) {
		case 0:
			if(rank == 0) cout << "Distribution: Uniform" << endl;
			generateUniform(numof_ref_points, dim, ref, MPI_COMM_WORLD);
			generateUniform(numof_query_points, dim, query, MPI_COMM_WORLD);
			break;

		case 1:
			if(rank == 0) cout << "Distribution: Hypersphere shell" << endl;
			generateUnitHypersphere(numof_ref_points, dim, ref, MPI_COMM_WORLD);
			generateUnitHypersphere(numof_query_points, dim, query, MPI_COMM_WORLD);
			break;

		case 2:
			if(rank == 0) cout << "Distribution: Unit gaussian" << endl;
			generateNormal(numof_ref_points, dim, ref, MPI_COMM_WORLD);
			generateNormal(numof_query_points, dim, query, MPI_COMM_WORLD);
			break;
     
		case 3:
		{
			if(rank == 0) cout << "Distribution: Mixture of random gaussians" << endl;
			assert(numof_ref_points%2==0);
			generateNormal(numof_ref_points, dim, ref, MPI_COMM_WORLD);
			#pragma omp parallel for
			for(int i = 0; i < (numof_ref_points/2)*dim; i++) ref[i] *= 2.0;
			#pragma omp parallel for
			for(int i = numof_ref_points/2; i < numof_ref_points; i++) ref[i*dim] += 1.0;
			
			assert(numof_query_points%2==0);
			generateNormal(numof_query_points, dim, query, MPI_COMM_WORLD);
			#pragma omp parallel for
			for(int i = 0; i < (numof_query_points/2)*dim; i++) query[i] *= 2.0;
			#pragma omp parallel for
			for(int i = numof_query_points/2; i < numof_query_points; i++) query[i*dim] += 1.0;
			
			//int *dummy_rlabels = new int[numof_ref_points];
			//int *dummy_qlabels = new int[numof_query_points];
			//double var[4] = { 1.5, 1.0, 1.0, 1.0 };
			//generateMixOfUserGaussian( numof_ref_points, dim, 2, 0.5, var, ref, dummy_rlabels, MPI_COMM_WORLD );
			//generateMixOfUserGaussian( numof_query_points, dim, 2, 0.5, var, query, dummy_qlabels, MPI_COMM_WORLD );
			//delete[] dummy_rlabels;
			//delete[] dummy_qlabels;
			break;
		}
	 
		case 4:
		{
			if(rank == 0) cout << "Distribution: Line" << endl;
			int * dummy_refids = new int[numof_ref_points];
			int * dummy_queryids = new int[numof_query_points];
			genPointInRandomLine(numof_ref_points, dim, ref, dummy_refids, MPI_COMM_WORLD, 1, rank*numof_ref_points);
			genPointInRandomLine(numof_query_points, dim, query, dummy_queryids, MPI_COMM_WORLD, 1, rank*numof_query_points);
			delete [] dummy_refids;
			delete [] dummy_queryids;
			break;
		}
	 
		default:
		cerr << "Invalid generator selection" << endl;
		exit(1);
	}

    cout<<"generate points done"<<endl;


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
	//for(int i = 0; i < numof_query_points*dim; i++) query[i] = ref[i];
	//for(int i = 0; i < numof_query_points; i++) queryids[i] = refids[i];

	if(tree_dbg_out == 5) {
		for(int i = 0; i < numof_ref_points; i++) {
			cout<<refids[i]<<" ";
			for(int j = 0; j < dim; j++)
				cout<<ref[i*dim+j]<<" ";
			cout<<endl;
		}
	}

	int nthreads = omp_get_max_threads();
	cout<<"max threads: "<<nthreads<<endl;


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


	vector< pair<double, long> > *knn_rkdt = new vector< pair<double, long> >();
    if(tree_search_type == 0 || tree_search_type == 2) {
		start_t = omp_get_wtime();
		stTreeSearch_rkdt(refData, queryData, k, max_points, max_tree_level, iter_tree, tree_rotate_type, knn_rkdt);
		cout<<"rkdt tree search (standard) time: "<<omp_get_wtime() - start_t<<endl;
	}


	if(tree_search_type == 1 || tree_search_type == 2) {
		start_t = omp_get_wtime();
		stTreeSearch_rkdt_a2a(ref, query, refids, queryids, numof_ref_points, numof_query_points, dim, max_points, max_tree_level, k, iter_tree, tree_rotate_type, knn_rkdt);

        cout<<"numof_query_points = "<<numof_query_points<<endl;
        for(int i = 0; i < numof_query_points; i++) {
            cout<<"["<<queryids[i]<<"] ";
            for(int j = 0; j < k; j++) {
                cout<<"("<<(*knn_rkdt)[i*k+j].second<<", "<<(*knn_rkdt)[i*k+j].first<<") "<<endl;
            }
        }

		cout<<"rkdt tree search (all-to-all special) time: "<<omp_get_wtime() - start_t<<endl;
	}



	if(tree_dbg_out == 5) {
		for(int i = 0; i < numof_query_points; i++) {
			cout<<queryids[i]<<" : ";
			for(int j = 0; j < k; j++)
				cout<<(*knn_rkdt)[i*k+j].second<<"-"<<(*knn_rkdt)[i*k+j].first<<" ";
			cout<<endl;
		}
	}


	delete[] ref;
	delete[] query;
	delete[] refids;
	delete[] queryids;

   MPI_Finalize();

   return 0;
}



