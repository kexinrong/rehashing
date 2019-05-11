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
#include "forest.h"

using namespace Torch;
using namespace std;


int main(int argc, char **argv) {

	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
  
	CmdLine cmd;
	const char *phelp = "Help";
    cmd.addInfo(phelp);

	char *ptrInputFile = NULL;
    cmd.addSCmdOption("-file", &ptrInputFile, "data.bin", "input binary file storing points");
    
	int dim;
	cmd.addICmdOption("-d", &dim, 4, "Dimension of the space (default = 4).");
    
	int rn;
	cmd.addICmdOption("-rn", &rn, 1000, "Number of referrence data points per process (default = 1000).");
	
	int qn;
	cmd.addICmdOption("-qn", &qn, 1000, "Number of query data points per process (default = 1000).");
    
	int k;
	cmd.addICmdOption("-k", &k, 2, "Number of nearest neighbors to find (2).");
    
	int gen;
	cmd.addICmdOption("-gen", &gen, 0, "Data generator (0). 0=uniform, 1=user input");
    
	int max_tree_level;
	cmd.addICmdOption("-mtl", &max_tree_level, 10, "maximum tree depth (10)");
    
	int max_points_per_node;
	cmd.addICmdOption("-mppn", &max_points_per_node, -1, "maximum number of points per tree node ( = numof_ref_points)");
    
	int min_comm_size_per_node;
	cmd.addICmdOption("-mcsptn", &min_comm_size_per_node, 1, "min comm size per tree node (1)");
   	
	int numiters;
	cmd.addICmdOption("-iter", &numiters, 5, "iterations of random rotation for greedy tree traverse (5)");
 
	int flag_tree_in_memory;
	cmd.addICmdOption("-fim", &flag_tree_in_memory, 1, "tree in memory (*1*) or not (0)");
 
	cmd.read(argc, argv);

    long numof_ref_points = rn;
    long numof_query_points = qn;
	if(max_points_per_node == -1) max_points_per_node = numof_ref_points;

	// ----------- parsing cmd complete! ------------------
	double *ref, *query;
	long *refids, *queryids;
	long nglobal, mglobal;
	long refid_offset, queryid_offset;
	ref = new double[numof_ref_points*dim];
	query = new double[numof_query_points*dim];

	double start_t, end_t;

	srand((unsigned)time(NULL)*rank);
    if(gen == 0) {
		if(rank == 0) cout << "Distribution: Uniform" << endl;
		generateUniform(numof_ref_points, dim, ref, MPI_COMM_WORLD);

        // make query the same on each rank
		generateUniform(numof_query_points, dim, query, MPI_COMM_WORLD);
        MPI_Bcast(query, numof_query_points*dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    else {
		string strInputFile = ptrInputFile;
		if(rank == 0) cout << "User Input File: " << strInputFile << endl;
		double *dummy_points;
		int total_numof_input_points = numof_ref_points*size;
        knn::parBinRead(strInputFile.c_str(), total_numof_input_points, dim, dummy_points, MPI_COMM_WORLD, 0);
		for(int i = 0; i < total_numof_input_points*dim; i++) ref[i] = dummy_points[i] / 1.0;
		delete [] dummy_points;
    }
	if(rank == 0) cout << "generate points done!" << endl;

	MPI_Allreduce( &numof_ref_points, &nglobal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
	MPI_Scan(&numof_ref_points, &refid_offset, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
	refid_offset -= numof_ref_points;
	refids = new long[numof_ref_points];
	for(int i = 0; i < numof_ref_points; i++) refids[i] = refid_offset + (long)i;

	//MPI_Allreduce( &numof_query_points, &mglobal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
	//MPI_Scan(&numof_query_points, &queryid_offset, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
	//queryid_offset -= numof_query_points;
	queryid_offset = 0;
	queryids = new long[numof_query_points];
	for(int i = 0; i < numof_query_points; i++) queryids[i] = queryid_offset + (long)i;

    // output ref points
    /*for(int r = 0; r < size; r++) {
        if(r == rank) {
	        for(int i = 0; i < numof_ref_points; i++) {
		        cout<<refids[i]<<" : ";
		        for(int j = 0; j < dim; j++)
			        cout<<ref[i*dim+j]<<" ";
                cout<<endl;
	        }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }*/

    // output query points
    /*for(int r = 0; r < size; r++) {
        if(r == rank) {
   	        for(int i = 0; i < numof_query_points; i++) {
		        cout<<queryids[i]<<" : ";
		        for(int j = 0; j < dim; j++)
			        cout<<query[i*dim+j]<<" ";
	        	cout<<endl;
	        }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }*/

	// 1. copy data
	binData *refData = new binData();
	refData->X.resize(dim*numof_ref_points);
	refData->gids.resize(numof_ref_points);
	refData->dim = dim;
	refData->numof_points = numof_ref_points;
	#pragma omp parallel for
	for(int i = 0; i < numof_ref_points*dim; i++) refData->X[i] = ref[i];
	#pragma omp parallel for
	for(int i = 0; i < numof_ref_points; i++) refData->gids[i] = refids[i];

	binData *queryData = new binData();
	queryData->X.resize(dim*numof_query_points);
	queryData->gids.resize(numof_query_points);
	queryData->dim = dim;
	queryData->numof_points = numof_query_points;
	#pragma omp parallel for
	for(int i = 0; i < numof_query_points*dim; i++) queryData->X[i] = query[i];
	#pragma omp parallel for
	for(int i = 0; i < numof_query_points; i++) queryData->gids[i] = queryids[i];

    start_t = omp_get_wtime();
    forest *rkdf = new forest();
    rkdf->setLeafSize(max_points_per_node);
    rkdf->setHeight(max_tree_level);
    rkdf->setCommSize(min_comm_size_per_node);
    rkdf->setRotateDataBack(true);
    rkdf->plant(refData, numiters);
    cout<<"rank "<<rank<<" datapool X.size = "<<rkdf->datapool->X.size()
        <<" id size = "<<rkdf->datapool->gids.size()
        <<" plant tree time: "<<omp_get_wtime()-start_t<<endl;

    delete refData;


    // output tree info
    /*for(int r = 0; r < size; r++) {
        if(r == rank) {
            for(int t = 0; t < rkdf->trees.size(); t++) {
                cout<<"rank"<<rank<<"==tree"<<t<<endl;
                if(!flag_tree_in_memory) {
                    string filename = "data--rank"+bxitoa(rank)+"--tree"+bxitoa(t)+".bin";
                    cout<<filename<<endl;
                    double *dummy_X;
                    int dummy_n = INT_MAX;
                    int dummy_dim;
                    knn::seqBinRead(filename.c_str(), 0, dummy_n, dummy_dim, dummy_X);
                    if(dummy_n != rkdf->datapool[t]->gids.size()) cout<<"unmatch found!"<<endl;
                    for(int i = 0; i < rkdf->datapool[t]->gids.size(); i++) {
                        cout<<rkdf->datapool[t]->gids[i]<<" : ";
                        for(int j = 0; j < dim; j++)
                            cout<<dummy_X[i*dim+j]<<" ";
                        cout<<endl;
                    }
                }
                else {
                    for(int i = 0; i < rkdf->datapool[t]->gids.size(); i++) {
                        cout<<rkdf->datapool[t]->gids[i]<<" : ";
                        for(int j = 0; j < dim; j++)
                            cout<<rkdf->datapool[t]->X[i*dim+j]<<" ";
                        cout<<endl;
                    }
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }*/


    start_t = omp_get_wtime();
    vector< pair<double, long> * > knn;
    vector<int> ksize;
    knn.resize(queryData->numof_points);
    for(int i = 0; i < queryData->numof_points; i++)
        knn[i] = new pair<double, long> [k];
    rkdf->find_knn(queryData, k, knn, ksize);
    cout<<"rank "<<rank<<" find_knn time "<<omp_get_wtime()-start_t<<endl;

    delete rkdf;


    /*
    for(int r = 0; r < size; r++) {
        if(rank == r) {
            for(int i = 0; i < ksize.size(); i++) {
                cout<<"(result) ID "<<queryData->gids[i]<<": local k = "<<ksize[i]<<" ";
                for(int j = 0; j < ksize[i]; j++) {
                    cout<<"("<<knn[i][j].second
                        <<","<<knn[i][j].first<<")  ";
                }
                cout<<endl;
            }
            cout<<endl;
        }
        cout.flush();
        MPI_Barrier(comm);
    }
    delete queryData;

    //for(int i = 0; i < knn.size(); i++) {
    //    delete [] knn[i];
    //    knn[i] = NULL;
    //}

    if(rank == 0) cout<<endl<<endl;


    // make query the same on each rank
	//generateUniform(numof_query_points, dim, query, MPI_COMM_WORLD);
    //MPI_Bcast(query, numof_query_points*dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	queryData = new binData();
	queryData->X.resize(dim*numof_query_points);
	queryData->gids.resize(numof_query_points);
	queryData->dim = dim;
	queryData->numof_points = numof_query_points;
	#pragma omp parallel for
	for(int i = 0; i < numof_query_points*dim; i++) queryData->X[i] = query[i];
	#pragma omp parallel for
	for(int i = 0; i < numof_query_points; i++) queryData->gids[i] = (long)i;

    vector< pair<double, long> > *all_knn = new vector< pair<double, long> >();
    rkdf->find_all_knn(queryData, k, all_knn);
    rkdf->seperate_knn(all_knn, k, knn, ksize);
    delete all_knn;

    //rkdf->find_knn(queryData, k, knn, ksize);


    for(int r = 0; r < size; r++) {
        if(rank == r) {
            for(int i = 0; i < ksize.size(); i++) {
                cout<<"(result) ID "<<queryData->gids[i]<<": local k = "<<ksize[i]<<" ";
                for(int j = 0; j < ksize[i]; j++) {
                    cout<<"("<<knn[i][j].second
                        <<","<<knn[i][j].first<<")  ";
                }
                cout<<endl;
            }
            cout<<endl;
        }
        cout.flush();
        MPI_Barrier(comm);
    }
    delete queryData;


    for(int i = 0; i < knn.size(); i++) {
        delete [] knn[i];
        knn[i] = NULL;
    }
    delete rkdf;
    */

    MPI_Finalize();
	return 0;

}


