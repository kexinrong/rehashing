#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <assert.h>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <fstream>

#include "CmdLine.h"
#include "generator.h"
#include "direct_knn.h"



using namespace std;
using namespace knn;
using namespace Torch;

struct TestOptions{
	int n;
	int m;
	int dim;
};



void ParseInput(CmdLine &cmd, TestOptions &o, int argc, char **argv)
{
	cmd.addICmdOption("-n", &o.n, 5000, "number of reference points generated per proc (5000)");
	cmd.addICmdOption("-m", &o.m, 5000, "number of query points generated per proc (5000)");
	cmd.addICmdOption("-d", &o.dim, 8, "dimensionality of points generated per proc (8)");
	cmd.read(argc, argv);
}



int main( int argc, char **argv)
{
        MPI_Init(&argc, &argv);
	CmdLine cmd;
	TestOptions o;
	ParseInput(cmd, o, argc, argv);

	int dim = o.dim;
	int n = o.n;
	int m = o.m;
        int k = 10;

        double *ref = new double[n*dim];
        double *query = new double[m*dim];
        long *refids = new long[n];
        long *queryids = new long[m];

	//Generate points uniformly distributed on the surface of the unit hypersphere  
        generateUnitHypersphere(o.n, dim, ref, MPI_COMM_WORLD);
        generateUnitHypersphere(o.m, dim, query, MPI_COMM_WORLD);
        for(int i = 0; i < o.n; i++) refids[i] = i;
	for(int i = 0; i < o.m; i++)
		queryids[i] = i;

 
        std::pair<double, long> *knnlowmem = new std::pair<double, long>[m*k];
/*
        double *dist = new double[KNN_MAX_BLOCK_SIZE*n];
        double* sqnormr = new double[n];
        double* sqnormq = new double[KNN_MAX_BLOCK_SIZE];
*/
	double start = omp_get_wtime();
	knn::directKQueryLowMem( ref, query, n, m, k, dim, knnlowmem /*, dist, sqnormr, sqnormq*/ ); 
	cout << omp_get_wtime() - start << endl;

	MPI_Finalize(); 
	return 0;
}
	


	
	


