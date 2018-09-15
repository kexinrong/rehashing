#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <assert.h>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <omp.h>

#include "CmdLine.h"
#include "generator.h"
#include "direct_knn.h"
#include "papi_perf.h"


using namespace std;
using namespace knn;
using namespace Torch;

struct TestOptions{
	int dim;
};



void ParseInput(CmdLine &cmd, TestOptions &o, int argc, char **argv)
{
	cmd.addICmdOption("-gd", &o.dim, 100, "dimensionality of points generated per proc (8)");
	cmd.read(argc, argv);
}



int main( int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	CmdLine cmd;
	TestOptions o;
	ParseInput(cmd, o, argc, argv);

	int dim = o.dim;
        int k = 2;

	init_papi();
	for(int n = 16; n <= 4096; n*=2) {
	        double *ref = new double[n*dim];
	        double *query = new double[n*dim];
	        long *refids = new long[n];
	        long *queryids = new long[n];
	
		//Generate points uniformly distributed on the surface of the unit hypersphere  
	        generateUnitHypersphere(n, dim, ref, MPI_COMM_WORLD);
	        #pragma omp parallel for
	        for(int i = 0; i < n; i++) refids[i] = i;
	        #pragma omp parallel for
		for(int i = 0; i < n*dim; i++) 
			query[i] = ref[i];
	
		#pragma omp parallel for
		for(int i = 0; i < n; i++)
			queryids[i] = refids[i];
	 
	        std::pair<double, long> *result = new std::pair<double, long>[n*k];
		double *dist = new double[KNN_MAX_BLOCK_SIZE*n];
		double* sqnormr = new double[n];
		double* sqnormq = new double[KNN_MAX_BLOCK_SIZE];		

		long iters = 100;
               
                double start = omp_get_wtime();
//                papi_mpi_flop_start();
                float global_real_time=0.0, global_proc_time=0.0, global_mflops=0.0;
      		long long global_flpins = 0;
	         int retval;
	         papi_thread_flops( &global_real_time, &global_proc_time, &global_flpins, &global_mflops);
		for(int i = 0; i < iters; i++) {
	        	knn::directKQueryLowMem( ref, query, n, n, k, dim, result,
                        	                 dist, sqnormr, sqnormq);
                }
	         papi_thread_flops( &global_real_time, &global_proc_time, &global_flpins, &global_mflops);
		double total = omp_get_wtime() - start;
		cout << n << "," << total/(double)iters << endl;
 		cout << "MFLOPS:" << global_mflops << endl;

                delete[] ref;
                delete[] query;
                delete[] refids;
                delete[] queryids;
                delete[] result;
                delete[] dist;
                delete[] sqnormr;
                delete[] sqnormq;

	}

	MPI_Finalize(); 
	return 0;
}
	


	
	


