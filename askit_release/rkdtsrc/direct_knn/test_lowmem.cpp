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
	int numof_points;
	int dim;
};



void ParseInput(CmdLine &cmd, TestOptions &o, int argc, char **argv)
{
	const char *pchHelp = "This is hopeless.";
	cmd.addInfo(pchHelp);
	cmd.addICmdOption("-cp", &o.numof_points, 5000, "number of data points generated per proc (5000)");
	cmd.addICmdOption("-gd", &o.dim, 8, "dimensionality of points generated per proc (8)");
	cmd.read(argc, argv);
}



int main( int argc, char **argv)
{
        MPI_Init(&argc, &argv);
	CmdLine cmd;
	TestOptions o;
	ParseInput(cmd, o, argc, argv);

	int dim = o.dim;
	int n = o.numof_points;
        int k = 10;

        double *ref = new double[n*dim];
        double *query = new double[n*dim];
        long *refids = new long[n];
        long *queryids = new long[n];

	//Generate points uniformly distributed on the surface of the unit hypersphere  
        generateUnitHypersphere(o.numof_points, dim, ref, MPI_COMM_WORLD);
        #pragma omp parallel for
        for(int i = 0; i < o.numof_points; i++) refids[i] = i;


        #pragma omp parallel for
	for(int i = 0; i < o.numof_points*dim; i++) 
		query[i] = ref[i];

	#pragma omp parallel for
	for(int i = 0; i < o.numof_points; i++)
		queryids[i] = refids[i];

 
        std::pair<double, long> *normalknn;
        std::pair<double, long> *knnlowmem = new std::pair<double, long>[n*k];

        normalknn = knn::directKQuery( ref, query, n, n, k, dim );
	knn::directKQueryLowMem( ref, query, n, n, k, dim, knnlowmem );

        int mismatches = 0;
        for(int i = 0; i < k*n; i++) {
           if( fabs(normalknn[i].first - knnlowmem[i].first) > 1.0e-6 ) mismatches++;
        }	

        cout << "mismatches: " << mismatches << endl;
	MPI_Finalize(); 
	return 0;
}
	


	
	


