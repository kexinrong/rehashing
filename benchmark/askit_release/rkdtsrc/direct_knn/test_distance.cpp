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
	int numof_query_points;
	int dim;
};



void ParseInput(CmdLine &cmd, TestOptions &o, int argc, char **argv)
{
	const char *pchHelp = "This is hopeless.";
	cmd.addInfo(pchHelp);
	cmd.addICmdOption("-n", &o.numof_points, 5000, "number of reference points generated per proc (5000)");
	cmd.addICmdOption("-m", &o.numof_query_points, 5000, "number of query points generated per proc (5000)");
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
	int n = o.numof_points;
	int m = o.numof_query_points;

        double *ref = new double[n*dim];
        double *query = new double[m*dim];

	//Generate points uniformly distributed on the surface of the unit hypersphere  
        generateUnitHypersphere(o.numof_points, dim, ref, MPI_COMM_WORLD);
        generateUnitHypersphere(o.numof_query_points, dim, query, MPI_COMM_WORLD);

	double *dist;
	dist = new double[n*m];

	knn::compute_distances( ref, query, n, m, dim, dist );

	cout << "Distance calculation complete." << endl;

        int mismatches = 0;
        for(int i = 0; i < m; i++) {
          for( int j = 0; j < n; j++ ) {
             double sqdist = 0.0;
             double temp;
             for( int k = 0; k < dim; k++ ) {
               temp = ref[j*dim+k] - query[i*dim+k];
               sqdist += temp*temp; 
             }
             if( sqrt(sqdist - dist[i*n+j]) > 1.0e-6 )
               mismatches++;
          }
        }	

        cout << "mismatches: " << mismatches << endl;
	MPI_Finalize(); 
	return 0;
}
	


	
	


