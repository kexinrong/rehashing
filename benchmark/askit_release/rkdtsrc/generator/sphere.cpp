#include <mpi.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <assert.h>
//#include <mkl.h>
//#include <mkl_blas.h>
//#include <mkl_cblas.h>
#include <omp.h>

#include "direct_knn.h"
#include "generator.h"

using std::string;
using std::ifstream;
using std::ofstream;
using std::istringstream;
using std::ios;


// n-sphere (NOTE, n should be greater than or equal to 2)
void generateNSphere(int numof_points, int dim, double max_radius, 
		double *points, MPI_Comm comm)
{
// generate n-dim sphere data which has NO overlap with other spheres
	
	double    *centers = NULL;
	double	  *dist = NULL;
	
	int        len;
	int        rank, nproc;
	int 	idx = 1;
	double rand_max = (double)RAND_MAX;
	double min_dist = rand_max;
	
	MPI_Status status;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nproc);

	srand((unsigned)time(NULL)+rank);

	//let proc 0 generate the sphere centers and distribute to others
	if (rank == 0) {
		if(1 == nproc) {
			centers = new double [dim];
			//#pragma omp parallel for
			for(int j = 0; j < dim; j++) {
				centers[j] = (double)rand()/rand_max;
			}
		}
		else {	
			centers = new double [dim*nproc];
			//#pragma omp parallel for
			for(int j = 0; j < nproc*dim; j++) {
				centers[j] = (double)rand()/rand_max;
			}
			len = nproc*nproc;
			dist = new double [len];
			knn::compute_distances(centers, centers, nproc, nproc, dim, dist );
			int index = 0;
			for(int i = 0; i < len; i++) {
				if(dist[i] > 0 && dist[i] < min_dist) {
					min_dist = dist[i];
					index = i;
				}
			}
			double t = 1.01*(max_radius)/min_dist;
			#pragma omp parallel for
			for(int j = 0; j < dim * nproc; j++) {
				centers[j] = centers[j] * t;
			}
		}
	}

	if(rank == 0) {
		for(int i = 1; i < nproc; i++)
			MPI_Send(centers+i*dim, dim, MPI_DOUBLE, i, i, comm);
	}
	else {
		centers = new double [dim];
		MPI_Recv(centers, dim, MPI_DOUBLE, 0, rank, comm, &status);
	}

	double * phi = new double [dim-1];
	for(int i = 0; i < numof_points; i++) {
		double r = (double)rand()*max_radius/rand_max;
		for(int t = 0; t < dim-3; t++) 
			phi[t] = (double)rand()*PI/rand_max;
		phi[dim-2] = (double)rand()*2*PI/rand_max;
		#pragma omp parallel for
		for(int k = 0; k < dim; k++)
			points[i*dim+k] = r;
		for(int k = 0; k < dim-1; k++) {
			for(int j = 0; j < k; j++) {
				points[i*dim+k] = points[i*dim+k] * sin(phi[j]);
			}
			points[i*dim+k] = points[i*dim+k] * cos(phi[k]);
			points[i*dim+k] += centers[k];
		}
		for(int j = 0; j < dim-1; j++)
			points[i*dim+dim-1] = points[i*dim+dim-1] * sin(phi[j]);
		points[i*dim+dim-1] += centers[dim-1];
	}
	
	delete [] centers;
	delete [] dist;
	delete [] phi;
	
}



