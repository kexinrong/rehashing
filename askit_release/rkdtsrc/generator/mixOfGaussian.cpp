#include <mpi.h>
#include <iostream>
#include <cstring>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <omp.h>

#include "direct_knn.h"
#include "generator.h"

using namespace std;

void generateMeansCircle(int numof_centers, int dim,
			double radius, double *centers)
{
	
	if(dim < 2) {
		cerr<<"dim should be >=2\n";
		return;
	}

	/*double * phi = new double [dim-1];
	for(int t = 0; t < dim - 3; t++) {
		srand((unsigned)time(NULL)*t*100);
		phi[t] = (double)rand()*PI/(double)RAND_MAX;
	}
	for(int i = 0; i < numof_centers; i++) {
		phi[dim-2] = i*2.0*PI/(double)numof_centers;
		for(int j = 0; j < dim; j++)
			centers[i*dim+j] = radius;
		for(int k = 0; k < dim-1; k++) {
			for(int j = 0; j < k; j++)
				centers[i*dim+k] *= sin(phi[j]);
			centers[i*dim+k] *= cos(phi[k]);
		}
		for(int j = 0; j < dim-1; j++)
			centers[i*dim+dim-1] *= sin(phi[j]);
	}

	delete [] phi;*/


	for(int i = 0; i < numof_centers; i++) {
		double phi = 2.0*PI*(double)i/(double)numof_centers;
		centers[i*dim+0] = radius * sin(phi);
		centers[i*dim+1] = radius * cos(phi);
	}
	double *eigv = new double [dim];
	double *Rotation = new double [dim*dim];
	for(int i = 0; i < dim; i++)
		eigv[i] = 1.0;
	generateCovMatrix(Rotation, eigv, dim);
	double *tmp = new double [dim];
	for(int i = 0; i < numof_centers; i++) {
		memcpy(tmp, centers+i*dim, sizeof(double)*dim);
		for(int j = 0; j < dim; j++) 
			centers[i*dim+j] = prod(tmp, Rotation+j*dim, dim);
	}

	delete [] eigv;
	delete [] Rotation;
	delete [] tmp;




}

	
void generateMeansRandom(int numof_centers, int dim,
		double min_separation, double *centers)
{		
		for(int j = 0; j < numof_centers*dim; j++) {
			centers[j] = (double)rand()/(double)RAND_MAX;
		}
		int len = numof_centers*numof_centers;
		double *dist = new double [len];
		knn::compute_distances(centers, centers, numof_centers, numof_centers, dim, dist);
		int index = 0;
		double min_dist = (double)RAND_MAX;
		for(int i = 0; i < len; i++) {
			if(dist[i] > 0 && dist[i] < min_dist) min_dist = dist[i];
		}
		double t = 1.01*(min_separation)/min_dist;
		#pragma omp parallel for
		for(int j = 0; j < dim * numof_centers; j++) {
			centers[j] = centers[j] * t;
		}
		delete [] dist; 
}


void generateMeansHypercube(int numof_centers, int dim, double edge, 
			    double *centers)
{
	for(int i = 0; i < numof_centers*dim; i++) 
		centers[i] = edge/2.0;
	if(dim < 2) cerr<<"dim shoulbe be at least 2\n";
	if(dim == 2) {
		int p = 0;
		vector<double> tmp(2);
		tmp[0] = edge/2.0;
		tmp[1] = -edge/2.0;
		for(int i = 0; i < 2; i++) {
			for(int j = 0; j < 2; j++) {
				centers[p*dim+0] = tmp[i];
				centers[p*dim+1] = tmp[j];
				p++;
			}
		}
	}
	if(dim > 2) {
		int p = 0;
		vector<double> tmp(2);
		tmp[0] = edge/2.0;
		tmp[1] = -edge/2.0;
		for(int i = 0; i < 2; i++) {
			for(int j = 0; j < 2; j++) {
				for(int k = 0; k < 2; k++) {
					centers[p*dim+0] = tmp[i];
					centers[p*dim+1] = tmp[j];
					centers[p*dim+2] = tmp[k];
					p++;
				}
			}
		}
	}

}


// mixture of unit gaussian with min dist min_dist among gaussian centers
void generateMixOfUnitGaussian( int numof_points, int dim, 
			        int numof_gaussians, double min_separation, 
				double *points, int *labels, MPI_Comm comm)
{
	int        rank, nproc;
	double rand_max = (double)RAND_MAX;
	double min_dist = rand_max;
	
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nproc);

	double    *centers = NULL;
	double	  *dist = NULL;
	centers = new double [dim*numof_gaussians];
	//let proc 0 generate the gaussian centers and distribute to others
	if (rank == 0) {
		generateMeansCircle(numof_gaussians, dim, min_separation, centers);
		//generateMeansHypercube(numof_gaussians, dim, min_separation, centers);
	}
	MPI_Bcast(centers, numof_gaussians*dim, MPI_DOUBLE, 0, comm);
	
	int divd, rem;
	divd = numof_points / numof_gaussians;
	rem = numof_points % numof_gaussians;
	
	int p = 0;
	for(int i = 0; i < numof_gaussians; i++) {
		int gaussian_size = (i < rem) ? (divd+1) : divd;
		vector<double> gaussian_tmp;
		gaussian_tmp.resize(gaussian_size*dim);
		generateNormal(gaussian_size, dim, &(gaussian_tmp[0]), comm);
		for(int j = 0; j < gaussian_size; j++) {
			labels[p] = i;
			for(int t = 0; t < dim; t++) {
				points[p*dim+t] = gaussian_tmp[j*dim+t] + centers[i*dim+t];
			}
			p++;
		}
	}
	
	delete [] centers;

}



// mixture of unit gaussian with min dist min_dist among gaussian centers
void generateMixOfUserGaussian( int numof_points, int dim, 
			        int numof_gaussians, double min_separation, 
				double *var,
				double *points, int *labels, MPI_Comm comm)
{
	int        rank, nproc;
	double rand_max = (double)RAND_MAX;
	double min_dist = rand_max;
	
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nproc);

	double    *centers = NULL;
	double	  *dist = NULL;
	centers = new double [dim*numof_gaussians];
	//let proc 0 generate the gaussian centers and distribute to others
	if (rank == 0) {
		generateMeansCircle(numof_gaussians, dim, min_separation, centers);
		//generateMeansHypercube(numof_gaussians, dim, min_separation, centers);
	}
	MPI_Bcast(centers, numof_gaussians*dim, MPI_DOUBLE, 0, comm);
	
	int divd, rem;
	divd = numof_points / numof_gaussians;
	rem = numof_points % numof_gaussians;
	
	int p = 0;
	for(int i = 0; i < numof_gaussians; i++) {
		int gaussian_size = (i < rem) ? (divd+1) : divd;
		vector<double> gaussian_tmp;
		gaussian_tmp.resize(gaussian_size*dim);
		generateNormal(gaussian_size, dim, &(gaussian_tmp[0]), comm);
		for(int j = 0; j < gaussian_size; j++) {
			labels[p] = i;
			for(int t = 0; t < dim; t++) {
				points[p*dim+t] = gaussian_tmp[j*dim+t]*var[i] + centers[i*dim+t];
			}
			p++;
		}
	}
	
	delete [] centers;

}


// mixture of random gaussian with min dist min_dist among gaussian centers
void generateMixOfRandomGaussian(int numof_points, int dim, 
				int numof_gaussians, double min_separation, double max_eigv,
				double *points, int *labels, MPI_Comm comm)
{
	int        rank, nproc;
	double rand_max = (double)RAND_MAX;
	double min_dist = rand_max;
	
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nproc);

	double    *centers = NULL;
	centers = new double [dim*numof_gaussians];
	//let proc 0 generate the gaussian centers and distribute to others
	if (rank == 0) {
		generateMeansCircle(numof_gaussians, dim, min_separation, centers);
		//generateMeansHypercube(numof_gaussians, dim, min_separation, centers);
	}
	MPI_Bcast(centers, numof_gaussians*dim, MPI_DOUBLE, 0, comm);
	
	int divd, rem;
	divd = numof_points / numof_gaussians;
	rem = numof_points % numof_gaussians;
	
	int p = 0;
	double * R = new double [dim*dim];
	double * eigv = new double [dim];
	for(int i = 0; i < numof_gaussians; i++) {
		int gaussian_size = (i < rem) ? (divd+1) : divd;
		vector<double> gaussian_tmp;
		gaussian_tmp.resize(gaussian_size*dim);
		if(rank == 0) {
			for(int l = 0; l < dim; l++) {
				eigv[l] = fabs( max_eigv * (double)rand()/(double)RAND_MAX );
			}
			generateCovMatrix(R, eigv, dim);
		}
		MPI_Bcast(R, dim*dim, MPI_DOUBLE, 0, comm);
		
		generateGaussian(&(gaussian_tmp[0]), gaussian_size, dim, R, comm);
		for(int j = 0; j < gaussian_size; j++) {
			labels[p] = i;
			for(int t = 0; t < dim; t++) {
				points[p*dim+t] = gaussian_tmp[j*dim+t] + centers[i*dim+t];
			}
			p++;
		}
	}
	
	delete [] centers;
	delete [] R;
	delete [] eigv;
}



// mixture of random gaussian with random rotations and specified eigenvalues, 
// min dist min_dist among gaussian centers
void generateMixOfSpecifiedGaussian(int numof_points, int dim, 
				    int numof_gaussians, double min_separation, double *eigv,
				    double *points, int *labels, MPI_Comm comm)
{
	int        rank, nproc;
	double rand_max = (double)RAND_MAX;
	double min_dist = rand_max;
	
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nproc);

	double    *centers = NULL;
	centers = new double [dim*numof_gaussians];
	//let proc 0 generate the gaussian centers and distribute to others
	if (rank == 0) {
		generateMeansCircle(numof_gaussians, dim, min_separation, centers);
		//generateMeansHypercube(numof_gaussians, dim, min_separation, centers);
	}
	MPI_Bcast(centers, numof_gaussians*dim, MPI_DOUBLE, 0, comm);
	
	int divd, rem;
	divd = numof_points / numof_gaussians;
	rem = numof_points % numof_gaussians;
	
	int p = 0;
	double * R = new double [dim*dim];
	for(int i = 0; i < numof_gaussians; i++) {
		int gaussian_size = (i < rem) ? (divd+1) : divd;
		vector<double> gaussian_tmp;
		gaussian_tmp.resize(gaussian_size*dim);
		if(rank == 0) {
			generateCovMatrix(R, eigv, dim);
		}
		MPI_Bcast(R, dim*dim, MPI_DOUBLE, 0, comm);
		
		generateGaussian(&(gaussian_tmp[0]), gaussian_size, dim, R, comm);
		for(int j = 0; j < gaussian_size; j++) {
			labels[p] = i;
			for(int t = 0; t < dim; t++) {
				points[p*dim+t] = gaussian_tmp[j*dim+t] + centers[i*dim+t];
			}
			p++;
		}
	}
	
	delete [] centers;
	delete [] R;
}


