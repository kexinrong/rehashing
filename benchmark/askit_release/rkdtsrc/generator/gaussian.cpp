#include <mpi.h>
#include <stdlib.h>
#include <iostream>
#include <cstring>
#include <math.h>
#include <float.h>
#include <omp.h>
#include <vector>
#include <mpi.h>
#include "direct_knn.h"
#include "generator.h"

using namespace std;


double norm(double * vec, int n)
{
	double a = 0.0;
	for(int i = 0; i < n; i++)
		a += vec[i]*vec[i];
	return sqrt(a);
}

inline void scale(double * src, double *dst, int n, double s)
// dst = s * src
{
	for(int i = 0; i < n; i++)
		dst[i] = src[i] * s;
}


// modified Gram-Schmidt algorithm to orthogonalize a matrix
// for convenience, do a row-orthogonalization
void GramSchmidt(double * A, double * Q, int rows, int cols)
// A: input matrix, the elements will be changed after this function
// Q: output matrix, the orthogonalized matrix
{
	for(int i = 0; i < rows; i++) {
		double rii = norm(A+i*cols, cols);		
		scale(A+i*cols, Q+i*cols, cols, 1/rii);
		for(int j = i+1; j < rows; j++) {
			double rij = prod(Q+i*cols, A+j*cols, cols);
			for(int t = 0; t < cols; t++)
				A[j*cols+t] -= rij*Q[i*cols+t];
		}
	}

}

// generate a covariance matrix with specified eigenvalues and random rotations. 
// be sure that elements of eigv should be all nonnegative
// return R^T where C = RR^T
void generateCovMatrix(double *R, double *eigv, int dim)
{
	double *tmp = new double [dim*dim];
	for(int i = 0; i < dim*dim; i++) {
		tmp[i] = (double)rand()/(double)RAND_MAX;
	}
	GramSchmidt(tmp, R, dim, dim);
	for(int i = 0; i < dim; i++) {
		for(int j = 0; j < dim; j++) {
			R[i*dim+j] *= sqrt(eigv[i]);
		}
	}

	delete [] tmp;
}


// generate gaussian with mean 0 and specified gaussian
void generateGaussian(double *points, int numof_points, int dim, 
		      double *R, MPI_Comm comm)
{
	int rank, nproc;
	MPI_Comm_size(comm, &nproc);
	MPI_Comm_rank(comm, &rank);
	
	double * tmp = new double [dim];
	generateNormal(numof_points, dim, points, comm);
	for(int i = 0; i < numof_points; i++) {
		memcpy(tmp, points+i*dim, sizeof(double)*dim);
		#pragma omp parallel for
		for(int j = 0; j < dim; j++) {
			points[i*dim+j] = prod(tmp, R+j*dim, dim);
		}
	}

	delete [] tmp;

}

