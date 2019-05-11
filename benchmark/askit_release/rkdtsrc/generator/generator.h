
#ifndef _KNN_GENERATOR_H_
#define _KNN_GENERATOR_H_


#include <mpi.h>
//#include "random123wrapper.h"
#include<blas.h>


#define PI 3.1416

/**
  * Generates random points on the interval [0, 1] among the processes in comm. The interval is divided into size pieces and the ith processor generates evenly distributed points amongst the ith subdivision of the line.
  * /param localPoints The local number of points to generate.
  * /param dim The dimension of the points to generate.
  * /param numbers (out) The array to store the generated points in.
  * /param globalIDs (out) The global ID of each generated point.
  * /param comm The communicator to use.
  * /param shuffle The generated points are randomly shuffled if this is set.
  * /param offset The offset of the first globalID (In general, should be totalNumberOfPoints / size * rank).
*/
void genPointInRandomLine(long localPoints, int dim, double *numbers, int *globalIDs, MPI_Comm comm, bool shuffle, int offset);


/**
 * Generates a set of points uniformly distributed on the surface of a d-dimensional hypersphere,
 * centered at the orgin.
 * @param n The local number of points to generate.
 * @param d The dimensionality of the generated points.
 * @param x [out] A pre-allocated array of n*d doubles.
 */
void generateUnitHypersphere(int n, int d, double *x, MPI_Comm comm);


/**
 * Generates a set of points uniformly distributed on a d-dimensional spherical manifold with radius
 * 1 and random center, embedded in spatialD-dimensional space.
 * @param n The local number of points to generate.
 * @param d The dimensionality of the generated points.
 * @param spatialD The dimensionality of the space the points are embedded in.
 * @param x [out] A pre-allocated array of n*spatialD doubles.
 */
void generateUnitHypersphereEmbedded(int n, int d, int spatialD, double *x, MPI_Comm comm);



/**
 * Generates a set of points from a multivariate N(0,1) distribution.
 * @param n The local number of points to generate.
 * @param d The dimensionality of the generated points.
 * @param x [out] A pre-allocated array of n*d doubles.
 */
void generateNormal(int n, int d, double *x, MPI_Comm comm);


/**
 * Generates a set of points from a uniform distribution in the unit hypercube.
 * @param n The local number of points to generate.
 * @param d The dimensionality of the generated points.
 * @param x [out] A pre-allocated array of n*d doubles.
 */
void generateUniform(int n, int d, double *x, MPI_Comm comm);


/**
 * Generates a set of points from a uniform distribution within seperated herspheres.
 * @param n The local number of points to generate.
 * @param d The dimensionality of the generated points.
 * @param max_radius The maximum radius of hyperspheres
 * @param x [out] A pre-allocated array of n*d doubles.
 */
void generateNSphere(int numof_points, int dim, double max_radius, double *points, MPI_Comm comm);


/**
 * Generates a set of points from a mixture of gaussian distribution
 * @param n The local number of points to generate.
 * @param d The dimensionality of the generated points.
 * @param numof_gaussians The number of gaussians
 * @param dist_separation The mininum distance among different gaussians 
 * @param x [out] A pre-allocated array of n*d doubles.
 */
void generateMixOfUnitGaussian( int numof_points, int dim, 
			        int numof_gaussians, double min_separation,
			        double *points, int *labels, MPI_Comm comm);

void generateMixOfUserGaussian( int numof_points, int dim, 
			        int numof_gaussians, double min_separation,
				double *var,
			        double *points, int *labels, MPI_Comm comm);

void GramSchmidt(double *A, double *Q, int rows, int cols);
void generateCovMatrix(double *R, double *eigv, int dim);

inline double prod(double *src1, double *src2, int n)
{
        double pd = 0.0;
	int one = 1;
	pd = ddot(&n, src1, &one, src2, &one);

        return pd;

}

void generateGaussian(double *points, int numof_points, int dim, double *R, 
		      MPI_Comm comm);


void generateMixOfSpecifiedGaussian(int numof_points, int dim,
				int numof_gaussians, double min_separation, double *eigv, 
				double *points, int *labels, MPI_Comm comm);

void generateMixOfRandomGaussian(int numof_points, int dim,
				int numof_gaussians, double min_separation, double max_eigv,
				double *points, int *labels, MPI_Comm comm);


void generateUniformEmbedding(int numof_points, int dim, int intrinsicDim, 
							  double *points, MPI_Comm comm);

void generateNormalEmbedding(int numof_points, int dim, int intrinsicDim, 
							  double *points, MPI_Comm comm);



#endif


