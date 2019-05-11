#ifndef _ROTATION_H__
#define _ROTATION_H__

#include <mpi.h>
#include <omp.h>
#include <blas.h>
#include <vector>
#include <stdlib.h>
#include <math.h>

/* ************************************************** */
// interface for FORTRAN CODE
#define FORTRAN(MYFUN) (MYFUN##_)
extern "C" {
	void FORTRAN(idd_random_transf_init)(int *nsteps, int *n, double *w, int *keep);
	void FORTRAN(idd_random_transf)(double *x, double *y, double *w);
	void FORTRAN(idd_random_transf_inverse)(double *x, double *y, double *w);
}
#define RROT_INIT FORTRAN(idd_random_transf_init)
#define RROT FORTRAN(idd_random_transf)
#define RROT_INV FORTRAN(idd_random_transf_inverse)
/* ************************************************** */

using namespace std;

void generateRotation(int dim, vector<double> &w );


void generateRotation(int dim, vector<double> &w, MPI_Comm comm);


void rotatePoints(const double *points, int numof_points, int dim, 
				  vector<double> &w,
				  // output
				  double *newPoints);

void inverseRotatePoints(const double *points, int numof_points, int dim, 
				  vector<double> &w,
				  // output
				  double *newPoints);

void newRotatePoints(double *points, int numof_points, int dim, vector<double> &w);


void newInverseRotatePoints(double *points, int numof_points, int dim, vector<double> &w);

/*
void generateRotation(int dim, double *R);


void generateRotation(int dim, double *R, MPI_Comm comm);


void rotatePoints(double *points, int numof_points, int dim, 
				  double *MatRotation,
				  // output
				  double *newPoints);

void inverseRotatePoints(double *points, int numof_points, int dim, 
				  double *MatRotation,
				  // output
				  double *newPoints);
*/



#endif
