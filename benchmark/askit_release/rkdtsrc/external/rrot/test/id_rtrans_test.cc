#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

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
int main(int argc, char *argv[])
{
	printf("\nUSAGE\n %s [dimension (default=3)] [number of vectors (default=1)] [nsteps (default=10)]\n\n",argv[0]);
	
	int d      = argc>1 ? int(atoi(argv[1])) : 3;                 // dimension
	int n      = argc>2 ? int(atoi(argv[2])) : 1;                 // number of vectors
	int nsteps = argc>3 ? int(ceil(log2(d)*atoi(argv[3]))) : 10;  // parameter used in randomized rotations

	//	printf("%d %d\n", d, nsteps);

	// setup arrays
	double *x = new double[d*n];
	double *t = new double[d*n];  //tmp array, will be overwriten upon output
	double *y = new double[d*n];
	double *z = new double[d*n];

	for( int i=0; i<n; i++)
		for( int j=0; j<d; j++) {
			x[i*d + j] = j;    // initialize all vectors to the same values. 
			t[i*d + j] = j;    // initialize all vectors to the same values. 
		}

	// work space needed by the randomized vector routines.
	int sz = int(ceil(3*nsteps*d + 2*d + d/4.0 + 50*sizeof(double) +1));
	double *w = new double[sz];
	int keep;
	RROT_INIT( &nsteps,&d,w,&keep);    // initialize rotation matrix.

	double start = omp_get_wtime();
	#pragma omp parallel for	
	for (int i=0; i<n; i++)	RROT(t+i*d, y+i*d ,w);   // apply rotation

	double end = omp_get_wtime();
	RROT_INV(y,z,w);


	// try to get the rotation matrix;



	// just print the first vector 
	printf("X\t\t\t Y\t\t\t Z\n");
	for(int i=0; i<d; i++)	printf( "%f\t\t %f\t\t %f\n", x[i], y[i], z[i]);

	// norm check for first vector.
	double nrmx=0;
	double nrmy=0;
	for( int i=0; i<d; i++){
		nrmx += x[i] * x[i];
		nrmy += y[i] * y[i];
	}
	
	printf("\nNorm X=%f \t\t  Norm Y=%f\n", sqrt(nrmx), sqrt(nrmy));
	printf("Elapsed time for rotation:%f\n", end-start);

	delete[] x;
	delete[] t;
	delete[] y;
	delete[] z;
	delete[] w;
	return 1;
}

