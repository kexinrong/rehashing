#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <iostream>

using namespace std;

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

	cout<<"input before of RROT: ";
	for(int i = 0; i < d; i++)
		cout<<t[i]<<" ";
	cout<<endl;

	#pragma omp parallel for	
	for (int i=0; i<n; i++)	RROT(t+i*d, y+i*d ,w);   // apply rotation
	double end = omp_get_wtime();
	
	cout<<"input after of RROT: ";
	for(int i = 0; i < d; i++)
		cout<<t[i]<<" ";
	cout<<endl;

	cout<<"output of RROT: ";
	for(int i = 0; i < d; i++)
		cout<<y[i]<<" ";
	cout<<endl;

	// try to get the rotation matrix;
	double *E = new double [d*d];
	memset(E, 0, sizeof(double)*d*d);
	for(int i = 0; i < d; i++)
		E[i*d+i] = 1.0;
	
	double *R = new double [d*d];
	for(int i = 0; i < d; i++) RROT(E+i*d, R+i*d, w);
	
	double *OE = new double [d*d];
	for(int i = 0; i < d; i++) RROT_INV(R+i*d, OE+i*d, w);
	
	double *re = new double [d];
	double *e =new double [d];
	e[2] = 1.0;
	RROT_INV(e, re, w);
	cout<<"inv of e_3: "<<endl;
	for(int i = 0; i < d; i++)
		cout<<re[i]<<" ";
	cout<<endl;
		

	cout<<"unit matrix: "<<endl;
	for(int i = 0; i < d; i++) {
		for(int j = 0; j < d; j++)
			cout<<E[i*d+j]<<" ";
		cout<<endl;
	}
	cout<<endl;

	cout<<"rotation matrix: "<<endl;
	for(int i = 0; i < d; i++) {
		for(int j = 0; j < d; j++)
			cout<<R[i*d+j]<<" ";
		cout<<endl;
	}
	cout<<endl;

	cout<<"inverse rotation matrix: "<<endl;
	for(int i = 0; i < d; i++) {
		for(int j = 0; j < d; j++)
			cout<<OE[i*d+j]<<" ";
		cout<<endl;
	}
	cout<<endl;
	
	
	printf("Elapsed time for rotation:%f\n", end-start);

	delete[] x;
	delete[] t;
	delete[] y;
	delete[] z;
	delete[] w;
	return 1;
}

