#ifndef __BLAS_H__
#define __BLAS_H__

#define sgemm sgemm_
#define dgemm dgemm_   
#define ssteqr ssteqr_
#define dsteqr dsteqr_
#define dgemv dgemv_
#define ddot ddot_
#define daxpy daxpy_
#define dscal dscal_
#define dasum dasum_

#ifdef __cplusplus
extern "C"{
#endif
    //#include <cblas.h>
    //#include <clapack.h>

    void sgemm_(const char* TRANSA, const char* TRANSB,
        const int* M, const int* N, const int* K,
        const float* ALPHA, const float* A, const int* LDA, 
        const float* B, const int* LDB, 
        const float* BETA, float* C, const int* LDC);  

    void dgemm_(const char* TRANSA, const char* TRANSB,
        const int* M, const int* N, const int* K,
        const double* ALPHA, const double* A, const int* LDA,
        const double* B, const int* LDB, 
        const double* BETA, double* C, const int* LDC); 
/*
    void ssteqr_(char *compz, const int *n, float *d, float *e, 
        float *z, const int *ldz, float *work, const int *info);

    void dsteqr_(char *compz, const int *n, double *d, double *e, 
        double *z, const int *ldz, double *work, const int *info);
 */

     void dgemv_(const char *trans, const int *m, const int *n,
          const double *alpha, const double *a, const int *lda, 
          const double *x, const int *incx, const double *beta, 
          double *y, const int *incy);

     double ddot_(const int* n, const double* x, const int* incx, 
          const double* y, const int* incy);


     void daxpy_(const int* n, const double *alpha, const double* x, 
	const int* incx, double* y, const int* incy);
 
	 void dscal_(const int*n, const double *alpha, const double* x, const int* incx);

	 double dasum_(const int*n, const double *x, const int* incx);

#ifdef __cplusplus
}
#endif

#endif
