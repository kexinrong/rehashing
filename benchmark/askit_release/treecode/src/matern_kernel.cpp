
#include "matern_kernel.hpp"

#define NEARZERO 1E-15
#define ZEROFIX  200

using namespace askit;

MaternKernel::MaternKernel(KernelInputs& inputs)
{	
	nu = inputs.nu;
	sqrt2nu = sqrt(2*nu);
	prefactor =  1.0 / ( pow(2, nu-1) * gsl_sf_gamma(nu) );
}

MaternKernel::~MaternKernel()
{}


void MaternKernel::Compute(std::vector<double>::iterator row_start, std::vector<double>::iterator row_end,
       std::vector<double>::iterator col_start, std::vector<double>::iterator col_end,  int d, 
       std::vector<double>& K, vector<int>& source_inds)
{
  
  double* row_points = &(*row_start);
  int num_rows = (row_end - row_start) / d;
  double* col_points = &(*col_start);
  int num_cols = (col_end - col_start) / d;

  K.resize(num_rows * num_cols);
  knn::compute_distances(row_points, col_points, num_rows, num_cols, d, K.data());
  Compute(K, K, d);
  
} 


void MaternKernel::Compute(std::vector<double>& distances_sqr, 
    std::vector<double>& K, int d)
{
 
	// 
	// this will set mattern_kernel(nu, nearzero) ~ 0 as opposed to INF
	for (int i = 0; i < K.size(); i++)
		K[i] = K[i]<NEARZERO  ? ZEROFIX : sqrt2nu*sqrt(K[i]); 


	for (int i=0; i<K.size(); i++) {
		double a = gsl_sf_bessel_Knu(nu,K[i]);
		double b = pow(K[i],nu); 
		K[i] = prefactor * a * b;
	}
}






