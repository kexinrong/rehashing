
//#include <float.h>
#include "polynomial_kernel.hpp"


using namespace std;
using namespace askit;


PolynomialKernel::PolynomialKernel(KernelInputs& inputs) 
  : 
  h(inputs.bandwidth),
  c(inputs.constant),
  p(inputs.power)
 {}

PolynomialKernel::~PolynomialKernel() {}


void PolynomialKernel::Compute(std::vector<double>::iterator row_start, std::vector<double>::iterator row_end,
       std::vector<double>::iterator col_start, std::vector<double>::iterator col_end,  int d, 
       std::vector<double>& K, std::vector<int>& source_inds)
{

  double* row_points = &(*row_start);
  int num_rows = (row_end - row_start) / d;

  double* col_points = &(*col_start);
  int num_cols = (col_end - col_start) / d;

  // Check that we allocated enough space, if not, resize to get enough
  if (num_rows * num_cols > K.size())
  {
    K.resize(num_rows * num_cols);
  }

//  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, num_rows, num_cols, d, 
  //  1.0, row_points, d, col_points, d, 0.0, K.data(), num_rows);
  double one = 1.0;
  double zero = 0.0;
  cblas_dgemm("T", "N", &num_rows, &num_cols, &d, 
    &one, row_points, &d, col_points, &d, &zero, K.data(), &num_rows);

  // Scale by 1/h
  int size = num_rows * num_cols;
  double scale = 1.0 / h;
  int onei = 1;
  //cblas_dscal(num_rows * num_cols, 1.0 / h, K.data(), 1);
  cblas_dscal(&size, &scale, K.data(), &onei);
  
  // Add c and exponentiate
  for (int i = 0; i < num_rows * num_cols; i++)
  {
    K[i] = pow(K[i] + c, p);
  }
  
} // Compute()



