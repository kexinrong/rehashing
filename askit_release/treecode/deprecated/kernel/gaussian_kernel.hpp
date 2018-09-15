

#ifndef GAUSSIAN_KERNEL_HPP_
#define GAUSSIAN_KERNEL_HPP_

#include "mkl.h"
#include "mkl_vml_functions.h"

#include "direct_knn/direct_knn.h"

class GaussianKernel {

public:
  
  GaussianKernel(double h, int max_mat_size) 
    : 
    minus_one_over_2h_sqr(-0.5 / (h*h)),
    max_matrix_size(max_mat_size)
   {
    
    workspace = new double[max_matrix_size];
    
  }
  
  ~GaussianKernel() {
    
    delete [] workspace;
    
  }
  
  // Returns the m x n array of results in K
  void Compute(double* row_points, int num_rows, double* col_points, int num_cols, int d, double* K);
  
protected:

  double* workspace;

  int max_matrix_size;

  double minus_one_over_2h_sqr;
  
};



#endif
