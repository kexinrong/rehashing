

#ifndef GAUSSIAN_KERNEL_HPP_
#define GAUSSIAN_KERNEL_HPP_

#include <mpi.h>

#include "kernel_inputs.hpp"
#include "blas_headers.hpp"

#include <iostream>
//#include <mkl.h>
//#include <mkl_vml_functions.h>

#include "direct_knn/direct_knn.h"

namespace askit {

// Class to compute kernel matrices for Gaussian kernel
class GaussianKernel {

public:
  
  // Constructor allocates some workspace to hold distances
  GaussianKernel(KernelInputs& inputs); 
  
  ~GaussianKernel();
  
  // Returns the m x n array of results in K
  // The last argument is only needed for the variable bandwidth kernel, is 
  // ignored otherwise
  void Compute(std::vector<double>::iterator row_start, std::vector<double>::iterator row_end,
       std::vector<double>::iterator col_start, std::vector<double>::iterator col_end,  int d, 
       std::vector<double>& K, std::vector<int>& source_inds);


  // If the distances are precomputed, this just applies the kernel to each one
  void Compute(std::vector<double>& distances, std::vector<double>& K);

  // The bandwidth
  double h;

  
protected:
  // -0.5 * h^(-2)
  double minus_one_over_2h_sqr;
  
  // If true, do the variable bandwidth method
  bool do_variable_bandwidth;

  int d;
  
  // for the variable bandwidth version -- we assume that these are indexed 
  // the same way as data_table (i.e. we can use the source indices to 
  // index into this vector)
  // IMPORTANT: these are already 0.5 * h^(-2)
  std::vector<double> bandwidths;
  
  // Helper function for the variable bandwidth method
  void ComputeVariable(double* row_points, int num_rows, double* col_points, int num_cols, 
    int d, std::vector<double>& K, std::vector<int>& source_inds);
  
};

} // namespace 

#endif
