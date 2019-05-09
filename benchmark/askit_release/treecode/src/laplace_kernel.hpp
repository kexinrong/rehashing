

#ifndef LAPLACE_KERNEL_HPP_
#define LAPLACE_KERNEL_HPP_

#include <mpi.h>

#include "kernel_inputs.hpp"

#include <iostream>
//#include <mkl.h>
//#include <mkl_vml_functions.h>

#include "direct_knn/direct_knn.h"

// Trying this for mkl inclusion weirdness
//#include "id.hpp"
//#include "askit_utils.hpp"

namespace askit {

// Class to compute kernel matrices for Gaussian kernel
class LaplaceKernel {

public:
  
  // Constructor allocates some workspace to hold distances
  LaplaceKernel(KernelInputs& inputs); 
  
  ~LaplaceKernel();
  
  // Returns the m x n array of results in K
  // num_rows x num_cols must be <= max_matrix size, or else will fail
  void Compute(std::vector<double>::iterator row_start, std::vector<double>::iterator row_end,
       std::vector<double>::iterator col_start, std::vector<double>::iterator col_end,  int d, 
       std::vector<double>& K, std::vector<int>& source_inds);
  
  // If the distances are precomputed, this just applies the kernel to each one
  // Note that the Laplace kernel depends on the distance, so it has to be 
  // passed here
  void Compute(std::vector<double>& distances, std::vector<double>& K, int d);
  
  // Need this to compile tests, fix properly later
  double h;
  
  
protected:

  
  
  
};

} // namespace 

#endif
