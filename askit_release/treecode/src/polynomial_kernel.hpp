

#ifndef POLYNOMIAL_KERNEL_HPP_
#define POLYNOMIAL_KERNEL_HPP_

#include "kernel_inputs.hpp"

#include <vector>
#include <math.h>
#include <iostream>
//#include <mkl.h>
//#include <mkl_vml_functions.h>

// Trying this for mkl inclusion weirdness
//#include "id.hpp"
//#include "askit_utils.hpp"

namespace askit {

  // Class to compute kernel matrices for Polynomial kernel
  // Computes (x' * y / h + c)^p
class PolynomialKernel {

public:
  
  // Constructor allocates some workspace to hold distances
  PolynomialKernel(KernelInputs& inputs); 
  
  ~PolynomialKernel();
  
  // Returns the m x n array of results in K
  // num_rows x num_cols must be <= max_matrix size, or else will fail
  void Compute(std::vector<double>::iterator row_start, std::vector<double>::iterator row_end,
       std::vector<double>::iterator col_start, std::vector<double>::iterator col_end,  int d, 
       std::vector<double>& K, std::vector<int>& source_inds);

   // scale for the dot product
   // Needs to be public for GPU stuff -- fix properly later
   double h;
  
protected:

  // tradeoff between high and low order terms
  double c;
  
  // degree of kernel
  double p;
  
};

} // namespace 

#endif
