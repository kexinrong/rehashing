

#ifndef MATERN_KERNEL_HPP_
#define MATERN_KERNEL_HPP_

#include <mpi.h>
#include "kernel_inputs.hpp"
#include <iostream>
#include "direct_knn/direct_knn.h"
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_gamma.h>


namespace askit {

class MaternKernel {

public:
  MaternKernel(KernelInputs& inputs); 
  ~MaternKernel();
  
  void Compute(std::vector<double>::iterator row_start, std::vector<double>::iterator row_end,
       std::vector<double>::iterator col_start, std::vector<double>::iterator col_end,  int d, 
       std::vector<double>& K, vector<int>& source_inds);

  void Compute(std::vector<double>& distances, std::vector<double>& K, int d);
  
  double nu;  
	double sqrt2nu;
	double prefactor; 
  
protected:
  
  
};

} // namespace 

#endif
