
#ifndef KERNEL_INPUTS_HPP_
#define KERNEL_INPUTS_HPP_

#include "blas_headers.hpp"
#include <vector>

namespace askit {

  enum KernelType {ASKIT_GAUSSIAN, ASKIT_LAPLACE, ASKIT_POLYNOMIAL};
  
  /** 
   * This class just holds the parameters needed for different kernel functions
   * It is the user's responsibility to set the parameters needed for the 
   * kernel function being used. 
   */
  class KernelInputs {
  
  public: 
    
    // For Gaussian kernel -- only need to specify these for Gaussian

    // h 
    double bandwidth; 
    // if true, do the source vector dependent variable bandwidth kernel
    bool do_variable_bandwidth;

    // The bandwidths for each source point -- filled in by the ASKIT 
    // constructor
    std::vector<double> variable_h;

    // For Laplace kernel -- no extra parameters necessary
  
  
    // For Polynomial kernel -- (x' * y / h + c)^p
    // The exponent to use, a double for now, maybe an int?
    double power;

    // The constant added (inside the exponent)    
    double constant;

		// MATERN KERNEL
		double nu; // good vaules: 1.5
    
    KernelType type;
    
    // Also need to specify the bandwidth above 
    
    KernelInputs() 
      :
    // defaulting to this, needs to be set for everything else 
    type(ASKIT_GAUSSIAN),
    do_variable_bandwidth(false),
    // these have defaults so that the output file will print more nicely
    bandwidth(0.0),
    power(0.0),
    constant(0.0),
    nu(0.0)
    {}
    
    
  }; // class

} // namespace

#endif
