
#include "laplace_kernel.hpp"

using namespace askit;

// Don't have any internal state to keep up with
LaplaceKernel::LaplaceKernel(KernelInputs& inputs)
{}

LaplaceKernel::~LaplaceKernel()
{}


void LaplaceKernel::Compute(std::vector<double>::iterator row_start, std::vector<double>::iterator row_end,
       std::vector<double>::iterator col_start, std::vector<double>::iterator col_end,  int d, 
       std::vector<double>& K, std::vector<int>& source_inds)
{
  
  double* row_points = &(*row_start);
  int num_rows = (row_end - row_start) / d;
  
  double* col_points = &(*col_start);
  int num_cols = (col_end - col_start) / d;
  
  // Resize to get the right number of points
  K.resize(num_rows * num_cols);
  
  // Compute the distances  
  knn::compute_distances(row_points, col_points, num_rows, num_cols, d, K.data());
  
  // Now, call the one that takes distances as an argument
  Compute(K, K, d);
  
} // Compute (no distances)


// If the distances are precomputed, this just applies the kernel to each one
void LaplaceKernel::Compute(std::vector<double>& distances_sqr, 
    std::vector<double>& K, int d)
{
  
  // copy the distances to the output -- this shouldn't do anything expensive
  // because they're both references
  K = distances_sqr;
  
  // The kernel depends on the dimension, in 1D it's just the distance  
  if (d == 1)
  {
    
    // just return the distances for d == 1, note that these are squared on 
    // input
    for (int i = 0; i < K.size(); i++)
    {
      K[i] = sqrt(K[i]);
    }
    
  }
  else if (d == 2)
  {
    // In 2D, its the natural log

    // scale the final result by this
    double scale = 0.5 / M_PI;
    
    // take the log of each distance
    for (int i = 0; i < K.size(); i++)
    {
      // handling the self-interaction
      if (K[i] < 1e-15)
        //K[i] = 1.0 / scale;
        K[i] = 0.0;
      else 
        K[i] = log(sqrt(K[i]));

    }

    int n = K.size();
    int one = 1;
    cblas_dscal(&n, &scale, K.data(), &one);
    
  } // d == 2
  else { // d > 2
    
      // the 0.5 factor is because the distances are squared
    double power = 0.5 * (2.0 - (double)d);
    double scale = tgamma(0.5 * d + 1.0) / ((double)d * (double)(d - 2) * pow(M_PI, 0.5 * d));
    
    for (int i = 0; i < K.size(); i++)
    {
      
      // Need to correct for the self-interaction here
      // Doing this so that the result turns out to be 1
      if (K[i] < 1e-15)
        //K[i] = 1.0 / scale;
        K[i] = 0.0;
      else {    
        K[i] = pow(K[i], power);
      }
      
    } // loop over kernel entries

    // the scale depends on a gamma function in higher dimensions    
    int n = K.size();
    int one = 1;
    cblas_dscal(&n, &scale, K.data(), &one);
        
  } // d > 2
  
  
}






