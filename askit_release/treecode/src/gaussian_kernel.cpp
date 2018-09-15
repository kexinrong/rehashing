
//#include <float.h>
#include "gaussian_kernel.hpp"


using namespace std;
using namespace askit;

// For debugging purposes only
/*
void print(double *arr, int nrows, int ncols)
{
    for(int i = 0; i < nrows; i++) {
        for(int j = 0; j < ncols; j++)
            cout<<arr[i*nrows+j]<<" ";
        cout<<endl;
    }
    cout<<endl;
}
*/

GaussianKernel::GaussianKernel(KernelInputs& inputs) 
  : 
  h(inputs.bandwidth),
  minus_one_over_2h_sqr(-0.5 / (h*h)),
  do_variable_bandwidth(inputs.do_variable_bandwidth),
  bandwidths(inputs.variable_h)
 {
   
   for (int i = 0; i < bandwidths.size(); i++)
   {
     bandwidths[i] = -0.5 / (bandwidths[i] * bandwidths[i]);
   }
   
 }

GaussianKernel::~GaussianKernel() {}


// Implements George's variable bandwidth algorithm for demonstration purposes
// Perturbs the bandwidth h according to the distance of the source point from 
// the origin
void GaussianKernel::ComputeVariable(double* row_points, int num_rows, double* col_points, int num_cols, 
  int d, std::vector<double>& K, vector<int>& h_inds)
{

  // scale factor for the variable bandwidth kernel
  double scale_factor = 0.5;
  
  int one = 1;

  // compute the squared distances
  knn::compute_distances(row_points, col_points, num_rows, num_cols, d, K.data());
  
  // Now, scale each column of K differently
  // TODO: Is there a BLAS routine for diagonal matrix multiply? 
  // loop over columns, scale each one
  // NOTE: bandwidths here is already scaled (-0.5 h^-2)
  for (int i = 0; i < num_cols; i++)
  {
    double this_h = bandwidths[h_inds[i]];
    cblas_dscal(&num_rows, &this_h, K.data()+i*num_rows, &one);
  }
  
  // now, take the exponential
  for (int i = 0; i < num_rows*num_cols; i++)
  {
    K[i] = exp(K[i]);
  }
  
} // ComputeVariable




void GaussianKernel::Compute(std::vector<double>::iterator row_start, std::vector<double>::iterator row_end,
       std::vector<double>::iterator col_start, std::vector<double>::iterator col_end,  int d, 
       std::vector<double>& K, vector<int>& source_inds)
{

  double* row_points = &(*row_start);
  int num_rows = (row_end - row_start) / d;
  
  double* col_points = &(*col_start);
  int num_cols = (col_end - col_start) / d;

  int one = 1;

  //int my_thread_id = omp_get_thread_num();

  // Check that we allocated enough space, if not, resize to get enough
  if (num_rows * num_cols > K.size())
  {
    //std::cout << "resizing kernel space\n";
    K.resize(num_rows * num_cols);
  }
  
  if (do_variable_bandwidth)
  {
    ComputeVariable(row_points, num_rows, col_points, num_cols, d, K, source_inds);
  }
  else {
    // use previously existing single bandwidth version
    
    // compute the squared distances
    knn::compute_distances(row_points, col_points, num_rows, num_cols, d, K.data());
    //std::cout << "\n Computing kernel, dist: " << K[0] << " \n";

    
    // multiply by -0.5 h^(-2)
    int n = num_rows * num_cols;
    cblas_dscal(&n, &minus_one_over_2h_sqr, K.data(), &one);

    // vdExp was causing occasional crashes when multi-threading -- not sure why
    // take exponential and store in K
    //vdExp(num_rows*num_cols, workspace.data(), K.data());
    for (int i = 0; i < num_rows*num_cols; i++)
    {
      K[i] = exp(K[i]);
    }

  } // not doing variable bandwidth

} // Compute()

// If the distances are precomputed, this just applies the kernel to each one
void GaussianKernel::Compute(std::vector<double>& distances_sqr, std::vector<double>& K)
{
  
  // copy the distances
  K = distances_sqr;
  
  // multiply by -0.5 h^(-2)
  int n = K.size();
  int one = 1;
  cblas_dscal(&n, &minus_one_over_2h_sqr, K.data(), &one);

  // vdExp was causing occasional crashes when multi-threading -- not sure why
  // take exponential and store in K
  //vdExp(num_rows*num_cols, workspace.data(), K.data());
  for (int i = 0; i < K.size(); i++)
  {
    K[i] = exp(K[i]);
  }
  
  
}




