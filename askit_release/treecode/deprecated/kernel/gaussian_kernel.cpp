
#include <float.h>
#include "gaussian_kernel.hpp"


using namespace std;

void print(double *arr, int nrows, int ncols)
{
    for(int i = 0; i < nrows; i++) {
        for(int j = 0; j < ncols; j++)
            cout<<arr[i*nrows+j]<<" ";
        cout<<endl;
    }
    cout<<endl;
}


void GaussianKernel::Compute(double* row_points, int num_rows, double* col_points, int num_cols, int d, double* K)
{

  if (num_rows * num_cols <= max_matrix_size)
  {
    
    // compute the squared distances
    knn::compute_distances(row_points, col_points, num_rows, num_cols, d, workspace);

    // multiply by -0.5 h^(-2)
    cblas_dscal(num_rows*num_cols, minus_one_over_2h_sqr, workspace, 1);

    // take exponential and store in K
    vdExp(num_rows*num_cols, workspace, K);

  }
  else {
    // it's too big
    
    // TODO: fix this
    std::cout << "didn't allocate enough space for the kernel workspace\n";
    
  }

}

