/**
  * ID headers
  */

#ifndef ID_HEADERS_HPP_
#define ID_HEADERS_HPP_

#include <vector>
#include <iostream>

#include <mkl.h>
#include <mkl_lapacke.h>

// container for the workspace needed for an ID calculation
class IDWorkspace {
  
public:
  
  int max_rank;
  int max_cols;
  
  lapack_int* skeleton;
  double* tau;
  double* R_11;

  IDWorkspace(int max_rank_in, int max_cols_in)
    :
  max_rank(max_rank_in),
  max_cols(max_cols_in)
  {
    skeleton = new lapack_int[max_cols];
    tau = new double[max_cols];
    R_11 = new double[max_rank * max_rank];
  }
  
  ~IDWorkspace() 
  {
    delete [] skeleton;
    delete [] tau;
    delete [] R_11;
  }

};



// for debugging purposes
void print_mat(double* mat, int m, int n);

void print_mat(double* mat, int m, int n, int jump);

void print_upper_triangular(double* mat, int m, int n);


/**
  * Computes the rank k interpolative decomposion of A
  * Returns the indicies in skeleton and the projection matrix in proj 
  *
  * IMPORTANT: will modify its input matrix A
  *
  * workspace needs at to have been created with at least this rank and num_cols
  * 
  * skeleton and proj should be preallocated to size rank and rank * (num_cols - rank)
  */
// References to pointers, awesome!
int compute_id(double* A, int num_rows, int num_cols, int rank, std::vector<lapack_int>& skeleton, 
               std::vector<double>& proj,
               IDWorkspace& workspace);


#endif 


