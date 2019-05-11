/**
  * ID headers
  */

#ifndef ID_HEADERS_HPP_
#define ID_HEADERS_HPP_

#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>

//#include <mkl.h>
#include <mkl_lapacke.h>

#include <omp.h>

namespace askit {

// container for the workspace needed for an ID calculation
class IDWorkspace {
  
public:
  
  
  // Needs to be size max_cols
  std::vector<lapack_int> skeleton;
  
  // Reflectors for lapack QR
  std::vector<double> tau;
  
  // Needed in linear solve, rank x rank
  std::vector<double> R_11;

  // Allocate the memory
  IDWorkspace(int max_rank_in, int max_cols_in)
    :
  skeleton(max_cols_in),
  tau(max_cols_in),
  R_11(max_rank_in*max_rank_in)
  {
  }
  
  // Free the memory
  ~IDWorkspace() 
  {
  }
  
}; //class IDWorkspace



// for debugging purposes
void print_mat(double* mat, int m, int n);

// for debugging purposes
void print_mat(double* mat, int m, int n, int jump);

// for debugging purposes
void print_upper_triangular(double* mat, int m, int n);


/**
 * After the QR factorization, this function computes the matrix proj.
 */
lapack_int solve_for_proj(double* A, int num_rows, int num_cols, int rank, 
  std::vector<lapack_int>& skeleton_out, std::vector<double>& proj, IDWorkspace& workspace);


/**
  * Computes the rank k interpolative decomposion of A
  * Returns the permutation in skeleton.  The first k indices are the indices
  * of the skeleton points.
  *
  * IMPORTANT: will modify its input matrix A
  *
  * workspace needs at to have been created with at least this rank and num_cols
  * or it will fail.
  * 
  */
int compute_id(double* A, int num_rows, int num_cols, int rank, std::vector<lapack_int>& skeleton, 
               std::vector<double>& proj, IDWorkspace& workspace, double& solve_for_proj_time, double& qr_time);

int compute_id(double* A, int num_rows, int num_cols, int rank, std::vector<lapack_int>& skeleton, 
              std::vector<double>& proj, IDWorkspace& workspace);


/**
 * Computes a variable rank ID of A.
 * epsilon is the absolute error tolerance in the estimated singular values.
 * max_rank is the maximum possible rank.
 *               
 * Returns the rank found. If the matrix cannot be compressed with fewer than 
 * max_rank columns, then returns a negative number.            
 * If num_cols <= max_rank and the matrix can't be compressed, then it 
 * returns min(num_cols, num_rows) -- i.e. the entire matrix
*/               
int compute_adaptive_id(double* A, int num_rows, int num_cols, 
  std::vector<lapack_int>& skeleton, std::vector<double>& proj, IDWorkspace& workspace,
  double epsilon, int max_rank, bool printout = false);


  /**
   * This version computes an adaptive ID while trying to correct for the 
  * effect of sampling a subset of rows on the spectrum of K.
  * 
  * It takes in coordinates for two sets of points: the nearest neighbors 
  * and the uniformly sampled points.  It then estimates the amount of the 
  * spectrum covered by the uniform samples and scales accordingly.
  * 
  * Do not need to specify both sets, can work in the case that only one of 
  * them is given.
  * 
  * N is the global number of points and m is the number of points owned 
  * by this node.
  */
template<class TKernel>
int compute_adaptive_id(TKernel& kernel, 
  std::vector<double>& source_coords, int num_cols,
  std::vector<double>& near_coords, int num_near,
  std::vector<double>& unif_coords, int num_unif, int dim, long N, long m,
  std::vector<lapack_int>& skeleton_out, std::vector<double>& proj, IDWorkspace& workspace,
  double epsilon, int max_rank, bool printout, std::vector<int>& source_inds);

  /**
   * This version is a simplified heuristic for adaptively choosing the rank.
   * 
   * The flag do_absolute controls whether we normalize by the estimated first
   * singular value before checking the rank cutoff -- i.e. it controls 
   * whether we should interpret epsilon as an absolute or relative condition
   * on the singular values of the matrix.
   *
   * absolute_scale -- in the absolute case, we try to scale the estimated 
   * singular values in order to account for the rows and columns we aren't 
   * looking at.
   *  
   * We compute the QR factorization.  We then let the rank s be the smallest 
   * s such that fabs(r_ss / r_11) < epsilon
   */
int compute_adaptive_id_simplified(double* A, int num_rows, int num_cols, 
  std::vector<lapack_int>& skeleton_out, std::vector<double>& proj, IDWorkspace& workspace,
  double epsilon, int max_rank, bool do_absolute, double absolute_scale,
  bool printout, double& solve_for_proj_time, double& qr_time);

  /**
   * We attempt to overcome the fact that the above estimator doesn't prune
   * aggressively enough when the kernel is sharp.  We compute an estimate of 
   * the minimum near field contrfibution, then include this in our rank 
   * estimator.
   * 
   * We choose the minimum rank such that: 
   * r_{rr} * node_size / (near_scale * leaf_size + r_11 * node_size)
  */
int compute_adaptive_id_scale_near(double* A, int num_rows, int num_cols, 
  std::vector<lapack_int>& skeleton_out, std::vector<double>& proj, IDWorkspace& workspace,
  double epsilon, int max_rank, bool printout, 
  int leaf_size, long node_size, double near_scale, double& solve_for_proj_time, double& qr_time);



} // namespace 


#include "id_impl.hpp"

#endif 


