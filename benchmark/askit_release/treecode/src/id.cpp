
#include "id.hpp"

using namespace askit;

void askit::print_mat(double* mat, int m, int n)
{
  
  // with fixed jump size
  print_mat(mat, m, n, m);
  
}

void askit::print_mat(double* mat, int m, int n, int jump)
{
  
  std::cout << "\n";
  for (int i = 0; i < m; i++)
  {
    for (int j = 0; j < n; j++)
    {
      // everything is column major
      std::cout << mat[i + j * jump] << ", ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
  
}

void askit::print_upper_triangular(double* mat, int m, int n)
{
  
  std::cout << "\n";
  for (int i = 0; i < m; i++)
  {
    for (int j = 0; j < i; j++)
    {
      std::cout << "0.0 , ";
    }
    for (int j = i; j < n; j++)
    {
      int mat_ind = i + j*(j+1)/2;
      std::cout << mat[mat_ind] << ", ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";  
}

// need this one because can't set default values for reference inputs in header
int askit::compute_id(double* A, int num_rows, int num_cols, int rank, std::vector<lapack_int>& skeleton, 
              std::vector<double>& proj, IDWorkspace& workspace)
{

  double proj_time = 0.0;
  double qr_time = 0.0;
  return compute_id(A, num_rows, num_cols, rank, skeleton, proj, workspace, proj_time, qr_time);

}


int askit::compute_id(double* A, int num_rows, int num_cols, int rank, 
    std::vector<lapack_int>& skeleton_out, 
    std::vector<double>& proj, IDWorkspace& workspace, double& solve_for_proj_time, double& qr_time)
{
  
  // assuming more rows than columns for now

  // do we need to bother? 
  if (rank < num_cols) 
  {
  
    // Check if we allocated enough space
    if (num_cols > workspace.tau.size())
    {
      std::cout << "Resizing ID workspace\n";
      workspace.tau.resize(num_cols);
    }
    if (rank * rank > workspace.R_11.size())
    {
      std::cout << "Resizing ID R_11 space.\n";
      workspace.R_11.resize(rank * rank);
    }
    
    lapack_int lda = num_rows; // because it's col major
  
    // set all pivots to 0, this indicates that everything is available to be 
    // pivoted 
    skeleton_out.resize(num_cols);
    for (int i = 0; i < num_cols; i++)
    {
      skeleton_out[i] = 0;
    }
  
    // scalar factors of elementary reflectors
    double* tau = workspace.tau.data();
  
    double qr_start = omp_get_wtime();
  
    // Now, compute the pivoted QR
    lapack_int output = LAPACKE_dgeqp3(LAPACK_COL_MAJOR, num_rows, num_cols, A, 
        lda, skeleton_out.data(), tau);

    qr_time = omp_get_wtime() - qr_start;
  
    double proj_start = omp_get_wtime();

    int success = solve_for_proj(A, num_rows, num_cols, rank, skeleton_out, proj, workspace);

    solve_for_proj_time = omp_get_wtime() - proj_start;
    
    return rank;

  }
  else {
    // we're not compressing at all
    skeleton_out.resize(num_cols);
    for (int i = 0; i < num_cols; i++)
    {
      skeleton_out[i] = i;
    }
  
    return num_cols;
  }
} // compute_id


lapack_int askit::solve_for_proj(double* A, int num_rows, int num_cols, int rank, 
  std::vector<lapack_int>& skeleton_out, std::vector<double>& proj, IDWorkspace& workspace)
{
  // now, we need to compute the min F-norm solution to the system
  // R_11 proj - R_12 -- where R_11 is the top left rank x rank submatrix of R
  // and R_12 is the top right rank x (num_cols - rank) submatrix
  
  // TODO: can I avoid creating a new R_11 here? 
  // Just set the below diagonal to zero and use it in place? 
  
  double* R_11 = workspace.R_11.data();

  double* R_12 = A + rank*num_rows;

  // Fill in R_11
  for (int i = 0; i < rank; i++)
  {
    for (int j = 0; j < i; j++)
    {
      R_11[i + j * rank] = 0.0;
    }
    for (int j = i; j < rank; j++)
    {
      R_11[i + j * rank] = A[i + j * num_rows];
    }
  }

  lapack_int solve_output = LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', rank, rank, 
      num_cols - rank, R_11, rank, R_12, num_rows);

  if (solve_output != 0)
  {
  
    std::cout << "DGELS fail (output: " << solve_output << "), rank: " << rank << " ncols: " << num_cols << " nrows: " << num_rows << "\n";

    std::cout << "R_11:\n";
    print_mat(R_11, rank, rank);
    std::cout << "R_12:\n";
    print_mat(R_12, rank, num_cols - rank, num_rows);
  
  }

  // Lapack doesn't do zero indexing, so this is important
  for (int i = 0; i < num_cols; i++)
  {
    skeleton_out[i] = skeleton_out[i] - 1;
  }

  //std::cout << "allocating proj matrix\n";

  proj.resize(rank * (num_cols - rank));
  // Now, copy out into proj
  for (int i = 0; i < rank; i++)
  {
    for (int j = 0; j < num_cols - rank; j++)
    {
    
      proj[i + j * rank] = R_12[i + j * num_rows];
    
    } // loop over columns
  } // loop over rows

  return solve_output;
  
} // solve_for_proj



// epsilon is the tolerance in the diagonal entries of R_11
// max_rank is the largest possible rank we can return
int askit::compute_adaptive_id(double* A, int num_rows, int num_cols, 
  std::vector<lapack_int>& skeleton_out, std::vector<double>& proj, IDWorkspace& workspace,
  double epsilon, int max_rank, bool printout)
{

  // avoid square roots later
  double epsilon_sqr = epsilon * epsilon;

  // Check if we allocated enough space
  if (num_cols > workspace.tau.size())
  {
    std::cout << "Resizing ID workspace\n";
    workspace.tau.resize(num_cols);
  }
  if (max_rank * max_rank > workspace.R_11.size())
  {
    std::cout << "Resizing ID R_11 space.\n";
    workspace.R_11.resize(max_rank * max_rank);
  }
  
  lapack_int lda = num_rows; // because it's col major
  
  // set all pivots to 0, this indicates that everything is available to be 
  // pivoted 
  skeleton_out.resize(num_cols);
  for (int i = 0; i < num_cols; i++)
  {
    skeleton_out[i] = 0;
  }
  
  // scalar factors of elementary reflectors
  double* tau = workspace.tau.data();
  
  //std::cout << "doing QR\n";
  
  // Now, compute the pivoted QR
  lapack_int output = LAPACKE_dgeqp3(LAPACK_COL_MAJOR, num_rows, num_cols, A, 
      lda, skeleton_out.data(), tau);

  // number of singular values
  int diag_size = std::min(num_rows, num_cols);
      
  if (printout)
  {
    std::cout << "Doing " << num_rows << " x " << num_cols << " adaptive ID.\n";
  }
  
  // Now, we need to examine the diagonal entries of R11 to determine the rank
  int rank = -1;
  for (int i = 0; i < diag_size-1 && i < max_rank; i++)
  {
    
    double r_ii = A[(i+1) + (i+1)*lda];

    if (printout)
      std::cout << "r_" << i+1 << ": " << r_ii << "; ";

    // this is the estimate of the singular value
    double sigma_sqr = r_ii * r_ii * (num_cols - i);
    // now, scale to account for the ID
    sigma_sqr *= 1.0 + num_cols * i * (num_cols - i);

    // if (printout)
    //   std::cout << "i: " << i << ", sigma: " << sigma_sqr << ", eps: " << epsilon_sqr << "\n";

    if (sigma_sqr < epsilon_sqr)
    {
      // add one because of zero indexing
      // this is the number of columns in the skeleton
      rank = i+1;
      break;
    }  
  
  } // loop over diagonal entries
  
  if (printout)
    std::cout << "\n\n";

  // we keep everything if it's small enough
  if (rank < 0 && diag_size < max_rank)
  {
    rank = diag_size;
  }

  // now, do the linear solve
  if (rank > 0)
    int success = solve_for_proj(A, num_rows, num_cols, rank, skeleton_out, proj, workspace);
  else {
    // if we failed, then we'll just force it
    rank = max_rank;
    int success = solve_for_proj(A, num_rows, num_cols, max_rank, skeleton_out, proj, workspace);
  }
  
  //std::cout << "returning rank " << rank << " adaptive ID.\n";
  
  return rank;

} // compute_adaptive_id

// The idea is to choose the adaptive rank with a simple heuristic
int askit::compute_adaptive_id_simplified(double* A, int num_rows, int num_cols, 
  std::vector<lapack_int>& skeleton_out, std::vector<double>& proj, IDWorkspace& workspace,
  double epsilon, int max_rank, bool do_absolute, double absolute_scale,
  bool printout, double& solve_for_proj_time, double& qr_time)
{

  // Check if we allocated enough space
  if (num_cols > workspace.tau.size())
  {
    std::cout << "Resizing ID workspace\n";
    workspace.tau.resize(num_cols);
  }
  if (max_rank * max_rank > workspace.R_11.size())
  {
    std::cout << "Resizing ID R_11 space.\n";
    workspace.R_11.resize(max_rank * max_rank);
  }
  
  lapack_int lda = num_rows; // because it's col major
  
  // set all pivots to 0, this indicates that everything is available to be 
  // pivoted 
  skeleton_out.resize(num_cols);
  for (int i = 0; i < num_cols; i++)
  {
    skeleton_out[i] = 0;
  }
  
  // scalar factors of elementary reflectors
  double* tau = workspace.tau.data();
  
  //std::cout << "doing QR\n";
  
  double qr_start = omp_get_wtime();
  // Now, compute the pivoted QR
  lapack_int output = LAPACKE_dgeqp3(LAPACK_COL_MAJOR, num_rows, num_cols, A, 
      lda, skeleton_out.data(), tau);
      
  qr_time = omp_get_wtime() - qr_start;

  // number of singular values
  int diag_size = std::min(num_rows, num_cols);
      
  // if (printout)
  // {
  //   std::cout << "Doing " << num_rows << " x " << num_cols << " adaptive ID.\n";
  // }
  
  // the do_relative flag controls whether we normalize by the estimated first
  // singular value
  double r_11 = fabs(A[0]);
  // otherwise, we'll scale the estimated singular value
  if (do_absolute)
    r_11 = 1.0 / absolute_scale;
  
  // Now, we need to examine the diagonal entries of R11 to determine the rank
  int rank = -1;
  for (int i = 0; i < diag_size-1 && i < max_rank; i++)
  {
    
    double r_ii = fabs(A[(i+1) + (i+1)*lda]);

    double ratio = r_ii / r_11;

    // if (printout)
//     {
//       std::cout << "r_" << i+1 << " / r_11: " << ratio << "\n";
//}
    if (printout)
    {
      std::cout << "i: " << i << ", r_ii: " << r_ii << ", r_11: " << r_11 << ", ratio: " << ratio << "\n";
    }
    

    if (ratio < epsilon)
    {
      // add one because of zero indexing
      // this is the number of columns in the skeleton
      rank = i+1;
      break;
    }  
  
  } // loop over diagonal entries
  
  if (printout)
    std::cout << "\n\n";

  // we keep everything if it's small enough
  if (rank < 0 && diag_size < max_rank)
  {
    rank = diag_size;
  }

  // now, do the linear solve
  double solve_for_proj_start = omp_get_wtime();
  if (rank > 0)
    int success = solve_for_proj(A, num_rows, num_cols, rank, skeleton_out, proj, workspace);
  else {
    // if we failed, then we'll just force it
    rank = max_rank;
    int success = solve_for_proj(A, num_rows, num_cols, max_rank, skeleton_out, proj, workspace);
  }

  solve_for_proj_time = omp_get_wtime() - solve_for_proj_start;
  
  
  return rank;
  
} // compute_adaptive_id_simplified()
    
    
// The idea is to choose the adaptive rank with a simple heuristic
int askit::compute_adaptive_id_scale_near(double* A, int num_rows, int num_cols, 
  std::vector<lapack_int>& skeleton_out, std::vector<double>& proj, IDWorkspace& workspace,
  double epsilon, int max_rank, bool printout, 
  int leaf_size, long node_size, double near_scale, double& solve_for_proj_time, double& qr_time)
{

  // Check if we allocated enough space
  if (num_cols > workspace.tau.size())
  {
    std::cout << "Resizing ID workspace\n";
    workspace.tau.resize(num_cols);
  }
  if (max_rank * max_rank > workspace.R_11.size())
  {
    std::cout << "Resizing ID R_11 space.\n";
    workspace.R_11.resize(max_rank * max_rank);
  }
  
  lapack_int lda = num_rows; // because it's col major
  
  // set all pivots to 0, this indicates that everything is available to be 
  // pivoted 
  skeleton_out.resize(num_cols);
  for (int i = 0; i < num_cols; i++)
  {
    skeleton_out[i] = 0;
  }
  
  // scalar factors of elementary reflectors
  double* tau = workspace.tau.data();
  
  //std::cout << "doing QR\n";
  
  double qr_start = omp_get_wtime();
  // Now, compute the pivoted QR
  lapack_int output = LAPACKE_dgeqp3(LAPACK_COL_MAJOR, num_rows, num_cols, A, 
      lda, skeleton_out.data(), tau);
      
  qr_time = omp_get_wtime() - qr_start;

  // number of singular values
  int diag_size = std::min(num_rows, num_cols);
      
  // if (printout)
  // {
  //   std::cout << "Doing " << num_rows << " x " << num_cols << " adaptive ID.\n";
  // }
  
  double r_11 = fabs(A[0]);
  
  
  // Now, we need to examine the diagonal entries of R11 to determine the rank
  int rank = -1;
  for (int i = 0; i < diag_size-1 && i < max_rank; i++)
  {
    
    double r_ii = fabs(A[(i+1) + (i+1)*lda]);

    double test_val = r_ii * node_size / (near_scale * leaf_size + r_11 * node_size);

    if (printout)
    {
      std::cout << "near_scale: " << near_scale << ", leaf_size: " << leaf_size << ", node_size: " << node_size << ", test_val: " << test_val;
      std::cout << ", r_ii: " << r_ii << ", r_11: " << r_11 << "\n";
    }

    if (test_val < epsilon)
    {
      // add one because of zero indexing
      // this is the number of columns in the skeleton
      rank = i+1;
      break;
    }  
  
  } // loop over diagonal entries
  
  if (printout)
    std::cout << "rank: " << rank << "\n\n";
  
  // we keep everything if it's small enough
  if (rank < 0 && diag_size < max_rank)
  {
    rank = diag_size;
  }

  // now, do the linear solve
  double solve_for_proj_start = omp_get_wtime();
  if (rank > 0)
    int success = solve_for_proj(A, num_rows, num_cols, rank, skeleton_out, proj, workspace);
  else {
    // if we failed, then we'll just force it
    rank = max_rank;
    int success = solve_for_proj(A, num_rows, num_cols, max_rank, skeleton_out, proj, workspace);
  }

  solve_for_proj_time = omp_get_wtime() - solve_for_proj_start;
  
  
  return rank;
  
} // compute_adaptive_id_simplified()
    




