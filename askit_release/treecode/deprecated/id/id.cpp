
#include "id.hpp"

void print_mat(double* mat, int m, int n)
{
  
  print_mat(mat, m, n, m);
  
}

void print_mat(double* mat, int m, int n, int jump)
{
  
  std::cout << "\n";
  for (int i = 0; i < m; i++)
  {
    for (int j = 0; j < n; j++)
    {
      std::cout << mat[i + j * jump] << ", ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
  
}

void print_upper_triangular(double* mat, int m, int n)
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

int compute_id(double* A, int num_rows, int num_cols, int rank, std::vector<lapack_int>& skeleton_out, 
    std::vector<double>& proj, IDWorkspace& workspace)
{
  
  // assuming more rows than columns for now
  
  if (num_cols > workspace.max_cols || rank > workspace.max_rank)
  {
    std::cerr << "didn't allocate enough space for ID.\n";
    return -1;
  }
  
  std::cout << "Setting up\n";

  lapack_int lda = num_rows; // because it's col major
  
  // set all pivots to 0, this indicates that everything is available to be 
  // pivoted 
  lapack_int* skeleton = workspace.skeleton;
  for (int i = 0; i < num_cols; i++)
  {
    skeleton[i] = 0;
    std::cout << skeleton[i] << ", ";
  }
  
  std::cout << "\n";
  
  // scalar factors of elementary reflectors
  double* tau = workspace.tau;
  
  // Now, compute the pivoted QR
  std::cout << "Calling QR\n";
  lapack_int output = LAPACKE_dgeqp3(LAPACK_COL_MAJOR, num_rows, num_cols, A, lda, skeleton, tau);
  
  std::cout << "QR result: " << output << "\n";

  for (int i = 0; i < num_cols; i++)
  {
    std::cout << skeleton[i] << ", ";
  }
  
  std::cout << "\n";
  
  // now, we need to compute the min F-norm solution to the system
  // R_11 T - R_12
  
  // R_11 is now A_copy(1:rank, 1:rank)
  // R_12 is A_copy(1:rank, rank+1:end)

  // Have to extract the matrices for the triangular solve
  double* R_11 = workspace.R_11;
  //double* R_12 = new double[rank * (num_cols - rank)];
  double* R_12 = A + rank*num_rows;
  
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

  std::cout << "solving for matrix\n";
  // upper triangular, don't transpose, not unit triangular
  // Doing this version because the triangular solver doesn't do any regularization. I think this does another QR step, but it should be trivial since R_11 is already triangular
  lapack_int solve_output = LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', rank, rank, num_cols - rank, R_11, rank, R_12, num_rows);
  
  std::cout << "linear solve output: " << solve_output << "\n";
  //print_mat(R_12, rank, num_cols - rank, num_cols);
  
  
  // Lapack doesn't do zero indexing, so this is important
  //std::cout << "skeleton: ";
  for (int i = 0; i < num_cols; i++)
  {
    skeleton[i] = skeleton[i] - 1;
    std::cout << skeleton[i] << ", ";
  }
  std::cout << "\n";
  
  for (int i = 0; i < rank; i++)
  {
    skeleton_out[i] = skeleton[i];
  }
  

  // Need to come up with a less dumb way to do this
  std::vector<lapack_int> sorted_skel(skeleton + rank, skeleton + num_cols);
  std::sort(sorted_skel.begin(), sorted_skel.end());
  
  // Now, extract proj, because we'll store it in the tree
  // now, the ith column of proj should be the permuted ith column of R_12
  for (int i = 0; i < rank; i++)
  {
    for (int j = 0; j < num_cols - rank; j++)
    {
      
      // this is the column we have
      lapack_int r12_col = skeleton[j + rank];
      // this is where it goes in proj
      lapack_int r12_ind = std::find(sorted_skel.begin(), sorted_skel.end(), r12_col) - sorted_skel.begin();
      
      // proj(i,r12_ind) = R12(i,j)
      proj[i + r12_ind * rank] = R_12[i + j * num_rows];
      
    }
  }
  
  return rank;
  
}


