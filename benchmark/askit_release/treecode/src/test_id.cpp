
#include "test_id.hpp"


using namespace askit;


// builds the approximation to A from the ID
// returns it in A_approx
double askit::reconstruct_mat(int num_rows, int num_cols, int k, 
  const double* A, std::vector<lapack_int>& skeleton, std::vector<double>& proj)
{
  
  std::vector<double> Acol(num_rows * k);
  std::vector<double> A_nocol(num_rows * (num_cols - k));
  
  for (int i = 0; i < num_rows; i++)
  {
    // Iterate through skeleton
    for (int j = 0; j < k; j++)
    {
      Acol[i + j*num_rows] = A[i + skeleton[j]*num_rows];
    } // for j (in skeleton)
    
    for (int j = k; j < num_cols; j++)
    {
      A_nocol[i + (j-k)*num_rows] = A[i + skeleton[j]*num_rows];    
    } // for j (not in skeleton)

  } // for i
  

  // A_approx = Acol * proj
  std::vector<double> A_approx(num_rows * (num_cols - k));
  std::cout << "Rebuilding approx\n";

  int num_cols_minus_k = num_cols - k;
  double oned = 1.0;
  double zerod = 0.0;
  int one = 1;
  cblas_dgemm("N", "N", &num_rows, &num_cols_minus_k, 
      &k, &oned, Acol.data(), &num_rows, proj.data(), &k, &zerod, A_approx.data(), &num_rows);
      
  return compute_error(num_rows, num_cols, k, A_nocol, A_approx, A);
  
} // reconstruct mat

double askit::compute_error(int num_rows, int num_cols, int k,
    std::vector<double>& A_nocol, std::vector<double>& A_approx, const double* A)
{
  
  std::cout << "Computing norms\n";
  // A_approx = A_approx - A_nocol
  int size = num_rows * (num_cols - k);
  double minusone = -1.0;
  int one = 1;
  cblas_daxpy(&size, &minusone, A_nocol.data(), &one, A_approx.data(), &one);
  
  // doing F-norm for now
  // stupid Lapack aux routines aren't found, so do it manually
  double norm_A_approx = 0.0;
  for (int i = 0; i < num_rows * (num_cols - k); i++)
  {
    norm_A_approx += A_approx[i] * A_approx[i];
  }
  std::cout << "approx norm: " << norm_A_approx << "\n";

  double norm_A = 0.0;
  for (int i = 0; i < num_rows * num_cols; i++)
  {
    norm_A += A[i] * A[i];
  }
  std::cout << "exact norm: " << norm_A << "\n";
  
  double error = sqrt(norm_A_approx / norm_A);
  
  return error;
  
}

double askit::test_id_error(const double* A, int num_rows, int num_cols, int k) 
{
  
  std::vector<lapack_int> skeleton(num_cols);
  std::vector<double> proj(k * (num_cols - k));
  
  // important because ID now overwrites it's input with QR stuff
  double* A_copy = new double[num_rows*num_cols];
  memcpy(A_copy, A, num_rows * num_cols * sizeof(double));

  IDWorkspace workspace(k, num_cols);
  compute_id(A_copy, num_rows, num_cols, k, skeleton, proj, workspace);
  
  double error = reconstruct_mat(num_rows, num_cols, k, A, skeleton, proj);
  
  std::cout << "\nRank " << k << " ID, F-norm error: " << error << "\n\n";  
  
  delete A_copy;
  
  return error;
  
}



double askit::test_adaptive_id(const double* A, int num_rows, int num_cols, double epsilon, int max_rank)
{
  
  std::vector<lapack_int> skeleton;
  std::vector<double> proj;
  
  // important because ID now overwrites it's input with QR stuff
  double* A_copy = new double[ num_rows * num_cols ];
  memcpy(A_copy, A, num_rows * num_cols * sizeof(double));

  IDWorkspace workspace(max_rank, num_cols);
  
  int rank = compute_adaptive_id(A_copy, num_rows, num_cols, 
    skeleton, proj, workspace, epsilon, max_rank);
  
  std::cout << "Computed rank " << rank << " adaptive ID.\n";

  double error = reconstruct_mat(num_rows, num_cols, rank, A, skeleton, proj);  
  
  std::cout << "\nRank " << rank << " adaptive ID, F-norm error: " << error << "\n\n";  
  
  delete A_copy;
  
  return error;
  
} // test_adaptive_id



