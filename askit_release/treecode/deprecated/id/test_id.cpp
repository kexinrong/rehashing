
#include "test_id.hpp"


double test_id_error(const double* A, int num_rows, int num_cols, int k) 
{
  
  std::vector<lapack_int> skeleton(k);
  std::vector<double> proj(k * (num_cols - k));
  
  // important because ID now overwrites it's input with QR stuff
  double* A_copy = new double[num_rows*num_cols];
  memcpy(A_copy, A, num_rows * num_cols * sizeof(double));

  IDWorkspace workspace(k, num_cols);

  compute_id(A_copy, num_rows, num_cols, k, skeleton, proj, workspace);
  
  // Acol is n x k
  double* Acol = new double[num_rows * k];
  // Acol = A(skeleton,:)
  
  for (int i = 0; i < num_rows; i++)
  {
    for (int j = 0; j < k; j++) {
      lapack_int col_ind = skeleton[j];
      Acol[i + j*num_rows] = A[i + col_ind * num_rows];  
    }
  }

  std::vector<lapack_int> sorted_skeleton(skeleton.begin(), skeleton.end());
  std::sort(sorted_skeleton.begin(), sorted_skeleton.end());
  
  double* A_nocol = new double[(num_cols - k) * num_rows];
  // now, build the matrix of non-skeleton entries
  for (int i = 0; i < num_rows; i++)
  {
    int skel_ind = 0;
    int nocol_ind = 0;
    for (int j = 0; j < num_cols; j++) {
      
      // don't look beyond the end of sorted skeleton
      if(skel_ind < k && sorted_skeleton[skel_ind] == j)
      {
        skel_ind++;
        continue;
      }

      A_nocol[i + nocol_ind*num_rows] = A[i + j * num_rows];
      nocol_ind++;

    }
    
  }
  
  // A_approx = Acol * proj
  // IMPORTANT: this is lacking the skeleton columns right now
  double* A_approx = new double[num_rows * (num_cols - k)];
  std::cout << "Rebuilding approx\n";
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, num_rows, num_cols - k, k, 1.0, Acol, num_rows, proj.data(), k, 0.0, A_approx, num_rows);

  std::cout << "Computing norms\n";
  // A_approx = A_approx - A_nocol
  cblas_daxpy(num_rows * (num_cols - k), -1.0, A_nocol, 1, A_approx, 1);
  
  
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
  std::cout << "\nRank " << k << " ID, F-norm error: " << error << "\n\n";  
  
  delete A_approx;
  delete A_nocol;
  delete Acol;
  delete A_copy;
  
  return error;
  
}