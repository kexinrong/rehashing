

#ifndef ASKIT_ID_IMPL_HPP_
#define ASKIT_ID_IMPL_HPP_


template<class TKernel>
int askit::compute_adaptive_id(TKernel& kernel, 
std::vector<double>& source_coords, int num_cols,
std::vector<double>& near_coords, int num_near,
std::vector<double>& unif_coords, int num_unif, int dim, long N, long m,
std::vector<lapack_int>& skeleton_out, std::vector<double>& proj, IDWorkspace& workspace,
double epsilon, int max_rank, bool printout, std::vector<int>& source_inds)
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
  
  // Form the three matrices
  int num_rows = num_near + num_unif;
  std::vector<double> all_targets = near_coords;
  all_targets.insert(all_targets.end(), unif_coords.begin(), unif_coords.end());
  
  std::vector<double> K(num_rows * num_cols);
  kernel.Compute(all_targets.begin(), all_targets.end(), source_coords.begin(), source_coords.end(), dim, K, source_inds);
  

  // scalar factors of elementary reflectors
  // we just re-use the same one for all three factorizations because we 
  // don't ever use it
  double* tau = workspace.tau.data();
  
  // Compute QR for the whole matrix
  lapack_int lda = num_rows; // because it's col major
  
  // set all pivots to 0, this indicates that everything is available to be 
  // pivoted 
  skeleton_out.resize(num_cols);
  for (int i = 0; i < num_cols; i++)
  {
    skeleton_out[i] = 0;
  }
  
  // Now, compute the pivoted QR
  lapack_int output = LAPACKE_dgeqp3(LAPACK_COL_MAJOR, num_rows, num_cols, K.data(), 
      lda, skeleton_out.data(), tau);

  double sigma_1 = sqrt(K[0] * K[0] * num_cols);

  if (printout)
    std::cout << "K[0]: " << K[0] << ", K[1,1]: " << K[1 + lda] << "\n"; 

  
  double sigma_f;  
  if (num_unif > 0 && num_near > 0) {

    std::vector<double> K_f(num_unif * num_cols);
    kernel.Compute(unif_coords.begin(), unif_coords.end(), source_coords.begin(), source_coords.end(), dim, K_f, source_inds);
    std::vector<lapack_int> skel_f(num_cols, 0);
  
    lapack_int output = LAPACKE_dgeqp3(LAPACK_COL_MAJOR, num_unif, num_cols, K_f.data(), 
        num_unif, skel_f.data(), tau);

    sigma_f = sqrt(K_f[0] * K_f[0] * num_cols);
  
  }
  else if (num_unif > 0) {
    sigma_f = sigma_1;
  }
  else {
    sigma_f = 0.0;
  }
  
  
  double sigma_n;  
  if (num_near > 0 && num_unif > 0) {

    std::vector<double> K_n(num_near * num_cols);
    kernel.Compute(near_coords.begin(), near_coords.end(), source_coords.begin(), source_coords.end(), dim, K_n, source_inds);
    std::vector<lapack_int> skel_n(num_cols, 0);
  
    lapack_int output = LAPACKE_dgeqp3(LAPACK_COL_MAJOR, num_near, num_cols, K_n.data(), 
        num_near, skel_n.data(), tau);

    // using the same estimate as below
    sigma_n = sqrt(K_n[0] * K_n[0] * num_cols);
  
  }
  else if (num_near > 0) {
    sigma_n = sigma_1;
  }
  else {
    sigma_n = 0.0;
  }
  
  // number of singular values
  int diag_size = std::min(num_rows, num_cols);
      
  if (printout)
  {
    std::cout << "Doing " << num_rows << " x " << num_cols << " adaptive ID.\n";
    std::cout << "N: " << N << ", m: " << m << ", num_near: " << num_near << ", num_unif: " << num_unif << "\n";
  }
  
  
  double scaling_factor = (sigma_n / sigma_1) + (sigma_f / sigma_1) * sqrt(((double)(N - m - num_near) / num_unif));
  scaling_factor = scaling_factor * scaling_factor;
  
  if (printout)
  {
    std::cout << "Sigma_n: " << sigma_n << ", Sigma_f: " << sigma_f << ", Sigma: " << sigma_1 << ", scaling: " << scaling_factor << ", epsilon: " << epsilon << "\n";
  }
  
  // Now, we need to examine the diagonal entries of R11 to determine the rank
  int rank = -1;
  for (int i = 0; i < diag_size-1 && i < max_rank; i++)
  {
    
    double r_ii = K[(i+1) + (i+1)*lda];

    // this is the estimate of the singular value
    double sigma_sqr = r_ii * r_ii * (num_cols - i);
    
    // this attempts to correct for the effects of sampling rows 
    sigma_sqr *= scaling_factor;
    
    if (printout)
      std::cout << "sigma_sqr_" << i+1 << ": " << sigma_sqr << "; ";

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
    int success = solve_for_proj(K.data(), num_rows, num_cols, rank, skeleton_out, proj, workspace);
  else {
    // if we failed, then we'll just force it
    rank = max_rank;
    int success = solve_for_proj(K.data(), num_rows, num_cols, max_rank, skeleton_out, proj, workspace);
  }
  
  //std::cout << "returning rank " << rank << " adaptive ID.\n";
  
  return rank;

}


#endif
