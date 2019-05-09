
#include <mpi.h>

#include "askit_alg.hpp"
#include "gaussian_kernel.hpp"
#include "laplace_kernel.hpp"
#include "polynomial_kernel.hpp"
#include "kernel_inputs.hpp"
#include "test_id.hpp"

#include <CmdLine.h>
#include <ompUtils.h>
#include <ctime>
#include <omp.h>
#include <float.h>
#include <limits>

// #include "generator.h"

using namespace askit;

/*
void SmallTest()
{
  // Create a small data set
  int dim = 2;
  int N = 12;

  std::vector<double> charge_vec(N, 1.0);
	
  fksData data;
  
  data.dim = dim;
  data.numof_points = N;
  data.X.resize(N*dim);
  data.gids.resize(N);
  
  // These points clearly form four groups, so all the nearest neighbors should be located in the same group
  
  data.X[0] = 0.0;
  data.X[1] = 0.0;
  data.X[2] = 1.0;
  data.X[3] = 0.0;
  data.X[4] = 0.0;
  data.X[5] = 1.0;
  
  data.X[6] = 10.0;
  data.X[7] = 10.0;
  data.X[8] = 11.0;
  data.X[9] = 10.0;
  data.X[10] = 10.0;
  data.X[11] = 11.0;
  
  data.X[12] = 20.0;
  data.X[13] = 20.0;
  data.X[14] = 21.0;
  data.X[15] = 20.0;
  data.X[16] = 20.0;
  data.X[17] = 21.0;
  
  data.X[18] = 30.0;
  data.X[19] = 30.0;
  data.X[20] = 31.0;
  data.X[21] = 30.0;
  data.X[22] = 30.0;
  data.X[23] = 31.0;
  
  for (int i = 0; i < N; i++)
  {
    data.gids[i] = (long)i;
  }
  
  data.charges = charge_vec;
  
  double h = 1.0;
  int num_skel_targets = 5;
  int id_rank = 2;
  int max_points_per_node = 3;
  int max_tree_level = 10;
  int min_comm_size_per_node = 1;
  int num_neighbors_in = 3;
  int num_neighbor_iterations = 2;
  int min_skeleton_level = 2;
  
  int oversampling_factor = 3;
  int num_uniform_required = 0;
  
  KernelInputs inputs;
  inputs.bandwidth = h;
  inputs.do_variable_bandwidth = false;
  
  const char* knn_file = "matlab_test_neighbors.txt";  
  
  AskitInputs ainputs(num_skel_targets, id_rank, max_points_per_node, max_tree_level, 
    num_neighbors_in, min_skeleton_level, oversampling_factor, num_uniform_required,
    knn_file);
  
  AskitAlg<GaussianKernel> driver(&data, inputs, ainputs);


  vector<double> target_coords(dim);
  for (int i = 0; i < dim; i++)
  {
    target_coords[i] = data.X[i];
  }

  cout << "\n\n =============== Small ASKIT Test =============== \n\n";

  vector<double> exact = driver.ComputeDirect(target_coords);
  std::cout << "Exact result: " << exact[0] << "\n";

  std::vector<double> all_approx = driver.ComputeAll();
  double approx = all_approx[0];
  std::cout << "Approx result: " << approx << "\n";

  vector<long> test_gids(1, 0);
  vector<int> test_lids(1, 0);
  vector<int> num_test_ids(1, 1);
  vector<int> displ(1, 0);

  vector<double> nn_only_vec = driver.ComputeDistNN(test_gids, test_lids, num_test_ids, displ);
  double nn_only = nn_only_vec[0];
  std::cout << "NN result: " << nn_only << "\n";
  
  double approx_err = fabs(approx - exact[0])/fabs(exact[0]);
  double nn_err = 
  
  AskitAlg<GaussianKernel> fmm_driver(&data, inputs, ainputs);
  
  vector<double> fmm_approx = fmm_driver.ComputeFMM();
  cout << "FMM result: " << fmm_approx[0] << "\n";
  
  
}
*/

/*
void LargerTest()
{
  
  int dim = 2;
  int N = 100;
  
  int num_skel_targets = 6;
  int id_rank = 3;
  int max_points_per_node = 5;
  int num_neighbors_in = 2;
  int min_skeleton_level = 2;
  int num_neighbor_iterations = 10;
  int oversampling_factor = 5;
  int max_tree_level = 10;
  int num_uniform_required = 0;
  
  double h = 1.0;
  
  fksData big_data;
  
  big_data.dim = dim;
  big_data.numof_points = N;
  big_data.X.resize(N * dim);
  big_data.gids.resize(N);
  
  for (int i = 0; i < N * dim; i++)
  {
    big_data.X[i] = rand() / (double)RAND_MAX;
  }
  
  for (int i = 0; i < N; i++)
  {
    big_data.gids[i] = (long)i;
  }

  std::vector<double> big_charges(N, 1.0);
  big_data.charges = big_charges;

  std::cout << "\n\nLarger test case: \n";

  KernelInputs inputs;
  inputs.bandwidth = h;
  
  const char* knn_file = "matlab_test_neighbors.txt";  
  AskitInputs big_inputs(num_skel_targets, id_rank, max_points_per_node, max_tree_level, 
    num_neighbors_in, min_skeleton_level, oversampling_factor, num_uniform_required,
    knn_file);
    
  AskitAlg<GaussianKernel> big_driver(&big_data, inputs, big_inputs);


  vector<double> target_coords(dim);
  for (int i = 0; i < dim; i++)
  {
    target_coords[i] = big_data.X[i];
  }

  std::cout << "Computing results\n";
  vector<double> exact_u = big_driver.ComputeDirect(target_coords);
  double exact = exact_u[0];
  std::cout << "Exact result: " << exact << "\n";

  std::vector<double> all_approx = big_driver.ComputeAll();
  double approx = all_approx[0];
  std::cout << "Approx result: " << approx << "\n";

  vector<long> test_gids(1, 0);
  vector<int> test_lids(1, 0);
  vector<int> num_test_ids(1, 1);
  vector<int> displ(1, 0);

  vector<double> nn_only_vec = big_driver.ComputeDistNN(test_gids, test_lids, num_test_ids, displ);
  double nn_only = nn_only_vec[0];
  std::cout << "NN result: " << nn_only << "\n";

  std::cout << "Using " << num_neighbors_in << " neighbors on " << N << " points in " << dim << " dimensions.\n";
  std::cout << "FMM relative error: " << fabs(exact - approx) / exact << "\n";
  std::cout << "NN relative error: " << fabs(exact - nn_only) / exact << "\n";
  
} // Big test
*/

void MatlabGaussianKernel(fksData& matlab_data)
{
  
  cout << "\n\n === Running MatlabGaussianKernel test === \n\n";
  
  int d = 5;
  int N = 1024;
  
  //num_neighbors_in = 4;
  int num_neighbors_in = 32;
  
  
  int max_points_per_node = 8;
  int max_tree_level = 30;
  
  int num_neighbor_iterations = 1;
  int rkdt_mppn = 1024;
  double h = 0.353553390593274;
  int num_skel_targets = 48; 
  int oversampling_factor = 5;
  int num_uniform_required = 0;
  
  int id_rank = 16;
  int min_skeleton_level = 1;
  
  int query_id = 0;
  
  // k = 4
//  double nn_res_matlab = 0.034468753581914;
//  double exact_res_matlab = 0.034671557465992;
//  double fmm_res_matlab = 0.034512459823961;
  
  // These come from the matlab implementation
  // k = 32
  double fmm_res_matlab = 0.034671556689793;
  double nn_res_matlab = 0.034671512661305;
  double exact_res_matlab = 0.034671557465992;
  
  // std::cout << "\n\nMatlab test case: \n";
  
  KernelInputs inputs;
  inputs.bandwidth = h;
  // just do exact neighbor search like in the matlab case
  
  const char* knn_file = "matlab_test_neighbors.txt";  
  
  AskitInputs matlab_inputs(num_skel_targets, id_rank, max_points_per_node, max_tree_level, 
    num_neighbors_in, min_skeleton_level, oversampling_factor, num_uniform_required,
    knn_file);
  
  AskitAlg<GaussianKernel> matlab_driver(&matlab_data, inputs, matlab_inputs);

  // std::cout << "Using " << num_neighbors_in << " neighbors on " << N << " points in " << d << " dimensions.\n";
  
  
  vector<double> target_coords(d);
  for (int i = 0; i < d; i++)
  {
    target_coords[i] = matlab_data.X[query_id * d + i];
  }
  
  //std::cout << "Computing results on matlab data:\n";
  vector<double> exact_u = matlab_driver.ComputeDirect(target_coords);
  double exact = exact_u[0];
  // std::cout << "Exact result: " << exact << "\n";

  vector<long> test_gids(1, 0);
  vector<int> test_lids(1, 0);
  vector<int> num_test_ids(1, 1);
  vector<int> displ(1, 0);

  vector<double> nn_only_vec = matlab_driver.ComputeDistNN(test_gids, test_lids, num_test_ids, displ);
  double nn_only = nn_only_vec[0];
  // std::cout << "NN result: " << nn_only << "\n";
  
  std::vector<double> all_approx = matlab_driver.ComputeAll();
  // std::cout << "All approx result: " << all_approx[query_id] << "\n";

  matlab_inputs.do_fmm = true;
  AskitAlg<GaussianKernel> fmm_driver(&matlab_data, inputs, matlab_inputs);
  vector<double> fmm_approx = fmm_driver.ComputeAll();
  // cout << "FMM approx result: " << fmm_approx[query_id] << "\n";
  
  double askit_error = fabs(fmm_res_matlab - all_approx[query_id]) / fmm_res_matlab;
  double nn_error = fabs(nn_res_matlab - nn_only) / nn_res_matlab;
  double exact_error = fabs(exact_res_matlab - exact) / exact_res_matlab;
  double askit_fmm_error = fabs(fmm_approx[query_id] - fmm_res_matlab) / fmm_res_matlab;
  
  // std::cout << "\nERRORS RELATIVE TO MATLAB IMPLEMENTATION FOR GAUSSIAN KERNEL:\n";
  // std::cout << "ASKIT all output: " << fabs(fmm_res_matlab - all_approx[query_id]) / fmm_res_matlab << "\n";
  // std::cout << "NN only output: " << fabs(nn_res_matlab - nn_only) / nn_res_matlab << "\n";
  // std::cout << "Exact computation output: " << fabs(exact_res_matlab - exact) / exact_res_matlab << "\n";
  // cout << "FMM approx output: " << fabs(fmm_approx[query_id] - fmm_res_matlab) / fmm_res_matlab << "\n";

  double eps = 1e-8;
  
  if (askit_error > eps)
  {
    cout << "!!!ASKIT Gaussian Test FAILED!!!\n\n";
    exit(1);
  }
  if (nn_error > eps)
  {
    cout << "!!!NN Gaussian Test FAILED!!!\n\n";
    exit(1);
  }
  if (exact_error > eps)
  {
    cout << "!!!Exact Gaussian Test FAILED!!!\n\n";
    exit(1);
  }
  if (askit_fmm_error > eps)
  {
    cout << "!!!FMM Gaussian Test FAILED!!!\n\n";
    exit(1);
  }
  
  cout << "\n\tMatlab Gaussian Test Passed\n\n";
    
}

void TestAdaptiveId(fksData& matlab_data)
{
  
  std::cout << "\n\n === Testing Adaptive Rank Algorithm === \n\n";
  
  int d = 5;
  int N = 1024;
  
  int num_neighbors_in = 1;
  
  int max_points_per_node = 8;
  int max_tree_level = 30;
  
  int num_neighbor_iterations = 1;
  int rkdt_mppn = 1024;
  
  // sigma = 2 in matlab
  double h = sqrt(2.0); // is 2/sqrt(2) 
  
  int oversampling_factor = 5;
  int num_uniform_required = 0;
  
  int id_rank = 500;
  // this is now additive for adaptive rank
  int num_skel_targets = 20; 
  int min_skeleton_level = 2;
  
  int query_id = 0;
    
  // k = 1
  double fmm_res_matlab = 2.675365217038062;
  double nn_res_matlab = 0.031250000000000;
  double exact_res_matlab = 2.675291092508136;
  
  std::cout << "\n\nMatlab test case (adaptive rank): \n";
  
  KernelInputs inputs;
  inputs.bandwidth = h;
  // just do exact neighbor search like in the matlab case
  
  const char* knn_file = "matlab_test_neighbors.txt";  
  
  AskitInputs matlab_inputs(num_skel_targets, id_rank, max_points_per_node, max_tree_level, 
    num_neighbors_in, min_skeleton_level, oversampling_factor, num_uniform_required,
    knn_file);

  matlab_inputs.use_adaptive_id = true;
  matlab_inputs.id_tol = 0.01;
  
  AskitAlg<GaussianKernel> matlab_driver(&matlab_data, inputs, matlab_inputs);

  // std::cout << "Using " << num_neighbors_in << " neighbors on " << N << " points in " << d << " dimensions.\n";

  vector<double> target_coords(d);
  for (int i = 0; i < d; i++)
  {
    target_coords[i] = matlab_data.X[query_id * d + i];
  }
  
  //std::cout << "Computing results on matlab data:\n";
  vector<double> exact_u = matlab_driver.ComputeDirect(target_coords);
  double exact = exact_u[0];
  // std::cout << "Exact result: " << exact << "\n";

  vector<long> test_gids(1, 0);
  vector<int> test_lids(1, 0);
  vector<int> num_test_ids(1, 1);
  vector<int> displ(1, 0);

  vector<double> nn_only_vec = matlab_driver.ComputeDistNN(test_gids, test_lids, num_test_ids, displ);
  double nn_only = nn_only_vec[0];
  // std::cout << "NN result: " << nn_only << "\n";
  
  std::vector<double> all_approx = matlab_driver.ComputeAll();
  // std::cout << "All approx result: " << all_approx[query_id] << "\n";
  
  // std::cout << "\nERRORS RELATIVE TO MATLAB IMPLEMENTATION FOR GAUSSIAN KERNEL:\n";
//   std::cout << "FMM all output: " << fabs(fmm_res_matlab - all_approx[query_id]) / fmm_res_matlab << "\n";
//   std::cout << "NN only output: " << fabs(nn_res_matlab - nn_only) / nn_res_matlab << "\n";
//   std::cout << "Exact computation output: " << fabs(exact_res_matlab - exact) / exact_res_matlab << "\n";
  
  double askit_error = fabs(fmm_res_matlab - all_approx[query_id]) / fmm_res_matlab;
  double nn_error = fabs(nn_res_matlab - nn_only) / nn_res_matlab;
  double exact_error = fabs(exact_res_matlab - exact) / exact_res_matlab;
  
  // Large error because it's all uniform sampling in this test
  double eps = 0.5;
  
  if (askit_error > eps)
  {
    cout << "!!!ASKIT AdaptiveID Test FAILED -- error: " << askit_error << " !!!\n\n";
    exit(1);
  }
  if (nn_error > eps)
  {
    cout << "!!!NN AdaptiveID Test FAILED!!!\n\n";
    exit(1);
  }
  if (exact_error > eps)
  {
    cout << "!!!Exact AdaptiveID Test FAILED!!!\n\n";
    exit(1);
  }
  
  cout << "\n\tAdaptive ID Test Passed\n\n";
    
  
} // test adaptive rank



void TestSimplifiedAdaptiveId(fksData& matlab_data)
{
  
  std::cout << "\n\n === Testing Simplified Adaptive Rank Algorithm === \n\n";
  
  int d = 5;
  int N = 1024;
  
  int num_neighbors_in = 1;
  
  int max_points_per_node = 8;
  int max_tree_level = 30;
  
  int num_neighbor_iterations = 1;
  int rkdt_mppn = 1024;
  
  // sigma = 2 in matlab
  double h = sqrt(2.0); // is 2/sqrt(2) 
  
  int oversampling_factor = 5;
  int num_uniform_required = 0;
  
  int id_rank = 500;
  // this is now additive for adaptive rank
  int num_skel_targets = 20; 
  int min_skeleton_level = 2;
  
  int query_id = 0;
    
  // std::cout << "\n\nMatlab test case (adaptive rank): \n";
  
  KernelInputs inputs;
  inputs.bandwidth = h;
  // just do exact neighbor search like in the matlab case
  
  const char* knn_file = "matlab_test_neighbors.txt";  
  
  AskitInputs matlab_inputs(num_skel_targets, id_rank, max_points_per_node, max_tree_level, 
    num_neighbors_in, min_skeleton_level, oversampling_factor, num_uniform_required,
    knn_file);

  matlab_inputs.use_simplified_adaptive_id = true;
  matlab_inputs.id_tol = 0.01;
  
  AskitAlg<GaussianKernel> matlab_driver(&matlab_data, inputs, matlab_inputs);

  // std::cout << "Using " << num_neighbors_in << " neighbors on " << N << " points in " << d << " dimensions.\n";
  
  vector<double> target_coords(d);
  for (int i = 0; i < d; i++)
  {
    target_coords[i] = matlab_data.X[query_id * d + i];
  }
  
  //std::cout << "Computing results on matlab data:\n";
  vector<double> exact_u = matlab_driver.ComputeDirect(target_coords);
  double exact = exact_u[0];
  // std::cout << "Exact result: " << exact << "\n";

  std::vector<double> all_approx = matlab_driver.ComputeAll();
  // std::cout << "All approx result: " << all_approx[query_id] << "\n";
  
  double askit_error = fabs(all_approx[query_id] - exact) / fabs(exact);

  double eps = 0.5;
  
  if (askit_error > eps)
  {
    cout << "!!!SimplifiedAdaptiveID Test FAILED!!!\n\n";
    exit(1);
  }
  
  cout << "\n\tSimplified Adaptive ID Test Passed\n\n";
    
  
  
} // test adaptive rank



void MatlabVariableGaussianKernel(fksData& matlab_data)
{

  std::cout << "\n\n === Testing Variable Bandwidth Kernel === \n\n";

  int d = 5;
  int N = 1024;
  
  //num_neighbors_in = 4;
  int num_neighbors_in = 32;
  
  int max_points_per_node = 8;
  int max_tree_level = 30;
  
  int num_neighbor_iterations = 1;
  int rkdt_mppn = 1024;
  double h = 0.353553390593274;
  int num_skel_targets = 48; 
  int oversampling_factor = 5;
  int num_uniform_required = 0;
  
  int id_rank = 16;
  int min_skeleton_level = 1;
  
  int query_id = 0;
  
  KernelInputs inputs;
  inputs.bandwidth = h;
  inputs.do_variable_bandwidth = true;

  const char* knn_file = "matlab_test_neighbors.txt";  
  
  AskitInputs var_inputs(num_skel_targets, id_rank, max_points_per_node, max_tree_level, 
    num_neighbors_in, min_skeleton_level, oversampling_factor, num_uniform_required,
    knn_file);
  
  AskitAlg<GaussianKernel> var_driver(&matlab_data, inputs, var_inputs);
    
  vector<double> target_coords(d);
  for (int i = 0; i < d; i++)
  {
    target_coords[i] = matlab_data.X[query_id * d + i];
  }
    
  //std::cout << "Computing results on matlab data:\n";
  vector<double> exact_u = var_driver.ComputeDirect(target_coords);
  double exact = exact_u[0];
  // std::cout << "Exact result: " << exact << "\n";

  vector<long> test_gids(1, 0);
  vector<int> test_lids(1, 0);
  vector<int> num_test_ids(1, 1);
  vector<int> displ(1, 0);

  vector<double> nn_only_vec = var_driver.ComputeDistNN(test_gids, test_lids, num_test_ids, displ);
  double nn_only = nn_only_vec[0];
  // std::cout << "NN result: " << nn_only << "\n";

  std::vector<double> all_approx = var_driver.ComputeAll();
  // std::cout << "All approx result: " << all_approx[query_id] << "\n";
  
  double var_fmm_res_matlab = 0.061096189252078;
  //double var_fmm_res_matlab = 0.061096520913874;
  double var_nn_res_matlab = 0.060702469730113;
  double var_exact_res_matlab = 0.061113503411325;
  
  double askit_error = fabs(var_fmm_res_matlab - all_approx[query_id]) / var_fmm_res_matlab;
  double nn_error = fabs(var_nn_res_matlab - nn_only) / var_nn_res_matlab;
  double exact_error = fabs(var_exact_res_matlab - exact) / var_exact_res_matlab;

  double eps = 1e-4;
  
  if (askit_error > eps)
  {
    cout << "!!!ASKIT Variable Kernel Test FAILED!!!\n\n";
    exit(1);
  }
  if (nn_error > eps)
  {
    cout << "!!!NN Variable Kernel Test FAILED!!!\n\n";
    exit(1);
  }
  if (exact_error > eps)
  {
    cout << "!!!Exact Variable Kernel Test FAILED!!!\n\n";
    exit(1);
  }
  
  cout << "\n\tVariable Kernel Test Passed\n\n";
    
}

void MatlabLaplaceKernel(fksData& matlab_data)
{
  
  std::cout << "\n\n === Testing Laplace Kernel === \n\n";
  
  int d = 5;
  int N = 1024;
  
  KernelInputs laplace_inputs;
  
  int num_neighbors_in = 32;
  int id_rank = 16;
  int max_points_per_node = 8;
  int num_skel_targets = 2*id_rank;
  int min_skeleton_level = 1;
  int max_tree_level = 30;
  int oversampling_factor = 5;
  int num_uniform_required = 0;
  
  int query_id = 0;

  const char* knn_file = "matlab_test_neighbors.txt";  
  
  AskitInputs laplace_ainputs(num_skel_targets, 
    id_rank, max_points_per_node, max_tree_level, 
    num_neighbors_in, min_skeleton_level, oversampling_factor, num_uniform_required,
    knn_file);
  
  AskitAlg<LaplaceKernel> laplace_driver(&matlab_data, laplace_inputs, laplace_ainputs);
    
  std::vector<double> all_approx = laplace_driver.ComputeAll();
  double approx = all_approx[query_id];
  // std::cout.precision(15);
  // std::cout << "Approx result: " << approx << "\n";

  // Results from MATLAB code
  // double laplace_exact_res_matlab = 0.015155690730995;
  double laplace_fmm_res_matlab = 0.011635114625272;
  // double laplace_nn_res_matlab = 0.004096005631006;
  
  double eps = 1e-10;
  double askit_error = fabs(laplace_fmm_res_matlab - approx) / laplace_fmm_res_matlab;
  
  if (askit_error > eps)
  {
    cout << "!!!ASKIT Laplace Kernel Test FAILED!!!\n\n";
    exit(1);
  }
  
  cout << "\n\tLaplace Kernel Test Passed\n\n";
  
}

void MatlabPolynomialKernel(fksData& matlab_data)
{
  
  std::cout << "\n\n === Testing Polynomial Kernel === \n\n";
  
  int d = 5;
  int N = 1024;
  
  KernelInputs polynomial_inputs;
  polynomial_inputs.bandwidth = 0.5;
  polynomial_inputs.constant = 1.0;
  polynomial_inputs.power = 2.0;
  
  int num_neighbors_in = 32;
  int id_rank = 16;
  int max_points_per_node = 8;
  int num_skel_targets = 32;
  int min_skeleton_level = 1;
  int max_tree_level = 30;
  int oversampling_factor = 5;
  int num_uniform_required = 0;
  
  int query_id = 0;

  const char* knn_file = "matlab_test_neighbors.txt";  
  
  AskitInputs poly_ainputs(num_skel_targets, 
    id_rank, max_points_per_node, max_tree_level, 
    num_neighbors_in, min_skeleton_level, oversampling_factor, num_uniform_required,
    knn_file);
  
  AskitAlg<PolynomialKernel> polynomial_driver(&matlab_data, polynomial_inputs, poly_ainputs);
    
  std::vector<double> all_approx = polynomial_driver.ComputeAll();
  double approx = all_approx[query_id];
  // std::cout << "All approx result: " << all_approx[query_id] << "\n";

  // Results from MATLAB code
  // double polynomial_exact_res_matlab = 1155.730501010907346;
  double polynomial_fmm_res_matlab = 1176.158468180620503;
  // double polynomial_nn_res_matlab = 196.840536642936570;
  
  double askit_error = fabs(polynomial_fmm_res_matlab - approx) / polynomial_fmm_res_matlab;
  
  double eps = 1e-14;
  
  if (askit_error > eps)
  {
    cout << "!!!ASKIT Polynomial Kernel Test FAILED!!!\n\n";
    exit(1);
  }
  
  cout << "\n\tPolynomial Kernel Test Passed\n\n";
  
}

void TestUpdateCharges(fksData& matlab_data)
{
 
  std::cout << "\n\n === Testing UpdateCharges === \n\n";

 
  int d = 5;
  int N = 1024;
  
  //num_neighbors_in = 4;
  int num_neighbors_in = 32;
  
  
  int max_points_per_node = 8;
  int max_tree_level = 30;
  
  int num_neighbor_iterations = 1;
  int rkdt_mppn = 1024;
  double h = 0.353553390593274;
  int num_skel_targets = 48; 
  int oversampling_factor = 5;
  int num_uniform_required = 0;
  
  int id_rank = 16;
  int min_skeleton_level = 1;
  
  KernelInputs inputs;
  inputs.bandwidth = h;
  // just do exact neighbor search like in the matlab case

  const char* knn_file = "matlab_test_neighbors.txt";  
  
  AskitInputs matlab_inputs(num_skel_targets, id_rank, max_points_per_node, max_tree_level, 
    num_neighbors_in, min_skeleton_level, oversampling_factor, num_uniform_required,
    knn_file);

  std::vector<double> q = matlab_data.charges;
  
  AskitAlg<GaussianKernel> update_driver(&matlab_data, inputs, matlab_inputs);

  std::vector<double> u = update_driver.ComputeAll();
  
  // double all of the charges
  for (int i = 0; i < u.size(); i++)
  {
    q[i] *= 2.0;
  }
  
  update_driver.UpdateCharges(q);
  
  std::vector<double> u2 = update_driver.ComputeAll();

  double ratio = u2[0] / u[0];
  
  if (fabs(ratio - 2.0) > 1e-15)
  {
    cout << "!!!ASKIT Update Charges Test FAILED!!!\n\n";
    exit(1);
  }
  
  cout << "\n\tUpdate Charges Test Passed\n\n";
   
}

/*
void TestProportionalNumTargets(fksData& matlab_data)
{
  
  // We'll run compute all and compute some separately, then make sure that 
  // the results are the same
  
  int d = 5;
  int N = 1024;
  
  //num_neighbors_in = 4;
  int num_neighbors_in = 32;
  
  int max_points_per_node = 8;
  int max_tree_level = 30;
  
  double h = 0.353553390593274;
  int num_skel_targets = 2; 
  int oversampling_factor = 5;
  int num_uniform_required = 0;
  
  int id_rank = 16;
  int min_skeleton_level = 2;
  
  int num_queries = 5;
  std::vector<int> query_inds(num_queries);
  query_inds[0] = 0;
  query_inds[1] = 100;
  query_inds[2] = 75;
  query_inds[3] = 22;
  query_inds[4] = 1000;
  
  std::cout << "\n\nProportional Num Targets test case: \n";
  
  KernelInputs inputs;
  inputs.bandwidth = h;
  // just do exact neighbor search like in the matlab case
  
  const char* knn_file = "matlab_test_neighbors.txt";  
  
  AskitInputs matlab_inputs(num_skel_targets, id_rank, max_points_per_node, max_tree_level, 
    num_neighbors_in, min_skeleton_level, oversampling_factor, num_uniform_required,
    knn_file);
  
  matlab_inputs.do_absolute_num_targets = false;
  
  AskitAlg<GaussianKernel> matlab_driver(&matlab_data, inputs, matlab_inputs);

  std::vector<double> approx = matlab_driver.ComputeAll();
  

  vector<double> target_coords(d*num_queries);
  for (int i = 0; i < num_queries; i++)
  {
    for (int j = 0; j < d; j++)
    {
      target_coords[i*d + j] = matlab_data.X[query_inds[i] * d + j];      
    }
  }

  double avg_error = 0.0;
  double max_error = 0.0;
  
  vector<double> exact_u = matlab_driver.ComputeDirect(target_coords);
  
  for (int i = 0; i < num_queries; i++)
  {
    double exact = exact_u[i];
    
    std::cout << "Approx result: " << approx[query_inds[i]] << ", direct result: " << exact << "\n";
    double this_error = fabs(approx[query_inds[i]] - exact) / exact;
    avg_error += this_error;
    max_error = std::max(max_error, this_error);
  }
  avg_error /= (double)num_queries;

  std::cout << "\nERRORS RELATIVE TO COMPUTE ALL:\n";
  std::cout << "Max Error: " << max_error << "\n";
  std::cout << "Avg Error: " << avg_error << "\n";
  
}
*/

/*
void TestProportionalNumTargetsAdaptive(fksData& matlab_data)
{
  
  // We'll run compute all and compute some separately, then make sure that 
  // the results are the same
  
  int d = 5;
  int N = 1024;
  
  //num_neighbors_in = 4;
  int num_neighbors_in = 32;
  
  int max_points_per_node = 8;
  int max_tree_level = 30;
  
  double h = 0.353553390593274;
  int num_skel_targets = 2; 
  int oversampling_factor = 2;
  int num_uniform_required = 0;
  
  int id_rank = 20;
  int min_skeleton_level = 2;
  
  int num_queries = 5;
  std::vector<int> query_inds(num_queries);
  query_inds[0] = 0;
  query_inds[1] = 100;
  query_inds[2] = 75;
  query_inds[3] = 22;
  query_inds[4] = 1000;
  
  std::cout << "\n\nProportional Num Targets ADAPTIVE test case: \n";
  
  KernelInputs inputs;
  inputs.bandwidth = h;
  // just do exact neighbor search like in the matlab case
  
  const char* knn_file = "matlab_test_neighbors.txt";  
  
  AskitInputs matlab_inputs(num_skel_targets, id_rank, max_points_per_node, max_tree_level, 
    num_neighbors_in, min_skeleton_level, oversampling_factor, num_uniform_required,
    knn_file);
  
  matlab_inputs.do_absolute_num_targets = false;
  matlab_inputs.use_simplified_adaptive_id = true;
  matlab_inputs.id_tol = 0.001;
  
  
  AskitAlg<GaussianKernel> matlab_driver(&matlab_data, inputs, matlab_inputs);

  std::vector<double> approx = matlab_driver.ComputeAll();
  
  vector<double> target_coords(d*num_queries);
  for (int i = 0; i < num_queries; i++)
  {
    for (int j = 0; j < d; j++)
    {
      target_coords[i*d + j] = matlab_data.X[query_inds[i] * d + j];      
    }
  }
  
  vector<double> exact_u = matlab_driver.ComputeDirect(target_coords);

  double avg_error = 0.0;
  double max_error = 0.0;
  for (int i = 0; i < num_queries; i++)
  {
    double exact = exact_u[i];
    
    std::cout << "Approx result: " << approx[query_inds[i]] << ", direct result: " << exact << "\n";
    double this_error = fabs(approx[query_inds[i]] - exact) / exact;
    avg_error += this_error;
    max_error = std::max(max_error, this_error);
  }
  avg_error /= (double)num_queries;

  std::cout << "\nERRORS RELATIVE TO COMPUTE ALL:\n";
  std::cout << "Max Error: " << max_error << "\n";
  std::cout << "Avg Error: " << avg_error << "\n";
  
}
*/

void TestSplitK(fksData& matlab_data)
{
  
  std::cout << "\n\n === Testing Split K Algorithm === \n\n";
  
  int d = 5;
  int N = 1024;
  
  int num_neighbors_in = 16;
  
  int max_points_per_node = 8;
  int max_tree_level = 30;
  
  int num_neighbor_iterations = 1;
  int rkdt_mppn = 1024;
  
  // sigma = 2 in matlab
  double h = sqrt(2.0); // is 2/sqrt(2) 
  
  int oversampling_factor = 5;
  int num_uniform_required = 0;
  
  int id_rank = 500;

  int num_skel_targets = 2; 
  int min_skeleton_level = 2;
  
  int query_id = 0;
    
  // std::cout << "\n\nMatlab test case (adaptive rank): \n";
  
  KernelInputs inputs;
  inputs.bandwidth = h;
  // just do exact neighbor search like in the matlab case
  
  const char* knn_file = "matlab_test_neighbors.txt";  
  
  bool do_split_k = true;
  
  
  AskitInputs matlab_inputs(num_skel_targets, id_rank, max_points_per_node, max_tree_level, 
    num_neighbors_in, min_skeleton_level, oversampling_factor, num_uniform_required,
    knn_file);

  matlab_inputs.use_simplified_adaptive_id = true;
  matlab_inputs.id_tol = 0.01;
  matlab_inputs.do_split_k = do_split_k;
  matlab_inputs.do_absolute_num_targets = false;
  
  AskitAlg<GaussianKernel> matlab_driver(&matlab_data, inputs, matlab_inputs);

  // std::cout << "Using " << num_neighbors_in << " neighbors on " << N << " points in " << d << " dimensions.\n";
  
  vector<double> target_coords(d);
  for (int i = 0; i < d; i++)
  {
    target_coords[i] = matlab_data.X[query_id * d + i];
  }
  
  //std::cout << "Computing results on matlab data:\n";
  vector<double> exact_u = matlab_driver.ComputeDirect(target_coords);
  double exact = exact_u[0];
  // std::cout << "Exact result: " << exact << "\n";

  std::vector<double> all_approx = matlab_driver.ComputeAll();
  // std::cout << "All approx result: " << all_approx[query_id] << "\n";
  
  double askit_error = fabs(all_approx[query_id] - exact) / fabs(exact);

  double eps = 0.5;
  
  if (askit_error > eps)
  {
    cout << "!!!Split K Test FAILED!!!\n\n";
    exit(1);
  }
  
  cout << "\n\tSplit K Test Passed\n\n";
    
  
  
}



void TestAdaptiveLevelRestriction(fksData& matlab_data)
{
  
  std::cout << "\n\n === Testing Adaptive Level Restriction Algorithm === \n\n";
  
  int d = 5;
  int N = 1024;
  
  int num_neighbors_in = 16;
  
  int max_points_per_node = 8;
  int max_tree_level = 30;
  
  int rkdt_mppn = 1024;
  
  // sigma = 2 in matlab
  double h = sqrt(2.0); // is 2/sqrt(2) 
  
  int oversampling_factor = 5;
  int num_uniform_required = 0;
  
  int id_rank = 30;

  int num_skel_targets = 2; 
  int min_skeleton_level = 2;
  
  int query_id = 0;
    
  // std::cout << "\n\nMatlab test case (adaptive rank): \n";
  
  KernelInputs inputs;
  inputs.bandwidth = h;
  // just do exact neighbor search like in the matlab case
  
  const char* knn_file = "matlab_test_neighbors.txt";  
  
  bool do_split_k = true;
  
  
  AskitInputs matlab_inputs(num_skel_targets, id_rank, max_points_per_node, max_tree_level, 
    num_neighbors_in, min_skeleton_level, oversampling_factor, num_uniform_required,
    knn_file);

  matlab_inputs.use_simplified_adaptive_id = true;
  matlab_inputs.id_tol = 5e-2;
  matlab_inputs.do_split_k = do_split_k;
  matlab_inputs.do_absolute_num_targets = false;
  matlab_inputs.do_adaptive_level_restriction = true;
  
  AskitAlg<GaussianKernel> matlab_driver(&matlab_data, inputs, matlab_inputs);

  // std::cout << "Using " << num_neighbors_in << " neighbors on " << N << " points in " << d << " dimensions.\n";
  
  vector<double> target_coords(d);
  for (int i = 0; i < d; i++)
  {
    target_coords[i] = matlab_data.X[query_id * d + i];
  }
  
  //std::cout << "Computing results on matlab data:\n";
  vector<double> exact_u = matlab_driver.ComputeDirect(target_coords);
  double exact = exact_u[0];
  // std::cout << "Exact result: " << exact << "\n";

  std::vector<double> all_approx = matlab_driver.ComputeAll();
  // std::cout << "All approx result: " << all_approx[query_id] << "\n";
  
  double askit_error = fabs(all_approx[query_id] - exact) / fabs(exact);

  double eps = 0.5;
  
  if (askit_error > eps)
  {
    cout << "!!!Adaptive Level Restriction Test FAILED!!!\n\n";
    exit(1);
  }
  
  cout << "\n\tAdaptive Level Restriction Test Passed\n\n";
    
  
  
} // TestAdaptive Level restriction


int main(int argc, char* argv[])
{
  
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

  ////////////////////////////////////////////////////////////
  
  int d = 5;
  int N = 1024;
  
  // Load the matlab data
  std::ifstream in("matlab_points.txt", std::ifstream::in);

  std::vector<double> matlab_ptr(d * N);

  for (int i = 0; i < d; i++) {
    for (int j = 0; j < N; j++) {
        in >> matlab_ptr[i + j * d];
    }
  }
  
  in.close();  

  std::ifstream charges_in("matlab_charges.txt");
  
  std::vector<double> matlab_charges(N);
  
  for (int i = 0; i < N; i++)
  {
    charges_in >> matlab_charges[i];
  }

  fksData matlab_data;
  
  matlab_data.dim = d;
  matlab_data.numof_points = N;
  matlab_data.X = matlab_ptr;
  
  matlab_data.gids.resize(N);
  for (int i = 0; i < N; i++)
  {
    matlab_data.gids[i] = (long)i;
  }
  
  matlab_data.charges = matlab_charges;
  
  // Comparison to matlab test code
  MatlabGaussianKernel(matlab_data);
  
  TestAdaptiveId(matlab_data);
  
  TestSimplifiedAdaptiveId(matlab_data);
  
  TestSplitK(matlab_data);
  
  // This test currently only works with the slow kernel evaluation code
#ifndef USE_KS
  MatlabVariableGaussianKernel(matlab_data);
#endif
    
  ////////////////////////////// Laplace Kernel ////////////////////////////
  
  MatlabLaplaceKernel(matlab_data);
  
  ////////////////////////////// Polynomial Kernel ////////////////////////////
  
  // MatlabPolynomialKernel(matlab_data);
  
  ////////////////////////// Update Charges ///////////////////////////
  
  TestUpdateCharges(matlab_data);
  
  //////////// New adaptive rank algorithm and adaptive level restriction
  
  TestAdaptiveLevelRestriction(matlab_data);
  
  ///////////////////////// New definition of ell -- proportional to num cols
  
  // TestProportionalNumTargets(matlab_data);

  // TestProportionalNumTargetsAdaptive(matlab_data);
  
  // These are covered by the tests above
  // TestGaussianKernel();
  //
  // TestID();
  //
  // TestLaplaceKernel();
  //
  // TestPolynomialKernel();
    
  cout << "\n\n\n===== ALL TESTS PASSED =====\n\n\n";
    
  MPI_Finalize();
	return 0;
  
}
