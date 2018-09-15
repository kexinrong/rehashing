
#include <mpi.h>

#include "askit_alg.hpp"
#include "gaussian_kernel.hpp"
#include "laplace_kernel.hpp"
#include "polynomial_kernel.hpp"
#include "kernel_inputs.hpp"
#include "askit_utils.hpp"

#include <CmdLine.h>
#include <ompUtils.h>
#include <ctime>
#include <omp.h>
#include <float.h>
#include <parallelIO.h>
#include <iostream>
#include <limits>

using namespace Torch;
using namespace askit;


/**
 * This class exists to store error estimates and timers between iterations
 * (different charge vectors).
 */
class ErrorCounters
{
  
public:
  
  // pointwise relative error
  double avg_fmm_err; 
  double max_fmm_err;
  
  // pointwise absolute error
  double avg_abs_err;
  double max_abs_err;
  
  // accuracy of neighbor evaluations only
  double avg_nn_err;
  
  // the l2 error \|u_ex - u_askit \|_2 / \|u_ex\|_2
  double l2_error;
  
  // same error, but only for contributions from nearest neighbors
  double nn_l2_error;
  
  // l1 norm of the charges
  double charge_l1_norm;
  
  // l2 norm of the charges
  double charge_l2_norm_sqr;

  // \|u(T) - \approxu(T) \|_2 * (N / |T|) / \|w\|_2
  double error_per_charge;
  
  double avg_potential;

  // |u_ex - u_askit|_\infty / |u_ex|_\infty
  // averaged over independent choices of charges
  double rel_inf_err;
  
  /////////////////////////
  
  // cooridinates of the target points
  vector<double> test_coordinates;
  
  // gids of target points owned by this rank
  vector<long> my_test_gids;
  
  // lids of target points owned by this rank
  vector<int> my_test_lids;
  
  // number of target points owned by each rank
  vector<int> num_test_lids;
  
  // displacements of target points owned by each rank (scan)
  vector<int> displ;
  
  // flag for whether we're doing test points or training points
  bool do_test_points;
  
  
  ErrorCounters(vector<double>& coords_in, vector<long>& gids_in,
    vector<int>& lids_in, vector<int>& num_ids_in, vector<int>& displ_in, 
    bool do_test_in)
    :
  avg_fmm_err(0.0),
  max_fmm_err(0.0),
  avg_abs_err(0.0),
  max_abs_err(0.0),
  avg_nn_err(0.0),
  l2_error(0.0),
  nn_l2_error(0.0),
  avg_potential(0.0),
  rel_inf_err(0.0),
  error_per_charge(0.0),
  test_coordinates(coords_in),
  my_test_gids(gids_in),
  my_test_lids(lids_in),
  num_test_lids(num_ids_in),
  displ(displ_in),
  do_test_points(do_test_in)
  {}
  
}; // class ErrorCounters




// Computes the error estimate from exact and NN-only computations
// 
// * potentials -- the output of ComputeAll() or ComputeFMM() for each 
// MPI rank, this function will re-shuffle them
// * num_error_checks -- the total number of points for which we will compute
// the exact potential in order to estimate the error
// * query_gids -- The global IDs of the error estimation points
// * alg -- The askit algorithm class
// * do_all -- (deprecated, always true now)
// * counters -- The error counters class 
template<class AlgClass>
void EstimateErrors(vector<double>& potentials, int num_error_checks, 
  vector<long>& query_gids, AlgClass& alg, 
  bool do_all, ErrorCounters& counters)
{

  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  double numerator_total = 0.0;
  double nn_numerator_total = 0.0;
  double denominator_total = 0.0;

  double avg_fmm_error = 0.0;
  double max_fmm_error = 0.0;
  double avg_nn_error = 0.0;
  double avg_abs_error = 0.0;
  double max_abs_error = 0.0;
  
  double rel_inf_err = 0.0;
  
  double max_potential = 0.0;
  
  double avg_potential = 0.0;

  double l2_error = 0.0;
  double nn_l2_error = 0.0;

  double charge_l1_norm = 0.0;
  double charge_l2_norm_sqr = 0.0;
  
  double error_per_charge = 0.0; 

  // Are we estimating error
  if (num_error_checks > 0) {

    // compute these potentials exactly
    // Only rank 0 collects the potentials
    
    // Order doesn't matter here -- its handled by the order of test coordinates
    std::vector<double> exact_potentials = alg.ComputeDirect(counters.test_coordinates);
    
    // These need to worry about order
    vector<double> nn_potentials;
    vector<double> approx_potentials;
    
    if (counters.do_test_points)
    {
      nn_potentials = alg.ComputeTestDistNN(counters.my_test_gids, counters.my_test_lids,
                                            counters.num_test_lids, counters.displ);
      // nn_potentials.resize(num_error_checks, 0.0);
    }
    else {
      nn_potentials = alg.ComputeDistNN(counters.my_test_gids, counters.my_test_lids, 
        counters.num_test_lids, counters.displ);
    }
    
    approx_potentials = alg.CollectPotentials(potentials, counters.my_test_gids, 
      counters.my_test_lids, counters.num_test_lids, counters.displ);

    double my_charge_l1_norm = alg.charge_l1_norm;
    double my_charge_l2_norm_sqr = alg.charge_l2_norm_sqr;
    
    double charge_l1_norm;
    MPI_Reduce(&my_charge_l1_norm, &charge_l1_norm, 1, 
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double charge_l2_norm_sqr;
    MPI_Reduce(&my_charge_l2_norm_sqr, &charge_l2_norm_sqr, 1, 
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
               
    // the total number of source points
    // long my_N = alg.N;
    // long global_N;
    // MPI_Reduce(&my_N, &global_N, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
               
    long global_N = alg.global_N;
               
    // vector<double> exact_potentials(num_error_checks, 1.0);
    // vector<double> nn_potentials(num_error_checks, 2.0);
    // vector<double> approx_potentials(num_error_checks, 3.0);

    // root process checks errors
    if (rank == 0)
    {

      cout<<"num_error_checks = "<<num_error_checks<<endl;
      cout<<"query_gids.size = "<<query_gids.size()<<endl;
      cout<<"exact_potentials.size = "<<exact_potentials.size()<<endl;
      cout<<"potentials.size = "<<potentials.size()<<endl;
      cout <<"Charge L1 Norm: " << charge_l1_norm << "\n";
      cout <<"Charge L2 Norm Squared: " << charge_l2_norm_sqr << "\n";
      cout << "Global N: " << global_N << "\n";

      for (int i = 0; i < num_error_checks; i++)
      {
        double exact_potential = exact_potentials[i];
        double nn_potential = nn_potentials[i];
        
        double approx_potential = approx_potentials[i];

        avg_potential += exact_potential;
        
        max_potential = max(max_potential, fabs(exact_potential));

        denominator_total += exact_potential * exact_potential;
        numerator_total += (approx_potential - exact_potential) * (approx_potential - exact_potential);
        nn_numerator_total += (nn_potential - exact_potential) * (nn_potential - exact_potential);
        
        avg_fmm_error += fabs(approx_potential - exact_potential) / fabs(exact_potential);
        max_fmm_error = max(max_fmm_error, fabs(approx_potential - exact_potential) / fabs(exact_potential));
        avg_nn_error += fabs(nn_potential - exact_potential) / fabs(exact_potential);

        avg_abs_error += fabs(approx_potential - exact_potential);
        max_abs_error = max(max_abs_error, fabs(approx_potential - exact_potential));
        
        error_per_charge += (approx_potential - exact_potential) * (approx_potential - exact_potential);

        printf("gid: %ld \t\t exact: %.8f \t\t approx: %.8f \t\t nn: %.8f \n", 
          query_gids[i], exact_potential, approx_potential, nn_potential);

      } // loop over error checks

      avg_fmm_error /= num_error_checks;
      avg_nn_error /= num_error_checks;
      avg_abs_error /= num_error_checks;
      
      // error_per_charge /= charge_l1_norm;
      // error_per_charge /= num_error_checks;

      // use this as an estimate of the contribution 
      cout << "\nRaw error per charge: " << error_per_charge << "\n";
      cout << "global N: " << global_N << "\n";
      cout << "num checks: " << num_error_checks << "\n";
      cout << "charge l2 norm sqr: " << charge_l2_norm_sqr << "\n\n";

      error_per_charge = sqrt(error_per_charge * (double)global_N / (double)num_error_checks / charge_l2_norm_sqr);
      
      
      // denominator_total +=1;
      l2_error = sqrt(numerator_total / denominator_total);
      nn_l2_error = sqrt(nn_numerator_total / denominator_total);
      
      rel_inf_err = max_abs_error / max_potential;

      if (counters.do_test_points)
        std::cout << "\n\nTEST ERRORS:\t (Averaged over " << num_error_checks << " queries).\n";
      else
        std::cout << "\n\nERRORS:\t (Averaged over " << num_error_checks << " queries).\n";
      
      std::cout << "ASKIT Avg Pointwise Relative error: " << avg_fmm_error << "\n";
      std::cout << "ASKIT Max Pointwise Relative error: " << max_fmm_error << "\n";
      std::cout << "ASKIT Avg Absolute error: " << avg_abs_error << "\n";
      std::cout << "ASKIT Max Absolute error: " << max_abs_error << "\n";
      std::cout << "ASKIT L2 Relative error: " << l2_error << "\n";
      cout << "ASKIT Error per charge: " << error_per_charge << "\n";
      cout << "Relative Inf error: " << rel_inf_err << "\n";
      std::cout << "NN Relative error: " << avg_nn_error << "\n";
      cout << "NN L2 Error: " << nn_l2_error << "\n";

      avg_potential /= num_error_checks;
      std::cout << "\nAverage size of potential: " << avg_potential << "\n";

      // fill in the outputs
      counters.avg_fmm_err += avg_fmm_error;
      counters.max_fmm_err += max_fmm_error;
      counters.avg_abs_err += avg_abs_error;
      counters.max_abs_err += max_abs_error;
      counters.avg_nn_err += avg_nn_error;
      counters.avg_potential += avg_potential;
      counters.l2_error += l2_error;
      counters.nn_l2_error += nn_l2_error;
      counters.error_per_charge += error_per_charge;
      counters.rel_inf_err += rel_inf_err;
      counters.charge_l1_norm += charge_l1_norm;
      counters.charge_l2_norm_sqr += charge_l2_norm_sqr;

    } // only the root computes errors

    MPI_Barrier(MPI_COMM_WORLD);

  } // more than one error check

} // Compute error


// Prints out the Timers and Error estimates
//
// * alg -- the algorithm class
// * askit_inputs -- the inputs used to build the algorithm class
// * kernel_inputs -- the inputs used for the kernel function
// * output_filename -- The filename for the printed results output
// * counters -- An ErrorCounters class used to collect the info between 
// iterations over indepenent charge vectors
// * num_error_checks -- the number of points used to estimate the error
// * num_error_repeats -- the number of independent charge vectors used to 
// estimate the error
// * filename -- the file containing the data points
// * knn_filename -- the file containing the KNN info
// * charge_filename -- the file containing the input charges, or a label 
// such as 'ones' or 'norm'
// * N -- number of source points on this MPI rank
// * d -- the dimension of the data
// * kernel_type -- type of kernel function
// * comp_option -- deprecated
template<class AlgClass>
void OutputTimes(AlgClass& alg, AskitInputs& askit_inputs, 
  KernelInputs& kernel_inputs, const char* output_filename,
  ErrorCounters& counters, int num_error_checks, int num_error_repeats,
  const char* filename, const char* knn_filename, const char* charge_filename,
  int N, int d, const char* kernel_type, const char* comp_option)
{
  
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  
  long global_total_skeleton_size;
  double global_avg_skeleton_size;
  int global_max_skeleton_size, global_min_skeleton_size;
  long global_total_num_skeletons;
  long global_downward_evals;
  
  long global_num_nodes_with_uniform;

  long total_skeleton_size = 0;
  int max_skeleton_size = 0;
  int min_skeleton_size = 0;
  long num_skeletons = 0;
  int num_targets = alg.N;
  long global_num_targets;
  
  // Sum the number of kernel evaluations per thread into one counter
  long num_downward_evals;

  double tree_build_time, skeletonization_time, list_blocking_time; 
  double evaluation_time, let_traversal_time, test_evaluation_time, exact_comp_time;
  double test_list_blocking_time;
  double update_charges_time;
  

#pragma omp parallel for reduction(+:num_skeletons)
  for (int i = 0; i < alg.self_skeleton_sizes.size(); i++)
  {
    if (alg.self_skeleton_sizes[i].first > 0)
      num_skeletons++;
  }

  // needs separate loops because omp doesn't like two different reduce 
  // operations in the same loop
#pragma omp parallel for reduction(+:total_skeleton_size)
  for (int i = 0; i < alg.self_skeleton_sizes.size(); i++)
  {
    total_skeleton_size += alg.self_skeleton_sizes[i].first;
  }

  // needs separate loops because omp doesn't like two different reduce 
  // operations in the same loop
#pragma omp parallel for reduction(max:max_skeleton_size)
  for (int i = 0; i < alg.self_skeleton_sizes.size(); i++)
  {
    max_skeleton_size = max(max_skeleton_size, alg.self_skeleton_sizes[i].first);
  }
  
#pragma omp parallel for reduction(min:min_skeleton_size)
  for (int i = 0; i < alg.self_skeleton_sizes.size(); i++)
  {
    min_skeleton_size = min(min_skeleton_size, alg.self_skeleton_sizes[i].first);
  }
  
  // This should be valid, since we've already performed the computation
  int num_levels = alg.tree->depth_let;
  
  // Now, collect the skeleton sizes per level
  vector<int> min_skeleton_size_per_level(num_levels, INT_MAX);
  vector<double> avg_skeleton_size_per_level(num_levels, 0);
  vector<int> num_skeleton_size_per_level(num_levels, 0);
  vector<int> max_skeleton_size_per_level(num_levels, 0);

  vector<int> num_unprunable_nodes_per_level(num_levels, 0);

  // Note that I don't assume anything about the number per level to allow for 
  // adaptive level restriction later on

  // cout << "\nnum levels " << num_levels << "\n\n";

  // Not bothering with parallel -- this is just post-processing and small
  for (int i = 0; i < alg.self_skeleton_sizes.size(); i++)
  {
    
    int level = alg.self_skeleton_sizes[i].second;
    
    if (level >= 0)
    {
    
      int skel_size = alg.self_skeleton_sizes[i].first;
    
      // cout << "skeleton level " << level << " size " << skel_size << "\n";

      min_skeleton_size_per_level[level] = min(min_skeleton_size_per_level[level], skel_size);
      avg_skeleton_size_per_level[level] += skel_size;
    
      if (skel_size > 0)
        num_skeleton_size_per_level[level]++;
      else 
        num_unprunable_nodes_per_level[level]++;
    
      max_skeleton_size_per_level[level] = max(max_skeleton_size_per_level[level], skel_size);
    
    }
  
  } // loop for per level skeleton statistics

  // Now, reduce them
  vector<int> global_min_skeleton_size_per_level(num_levels, 0);
  vector<double> global_avg_skeleton_size_per_level(num_levels, 0);
  vector<int> global_num_skeleton_size_per_level(num_levels, 0);
  vector<int> global_max_skeleton_size_per_level(num_levels, 0);
  vector<int> global_num_unprunable_nodes_per_level(num_levels,0);

  MPI_Reduce(min_skeleton_size_per_level.data(), global_min_skeleton_size_per_level.data(), num_levels, 
    MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
  
  MPI_Reduce(avg_skeleton_size_per_level.data(), global_avg_skeleton_size_per_level.data(), num_levels, 
    MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  
  MPI_Reduce(num_skeleton_size_per_level.data(), global_num_skeleton_size_per_level.data(), num_levels, 
    MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  
  MPI_Reduce(max_skeleton_size_per_level.data(), global_max_skeleton_size_per_level.data(), num_levels, 
    MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

  MPI_Reduce(num_unprunable_nodes_per_level.data(), global_num_unprunable_nodes_per_level.data(), num_levels, 
    MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0)
  {
    for (int i = 0; i < num_levels; i++)
    {
      if (global_num_skeleton_size_per_level[i] > 0)
        global_avg_skeleton_size_per_level[i] /= global_num_skeleton_size_per_level[i];
    }
  }

  // Now, collect stats
  // Sum the number of kernel evaluations per thread into one counter
  num_downward_evals = alg.num_downward_kernel_evals;

  long num_nodes_with_uniform = alg.num_nodes_with_uniform;

  MPI_Barrier(MPI_COMM_WORLD);
  
  MPI_Reduce(&num_skeletons, &global_total_num_skeletons, 1, 
             MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Reduce(&num_downward_evals, &global_downward_evals, 1,
             MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Reduce(&total_skeleton_size, &global_total_skeleton_size, 1,
             MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Reduce(&max_skeleton_size, &global_max_skeleton_size, 1, MPI_INT,
             MPI_MAX, 0, MPI_COMM_WORLD);

  MPI_Reduce(&min_skeleton_size, &global_min_skeleton_size, 1, MPI_INT,
            MPI_MIN, 0, MPI_COMM_WORLD);

  MPI_Reduce(&num_nodes_with_uniform, &global_num_nodes_with_uniform, 1,
             MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
             
  MPI_Reduce(&num_targets, &global_num_targets, 1, 
             MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  
  tree_build_time = alg.tree_build_time;
  skeletonization_time = alg.skeletonization_time;
  list_blocking_time = alg.list_blocking_time;
  evaluation_time = alg.evaluation_time;
  test_evaluation_time = alg.test_evaluation_time;
  exact_comp_time = alg.exact_comp_time;
  let_traversal_time = alg.let_traversal_time;
  test_list_blocking_time = alg.test_list_blocking_time;
  // average over 
  update_charges_time = alg.update_charges_time / (double)num_error_repeats;

  // collect the extra timers for every rank
  double all_construct_tree_list_time;
  double all_compute_leaf_neighbors_time;
  double all_merge_neighbor_lists_time;
  double all_collect_local_samples_time;
  double all_collect_neighbor_coords_time;
  double all_distributed_upward_pass_time;
  double all_uniform_sample_sibling_time;
  double all_dist_compute_skeleton_time;
  double all_merge_skeletons_of_kids_time;
  double all_merge_nn_list_time;
  double all_get_dist_node_excl_nn_time;
  double all_compute_adaptive_id_time;
  double all_apply_proj_time;
  double all_subsample_self_targets_time;
  double all_compute_skeleton_time;

  double all_solve_for_proj_time;
  double all_qr_time;
  double all_max_qr_time;
  double all_kernel_compute_time;
  double all_dist_test_set_time;
  double all_update_test_let_time;

  double all_pass_potentials_down_time;
  double all_merge_fmm_lists_basic_time;
  double all_merge_basic_set_difference_time;
  double all_fmm_add_to_map_time;
  
  double all_merge_fmm_lists_aggressive_time;
  double all_less_than_tree_order_time;
  double all_is_ancestor_time;
  double all_split_node_time;
  double all_merge_tree_list_full_time;
  
  double all_compute_near_scale_time;
  
  double all_frontier_exchange_time;
  double all_prune_by_knn_time;

  MPI_Reduce(&(alg.construct_tree_list_time),  
    &all_construct_tree_list_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&(alg.compute_leaf_neighbors_time),  
    &all_compute_leaf_neighbors_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&(alg.merge_neighbor_lists_time),  
    &all_merge_neighbor_lists_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&(alg.collect_local_samples_time),  
    &all_collect_local_samples_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&(alg.collect_neighbor_coords_time),  
    &all_collect_neighbor_coords_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&(alg.distributed_upward_pass_time),  
    &all_distributed_upward_pass_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&(alg.uniform_sample_sibling_time),  
    &all_uniform_sample_sibling_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&(alg.dist_compute_skeleton_time),  
    &all_dist_compute_skeleton_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&(alg.merge_skeletons_of_kids_time),  
    &all_merge_skeletons_of_kids_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&(alg.merge_nn_list_time),  
    &all_merge_nn_list_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&(alg.get_dist_node_excl_nn_time),  
    &all_get_dist_node_excl_nn_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&(alg.compute_adaptive_id_time),  
    &all_compute_adaptive_id_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&(alg.apply_proj_time),  
    &all_apply_proj_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&(alg.subsample_self_targets_time),  
    &all_subsample_self_targets_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&(alg.compute_skeleton_time),  
    &all_compute_skeleton_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&(alg.solve_for_proj_time),  
    &all_solve_for_proj_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&(alg.qr_time),  
    &all_qr_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&(alg.max_qr_time),  
    &all_max_qr_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&(alg.kernel_compute_time),  
    &all_kernel_compute_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&(alg.dist_test_set_time),  
    &all_dist_test_set_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&(alg.update_test_let_time),  
    &all_update_test_let_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Reduce(&(alg.pass_potentials_down_time),  
    &all_pass_potentials_down_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&(alg.merge_fmm_lists_basic_time),  
    &all_merge_fmm_lists_basic_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&(alg.merge_basic_set_difference_time),  
    &all_merge_basic_set_difference_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&(alg.fmm_add_to_map_time),  
    &all_fmm_add_to_map_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Reduce(&(alg.merge_fmm_lists_aggressive_time),  
    &all_merge_fmm_lists_aggressive_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&(alg.less_than_tree_order_time),  
    &all_less_than_tree_order_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&(alg.is_ancestor_time),  
    &all_is_ancestor_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&(alg.split_node_time),  
    &all_split_node_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&(alg.merge_tree_list_full_time),  
    &all_merge_tree_list_full_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Reduce(&(alg.compute_near_scale_time),  
    &all_compute_near_scale_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Reduce(&(alg.frontier_exchange_time), &all_frontier_exchange_time, 1, 
    MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&(alg.prune_by_knn_time), &all_prune_by_knn_time, 1, 
    MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  
    

  // collect the extra timers for every rank
  all_construct_tree_list_time /= (double)size;
  all_compute_leaf_neighbors_time /= (double)size;
  all_merge_neighbor_lists_time /= (double)size;
  all_collect_local_samples_time /= (double)size;
  all_collect_neighbor_coords_time /= (double)size;
  all_distributed_upward_pass_time /= (double)size;
  all_uniform_sample_sibling_time /= (double)size;
  all_dist_compute_skeleton_time /= (double)size;
  all_merge_skeletons_of_kids_time /= (double)size;
  all_merge_nn_list_time /= (double)size;
  all_get_dist_node_excl_nn_time /= (double)size;
  all_compute_adaptive_id_time /= (double)size;
  all_apply_proj_time /= (double)size;
  all_subsample_self_targets_time /= (double)size;
  all_compute_skeleton_time /= (double)size;

  all_solve_for_proj_time /= (double)size;
  all_qr_time /= (double)size;
  all_max_qr_time /= (double)size;
  all_kernel_compute_time /= (double)size;
  all_dist_test_set_time /= (double)size;
  all_update_test_let_time /= (double)size;

  all_pass_potentials_down_time /= (double)size;
  all_merge_fmm_lists_basic_time /= (double)size;
  all_merge_basic_set_difference_time /= (double)size;
  all_fmm_add_to_map_time /= (double)size;

  all_merge_fmm_lists_aggressive_time /= (double)size;
  all_less_than_tree_order_time /= (double)size;
  all_is_ancestor_time /= (double)size;
  all_split_node_time /= (double)size;
  all_merge_tree_list_full_time /= (double)size;

  all_compute_near_scale_time /= (double)size;

  all_frontier_exchange_time /= (double)size;
  all_prune_by_knn_time /= (double)size;

  // rank 0 prints out stats, etc.
  if (rank == 0) {

    // average errors over iterations
    counters.avg_fmm_err /= (double)num_error_repeats;
    counters.max_fmm_err /= (double)num_error_repeats;
    counters.avg_abs_err /= (double)num_error_repeats;
    counters.max_abs_err /= (double)num_error_repeats;
    counters.l2_error /= (double)num_error_repeats;
    counters.nn_l2_error /= (double)num_error_repeats;
    counters.avg_nn_err /= (double)num_error_repeats;
    counters.avg_potential /= (double)num_error_repeats;
    counters.error_per_charge /= (double)num_error_repeats;
    counters.rel_inf_err /= (double)num_error_repeats;
    counters.charge_l1_norm /= (double)num_error_repeats;
    counters.charge_l2_norm_sqr /= (double)num_error_repeats;
    
    global_avg_skeleton_size = (double)global_total_skeleton_size / global_total_num_skeletons;

    std::cout << "\nSTATS: \n";
    std::cout << "Kernel evaluations in downward pass: " << (double)global_downward_evals / global_num_targets << "\n";
    std::cout << "Min. Skeleton Size: " << global_min_skeleton_size << "\n";
    std::cout << "Avg. Skeleton Size: " << global_avg_skeleton_size << "\n";
    std::cout << "Max. Skeleton Size: " << global_max_skeleton_size << "\n";
    // TODO: make this work in distributed case
    cout << "Leaf list size before FMM: " << (double)alg.leaf_list_size_before / N << "\n";
    cout << "Leaf list size after FMM: " << (double)alg.leaf_list_size_after / N << "\n";

    cout << "\nSkeleton sizes per level: \n";
    for (int i = askit_inputs.min_skeleton_level; i < num_levels; i++)
    {
    
      cout << "Level " << i << ": Min " << global_min_skeleton_size_per_level[i] << ", ";
      cout << "Avg: " << global_avg_skeleton_size_per_level[i] << ", ";
      cout << "Max: " << global_max_skeleton_size_per_level[i] << "\n";
      
    }
    

    std::cout << "\nRUNTIMES:\n";
    std::cout << "Tree construction and NN exchange: " << tree_build_time << " s.\n";
    std::cout << "LET construction, communication, and target list construction: " << let_traversal_time << " s.\n";
    std::cout << "Skeleton construction: " << skeletonization_time << " s.\n";
    std::cout << "Interaction List Blocking: " << list_blocking_time << " s.\n";
    std::cout << "Evaluation: " << evaluation_time << " s.\n";
    std::cout << "Test Evaluation: " << test_evaluation_time << " s.\n";
    std::cout << "Direct Comp. (on evaluation points): " << exact_comp_time << " s.\n";
    std::cout << "Test Interaction List Blocking: " << test_list_blocking_time << " s.\n";
    std::cout << "Update Charges time (average): " << update_charges_time << " s.\n";
    
    std::cout << "\nEXTRA TIMERS (Rank 0):\n";
    std::cout << "construct_tree_list_time: " << alg.construct_tree_list_time << " s.\n";
    std::cout << "compute_leaf_neighbors_time: " << alg.compute_leaf_neighbors_time << " s.\n";
    std::cout << "merge_neighbor_lists_time: " << alg.merge_neighbor_lists_time << " s.\n";
    std::cout << "collect_local_samples_time: " << alg.collect_local_samples_time << " s.\n";
    std::cout << "collect_neighbor_coords_time: " << alg.collect_neighbor_coords_time << " s.\n";
    std::cout << "distributed_upward_pass_time: " << alg.distributed_upward_pass_time << " s.\n";
    std::cout << "uniform_sample_sibling_time: " << alg.uniform_sample_sibling_time << " s.\n";
    std::cout << "dist_compute_skeleton_time: " << alg.dist_compute_skeleton_time << " s.\n";
    std::cout << "merge_skeletons_of_kids_time: " << alg.merge_skeletons_of_kids_time << " s.\n";
    std::cout << "merge_nn_list_time: " << alg.merge_nn_list_time << " s.\n";
    std::cout << "get_dist_node_excl_nn_time: " << alg.get_dist_node_excl_nn_time << " s.\n";
    std::cout << "compute_adaptive_id_time: " << alg.compute_adaptive_id_time << " s.\n";
    std::cout << "apply_proj_time: " << alg.apply_proj_time << " s.\n";
    std::cout << "subsample_self_targets_time: " << alg.subsample_self_targets_time << " s.\n";
    std::cout << "compute_skeleton_time: " << alg.compute_skeleton_time << " s.\n";

    cout << "solve_for_proj_time: " << alg.solve_for_proj_time << " s.\n";
    cout << "qr_time: " << alg.qr_time << " s.\n";
    cout << "max_qr_time: " << alg.max_qr_time << " s.\n";
    cout << "kernel_compute_time: " << alg.kernel_compute_time << " s.\n";

    cout << "dist_test_set_time: " << alg.dist_test_set_time << " s.\n";
    cout << "update_test_let_time: " << alg.update_test_let_time << " s.\n";

    cout << "pass_potentials_down_time: " << alg.pass_potentials_down_time << " s.\n";
    cout << "merge_fmm_lists_basic_time: " << alg.merge_fmm_lists_basic_time << " s.\n";
    cout << "merge_basic_set_difference_time: " << alg.merge_basic_set_difference_time << " s.\n";
    cout << "fmm_add_to_map_time: " << alg.fmm_add_to_map_time << " s.\n";
    
    cout << "merge_fmm_lists_aggressive_time: " << alg.merge_fmm_lists_aggressive_time << " s.\n";
    cout << "less_than_tree_order_time: " << alg.less_than_tree_order_time << " s.\n";
    cout << "is_ancestor_time: " << alg.is_ancestor_time << " s.\n";
    cout << "split_node_time: " << alg.split_node_time << " s.\n";
    cout << "merge_tree_list_full_time: " << alg.merge_tree_list_full_time << " s.\n";

    cout << "compute_near_scale_time: " << alg.compute_near_scale_time << " s.\n";

    cout << "frontier_exchange_time: " << alg.frontier_exchange_time << " s.\n";
    cout << "prune_by_knn_time: " << alg.prune_by_knn_time << " s.\n";

    // Now, output the data, etc.
    FILE* file = fopen(output_filename, "w");

    fprintf(file, "INPUTS: \n");
    fprintf(file, "-k %d\n", askit_inputs.num_neighbors_in);
    fprintf(file, "-fks_mppn %d\n", askit_inputs.max_points_per_node);
    fprintf(file, "-fks_mtl %d\n", askit_inputs.max_tree_level);
    fprintf(file, "-h %g\n", kernel_inputs.bandwidth);
    fprintf(file, "-c %g\n", kernel_inputs.constant);
    fprintf(file, "-p %g\n", kernel_inputs.power);
    fprintf(file, "-num_skel_targets %d\n", askit_inputs.num_skel_targets);
    fprintf(file, "-id_rank %d\n", askit_inputs.rank);
    fprintf(file, "-oversampling_fac %d\n", askit_inputs.oversampling_factor_in);
    fprintf(file, "-num_uniform_required %d\n", askit_inputs.num_uniform_required);
    fprintf(file, "-min_skeleton_level %d\n", askit_inputs.min_skeleton_level);
    fprintf(file, "-err %d\n", num_error_checks);
    fprintf(file, "-num_error_repeats %d\n", num_error_repeats);
    fprintf(file, "-data %s\n", filename);
    fprintf(file, "-knn_filename %s\n", knn_filename);
    fprintf(file, "-charges %s\n", charge_filename);
    fprintf(file, "-N %d\n", N);
    fprintf(file, "-d %d\n", d);
    fprintf(file, "-kernel_type %s\n", kernel_type);
    fprintf(file, "-do_variable_bandwidth %d\n", kernel_inputs.do_variable_bandwidth);
    fprintf(file, "-num_procs %d\n", size);
    fprintf(file, "-compress_self %d\n", askit_inputs.compress_self);
    fprintf(file, "-skeletonize_self %d\n", askit_inputs.skeletonize_self);
    fprintf(file, "-use_adaptive_id %d\n", askit_inputs.use_adaptive_id);
    fprintf(file, "-use_simplified_adaptive_id %d\n", askit_inputs.use_simplified_adaptive_id);
    fprintf(file, "-traverse_to_self_only %d\n", askit_inputs.traverse_to_self_only);
    fprintf(file, "-do_absolute_num_targets %d\n", askit_inputs.do_absolute_num_targets);
    fprintf(file, "-do_scale_near_adaptive %d\n", askit_inputs.do_scale_near_adaptive);
    fprintf(file, "-id_tol %g\n", askit_inputs.id_tol);
    fprintf(file, "-comp_option %s\n", comp_option);
    fprintf(file, "-dont_store_proj %d\n", askit_inputs.dont_store_proj);
    fprintf(file, "-do_fmm %d\n", askit_inputs.do_fmm);
    fprintf(file, "-merge_aggressive %d\n", askit_inputs.merge_aggressive);
    fprintf(file, "-do_adaptive_level_restriction %d\n", askit_inputs.do_adaptive_level_restriction);
    fprintf(file, "-do_absolute_id_cutoff %d\n", askit_inputs.do_absolute_id_cutoff);
    fprintf(file, "-do_split_k %d\n", askit_inputs.do_split_k);
    fprintf(file, "-pruning_num_neighbors %d\n", askit_inputs.pruning_num_neighbors);
    fprintf(file, "-neighbors_to_pass_up %d\n", askit_inputs.neighbors_to_pass_up);

    fprintf(file, "\n\nERRORS: \n");
    fprintf(file, "-avg_rel_err %g\n", counters.avg_fmm_err);
    fprintf(file, "-max_rel_err %g\n", counters.max_fmm_err);
    fprintf(file, "-avg_abs_err %g\n", counters.avg_abs_err);
    fprintf(file, "-max_abs_err %g\n", counters.max_abs_err);
    fprintf(file, "-l2_err %g\n", counters.l2_error);
    fprintf(file, "-error_per_charge %g\n", counters.error_per_charge);
    fprintf(file, "-nn_err %g\n", counters.avg_nn_err);
    fprintf(file, "-nn_l2_err %g\n", counters.nn_l2_error);
    fprintf(file, "-rel_inf_err %g\n", counters.rel_inf_err);

    fprintf(file, "\n\nTIMINGS: \n");
    fprintf(file, "-tree_build_time %g\n", tree_build_time);
    fprintf(file, "-skeletonization_time %g\n", skeletonization_time);
    fprintf(file, "-list_blocking_time %g\n", list_blocking_time);
    fprintf(file, "-evaluation_time %g\n", evaluation_time);
    fprintf(file, "-test_evaluation_time %g\n", test_evaluation_time);
    fprintf(file, "-exact_comp_time %g\n", exact_comp_time);
    fprintf(file, "-let_traversal_time %g\n", let_traversal_time);
    fprintf(file, "-update_charges_time %g\n", update_charges_time);
    fprintf(file, "-total_time %g\n", tree_build_time + skeletonization_time 
          + list_blocking_time + evaluation_time + let_traversal_time);

    fprintf(file, "\n\nCOUNTS: \n");
    fprintf(file, "-downward_kernel_evals %g\n", (double)global_downward_evals/global_num_targets);
    fprintf(file, "-min_skeleton_size %d\n",  global_min_skeleton_size);
    fprintf(file, "-avg_skeleton_size %g\n",  global_avg_skeleton_size);
    fprintf(file, "-max_skeleton_size %d\n", global_max_skeleton_size);
    fprintf(file, "-avg_potential %g\n", counters.avg_potential);
    fprintf(file, "-num_nodes_with_uniform %ld\n", global_num_nodes_with_uniform);
    fprintf(file, "-near_scale %g\n", alg.near_scale);
    fprintf(file, "-charge_l1_norm %g\n", counters.charge_l1_norm);
    fprintf(file, "-charge_l2_norm_sqr %g\n", counters.charge_l2_norm_sqr);
    
    fprintf(file, "\n\nEXTRA TIMERS: \n");
    fprintf(file, "-construct_tree_list_time: %g\n", all_construct_tree_list_time); 
    fprintf(file, "-compute_leaf_neighbors_time: %g\n", all_compute_leaf_neighbors_time); 
    fprintf(file, "-merge_neighbor_lists_time: %g\n", all_merge_neighbor_lists_time); 
    fprintf(file, "-collect_local_samples_time: %g\n", all_collect_local_samples_time); 
    fprintf(file, "-collect_neighbor_coords_time: %g\n", all_collect_neighbor_coords_time); 
    fprintf(file, "-distributed_upward_pass_time: %g\n", all_distributed_upward_pass_time); 
    fprintf(file, "-uniform_sample_sibling_time: %g\n", all_uniform_sample_sibling_time); 
    fprintf(file, "-dist_compute_skeleton_time: %g\n", all_dist_compute_skeleton_time); 
    fprintf(file, "-merge_skeletons_of_kids_time: %g\n", all_merge_skeletons_of_kids_time); 
    fprintf(file, "-merge_nn_list_time: %g\n", all_merge_nn_list_time); 
    fprintf(file, "-get_dist_node_excl_nn_time: %g\n", all_get_dist_node_excl_nn_time); 
    fprintf(file, "-compute_adaptive_id_time: %g\n", all_compute_adaptive_id_time); 
    fprintf(file, "-apply_proj_time: %g\n", all_apply_proj_time); 
    fprintf(file, "-subsample_self_targets_time: %g\n", all_subsample_self_targets_time); 
    fprintf(file, "-compute_skeleton_time: %g\n", all_compute_skeleton_time); 
    fprintf(file, "-solve_for_proj_time: %g\n", all_solve_for_proj_time); 
    fprintf(file, "-qr_time: %g\n", all_qr_time); 
    fprintf(file, "-max_qr_time: %g\n", all_max_qr_time); 
    fprintf(file, "-kernel_compute_time: %g\n", all_kernel_compute_time); 
    fprintf(file, "-dist_test_set_time: %g\n", all_dist_test_set_time); 
    fprintf(file, "-update_test_let_time: %g\n", all_update_test_let_time); 

    // FMM timers    
    fprintf(file, "-pass_potentials_down_time: %g\n", all_pass_potentials_down_time); 
    fprintf(file, "-merge_fmm_lists_basic_time: %g\n", all_merge_fmm_lists_basic_time); 
    fprintf(file, "-merge_basic_set_difference_time: %g\n", all_merge_basic_set_difference_time); 
    fprintf(file, "-fmm_add_to_map_time: %g\n", all_fmm_add_to_map_time); 
    fprintf(file, "-merge_fmm_lists_aggressive_time: %g\n", all_merge_fmm_lists_aggressive_time); 
    fprintf(file, "-less_than_tree_order_time: %g\n", all_less_than_tree_order_time); 
    fprintf(file, "-is_ancestor_time: %g\n", all_is_ancestor_time); 
    fprintf(file, "-split_node_time: %g\n", all_split_node_time); 
    fprintf(file, "-merge_tree_list_full_time: %g\n", all_merge_tree_list_full_time); 
    fprintf(file, "-compute_near_scale_time: %g\n", all_compute_near_scale_time);     

    // adaptive level restriction timers
    fprintf(file, "-frontier_exchange_time: %g\n", all_frontier_exchange_time);     
    fprintf(file, "-prune_by_knn_time: %g\n", all_prune_by_knn_time);     
    
    // Detailed skeleton sizes
    fprintf(file, "\n\nDETAILED SKELETON SIZES:\n");
    for (int i = askit_inputs.min_skeleton_level; i < num_levels; i++)
    {
      fprintf(file, "Level %d: Min %d, Avg %g, Max %d, Unpruned: %d\n", i, global_min_skeleton_size_per_level[i], global_avg_skeleton_size_per_level[i], global_max_skeleton_size_per_level[i], global_num_unprunable_nodes_per_level[i]);
    }
    
  } // only rank 0 prints output
  
} // OutputTimers



// Actually runs the algorithm and handles the error estimation
//
// * alg -- algorithm class (already constructed)
// * kernel_type -- type of kernel used
// * comp_option -- deprecated
// * output_filename -- the file where the final results are written
// * num_error_checks -- the number of points used to estimate the error
// * kernel_inputs -- inputs to the kernel function
// * askit_inputs -- inputs to ASKIT constructor (parameters, etc.)
// * filename -- file containing the source coordinates
// * knn_filename -- file containing KNN info
// * charge_filename -- file containing input charges (or charge type)
// * N -- number of source points on this MPI rank
// * d -- data dimensionality
// * glb_numof_ref_points -- total number of source points over all ranks
// * num_error_repeats -- Number of independent charge vectors to average over
// * query_gids -- global ids of targets used to estimate error
// * error_point_coordinates -- coordinates of targets used to estimate 
// error
template<class AlgClass>
void RunAlg(AlgClass& alg, char* kernel_type, char* comp_option, char* output_filename, int num_error_checks,
  KernelInputs& kernel_inputs, AskitInputs& askit_inputs, char* filename, char* knn_filename, 
  char* charge_filename, int N, int d, long glb_numof_ref_points, 
  int num_error_repeats, vector<long>& query_gids, vector<double>& error_point_coordinates)
{

// , vector<long>& my_test_gids, vector<int>& num_test_gids

  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Compute results for all points or a subsets
  if (strcmp(comp_option, "all") == 0)
  {

    bool do_all = (strcmp(comp_option, "all") == 0);

    int num_targets;

    double avg_fmm_error = 0.0;
    double max_fmm_error = 0.0;
    double avg_abs_error = 0.0;
    double max_abs_error = 0.0;
    double l2_error = 0.0;
    double avg_nn_error = 0.0;
    double avg_potential = 0.0;

    // cout << "Rank " << rank << ": collecting lids\n";

    // figure out where the gids got moved to
    vector<int> my_test_lids;
    vector<int> num_test_lids(size, 0);
    int my_num_test_lids = 0;
    
    vector<long> my_test_gids;
    // vector<int> num_test_gids // this is the same as num_test_lids
    // int my_num_test_lids = 0; // also the same
    for (int i = 0; i < num_error_checks; i++)
    {
      int lid = alg.tree->pos(query_gids[i]);
      if (lid >= 0 && lid < alg.N) // this means it's one of our original points
      {
        // cout << "Rank " << rank << ": gid: " << query_gids[i] << ", lid " << lid << "\n";
        my_test_gids.push_back(query_gids[i]);
        my_test_lids.push_back(lid);
        my_num_test_lids++;
      } 
    }

    // cout << "Rank " << rank << " calling Allgather\n";
    
    // now, collect the number for each process
    MPI_Allgather(&my_num_test_lids, 1, MPI_INT, num_test_lids.data(), 1, 
      MPI_INT, MPI_COMM_WORLD);

    // cout << "Rank " << rank << " finished calling Allgather\n";
    
    vector<int> displ(size);
    displ[0] = 0;
    for (int i = 1; i < size; i++)
    {
      displ[i] = displ[i-1] + num_test_lids[i-1];
    }
    
 
    // build a counters structure
    ErrorCounters training_error(error_point_coordinates, my_test_gids, my_test_lids, 
      num_test_lids, displ, false);
        
    for (int i = 0; i < num_error_repeats; i++)
    {

      // do the calculation, these are indexed by local id
      std::vector<double> potentials;

      if (do_all)
      {
        num_targets = N;
        potentials = alg.ComputeAll();
      }
      else
      {
        cout << "ERROR: ComputeSome() not supported.\n";
        //num_targets = num_error_checks;
        //potentials = alg.ComputeSome(query_lids);
      }

      // save results -- if needed
      if (askit_inputs.save_training_potentials)
      {
        double *yest = new double [potentials.size()];
        alg.tree->shuffle_back(alg.N, potentials.data(),
                               alg.tree->inProcData->gids.data(),
                               alg.N, yest, MPI_COMM_WORLD);
        knn::mpi_binwrite(askit_inputs.training_potentials_filename, potentials.size(), 1, yest, MPI_COMM_WORLD);
        delete [] yest;
      } // save the test potentials
      

      MPI_Barrier(MPI_COMM_WORLD);
      
      // we're not doing test points here
      EstimateErrors(potentials, num_error_checks, query_gids, alg, do_all,
        training_error);

      MPI_Barrier(MPI_COMM_WORLD);

      
      // don't bother updating charges if this is the last iteration
      if (i < num_error_repeats - 1)
      {

        // now, generate new charges
        //vector<double> new_charges = NormalCharges(N);
        // cout << "rank " << rank << " generating charges\n";
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        // vector<double> new_charges(N, ((double)(i+2))/sqrt(glb_numof_ref_points));
        vector<double> new_charges(N);
        generateNormal(N, 1, new_charges.data(), MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        
        // cout << "rank " << rank << " updating charges\n";
        alg.UpdateCharges(new_charges);
        
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
          cout << "rank " << rank << ": UpdateCharges time " << i << ": " << alg.update_charges_time << "\n";

      }

    } // loop over samples of charges

    OutputTimes(alg, askit_inputs, kernel_inputs, output_filename, training_error,
      num_error_checks, num_error_repeats, filename, knn_filename, charge_filename,
      N, d, kernel_type, comp_option);

  }
  else {

    std::cout << "Invalid comp option\n";
    return;

  }

} // RunAlg


int main(int argc, char* argv[])
{

  MPI::Init(argc, argv);

  int rank, size;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  srand((unsigned)time(NULL)*rank);

  // Set up and read in the parameters
  CmdLine cmd;
  const char *phelp = "Executable for ASKIT. IMPORTANT: number of MPI ranks must be a power of two!";
  cmd.addInfo(phelp);


  /////////////////////////// Input Data /////////////////////////////////
  char* filename;
  cmd.addSCmdOption("-data", &filename, "REQUIRED",
    "File containing data set.");

  char* charge_file;
  cmd.addSCmdOption("-charges", &charge_file, "ones",
    "File containing charges (or setting for randomly generating them).  'ones' -- all charges are 1/sqrt(N) (default). Set to 'rand' for uniformly distributed charges in the unit interval and `norm' for standard normal charges.");

  char* knn_filename;
  cmd.addSCmdOption("-knn_file", &knn_filename, "REQUIRED",
    "File containing the neighbor info.");

  bool is_binary_file;
  cmd.addBCmdOption("-binary", &is_binary_file, false,
    "indicate the input file is binary file [true] or ascii file [false] (default: false)");

  long N;
  cmd.addLCmdOption("-N", &N, 0, "Total number of data points in input file.");

  int d;
  cmd.addICmdOption("-d", &d, 0, "Dimension of data points.");


  ////////////////////// Main ASKIT parameters //////////////////////////

  int num_neighbors_in;
  cmd.addICmdOption("-k", &num_neighbors_in, 10, "Number of nearest neighbors to use (Default: 10).");

  int max_points_per_node;
  cmd.addICmdOption("-fks_mppn", &max_points_per_node, 100, "(m) Maximum number of points per tree node (Default: 100)");

  int id_rank;
  cmd.addICmdOption("-id_rank", &id_rank, 10, "(s) Rank of ID to compute for skeletons. This is s_max in the adaptive rank version (Default: 10).");

  int min_skeleton_level;
  cmd.addICmdOption("-min_skeleton_level", &min_skeleton_level, 2,
    "(L) Minimum tree level at which to construct a skeleton, with the root at 0.  Above this, we never prune (Default: 2).");

  bool do_simplified_adaptive_rank;
  cmd.addBCmdOption("-do_simplified_adaptive_rank", &do_simplified_adaptive_rank, false,
    "If true, adaptively chooses the skeleton rank using the simplified criterion (r_ss / r_11 < id_tol). id_rank is the max possible rank. (Default: false)");
    
  double id_tol;
  cmd.addRCmdOption("-id_tol", &id_tol, 0.1, "(\tau) Error tolerance in estimated singular values in skeletonization. If using simplified adaptive rank, this is the cutoff for the diagonal entries of R. (default: 0.1)");


  /////////////////////// FMM ////////////////////////////////////////////

  bool do_fmm;
  cmd.addBCmdOption("-do_fmm", &do_fmm, false, "Use the FMM version of the ASKIT.");

  //////////////////////// Kernel Parameters //////////////////////////////

  char* kernel_type;
  cmd.addSCmdOption("-kernel_type", &kernel_type, "gaussian",
    "Type of kernel to use. Options are gaussian, laplace, polynomial. Default: gaussian.");

  bool do_variable_bandwidth;
  cmd.addBCmdOption("-do_variable_bandwidth", &do_variable_bandwidth, false,
    "Turns on the variable bandwidth Gaussian kernel.");

  double bandwidth;
  cmd.addRCmdOption("-h", &bandwidth, 1.0, "Bandwidth of the Gaussian and polynomial kernels (Default 1.0).");

  double poly_constant;
  cmd.addRCmdOption("-c", &poly_constant, 1.0,
    "Additive constant (tradeoff between high and low order terms) for polynomial kernel. (Default 1.0)");

  double poly_degree;
  cmd.addRCmdOption("-p", &poly_degree, 2.0,
    "Degree of the polynomial kernel.  (Default 2)");

  /////////////////////// Output ///////////////////////////////////////

  char* output_filename;
  cmd.addSCmdOption("-output", &output_filename, "output.out",
    "File for output of timings, errors, etc.");

  bool save_training_potentials;
  cmd.addBCmdOption("-save_training_potentials", &save_training_potentials, false,
    "Flag to save training set estimated potentials to file.");
    
  char* training_potentials_filename;
  cmd.addSCmdOption("-training_potentials_file", &training_potentials_filename, "training_potentials.bin",
    "File for saving training set potentials. Only written to if `save_training_potentials' flag is set.");

  bool save_test_potentials;
  cmd.addBCmdOption("-save_test_potentials", &save_test_potentials, false,
    "Flag to save test set estimated potentials to file.");
  
  char* test_potentials_filename;
  cmd.addSCmdOption("-test_potentials_file", &test_potentials_filename, "test_potentials.bin",
    "File for saving test set potentials. Only written to if `save_test_potentials' flag is set.");



  ////////////////////// TEST POINT EVALUATION ////////////////////////////
  bool do_test_evaluation;
  cmd.addBCmdOption("-do_test_evaluation", &do_test_evaluation, false,
    "If true, we read in test points and do the evaluation for them as well.");

  char* test_data_file;
  cmd.addSCmdOption("-test_data_file", &test_data_file, "none", 
    "Filename containing the test data coordinates.");

  char* test_knn_file;
  cmd.addSCmdOption("-test_knn_file", &test_knn_file, "none", 
    "Filename containing the KNN info for the test data.");

  long num_test_points;
  cmd.addLCmdOption("-num_test_points", &num_test_points, 0, 
    "Number of test points in test_data_file.");



  ////////////////////// Advanced parameters ///////////////////////////

  int max_tree_level;
  cmd.addICmdOption("-fks_mtl", &max_tree_level, 30, "(Advanced.) Maximum tree depth (default = 30)");

  int num_skel_targets;
  cmd.addICmdOption("-num_skel_targets", &num_skel_targets, 2, "(Advanced.) Minimum number of targets needed to construct a skeleton (2). Is this times number of sources unless flag `do_absolute_num_targets' is set.");

  int num_uniform_required;
  cmd.addICmdOption("-num_uniform_required", &num_uniform_required, 0, "(Advanced.) We always take at least this many uniform samples, even if we have enough neighbors. Default: 0.");

  int oversampling_factor;
  cmd.addICmdOption("-oversampling_fac", &oversampling_factor, 5, "(Advanced.) Amount to oversample for far-field uniform sampling. (5)");


  
  ///////////////// Error estimation /////////////////////////

  int num_error_checks;
  cmd.addICmdOption("-err", &num_error_checks, 0,
    "Chooses this number of target points, computes the NN and exact results, and reports average error. Default: 0.");

  int num_error_repeats;
  cmd.addICmdOption("-num_error_repeats", &num_error_repeats, 1,
    "Number of different (normal) charges to repeat the error estimate over. default: 1.");


  /////////////////// Advanced Flags ////////////////////////////////////////

  bool compress_self;
  cmd.addBCmdOption("-compress_self", &compress_self, false,
    "(Advanced.) If true, tries to compress the near-field interactions as well. (default: false)");

  bool skeletonize_self;
  cmd.addBCmdOption("-skeletonize_self", &skeletonize_self, false,
    "(Advanced.) If true, includes the subsampled self-interactions in the target list when computing skeletons. (default: false)");

  // note that if both this and do_adaptive_rank are set to true, this one will be run
  bool do_scale_near_adaptive;
  cmd.addBCmdOption("-do_scale_near_adaptive", &do_scale_near_adaptive, false, 
    "(Advanced.) If true, we use the alternative criterion for adaptive rank cutoff which incorporates a lower bound on the near field contribution.");  
    
  bool do_absolute_id_cutoff;
  cmd.addBCmdOption("-do_absolute_id_cutoff", &do_absolute_id_cutoff, false,
    "(Advanced.) If true, then we use an absolute cutoff for the simplified adaptive rank selection algorithm.");
    
  // flag for an absolute number of targets in skeletonization.  Otherwise, 
  // use M log M for M columns 
  bool do_absolute_num_targets;
  cmd.addBCmdOption("-do_absolute_num_targets", &do_absolute_num_targets, false,
    "(Advanced.) If true, then the number of targets for the ID is equal to num_skel_targets (self, then neighbor, then uniform) + num_uniform_required (always uniform). Otherwise, the number of samples is num_skel_targets*M log M where M is the number of columns being skeletonized.");

  // traverse to self does the traversal as if k = 1, but uses the larger 
  // value of k for sampling targets in skeletonization
  bool traverse_to_self_only;
  cmd.addBCmdOption("-traverse_to_self_only", &traverse_to_self_only, false,
    "(Advanced.) If true, we traverse in the downward pass as if k = 1 (i.e. only to the self leaves), even though we still use neighbor information to choose samples.");

  bool dont_store_proj;
  cmd.addBCmdOption("-dont_store_proj", &dont_store_proj, false,
    "(Advanced.) If true, we don't store P in the skeletons to save memory. IMPORTANT: needs to be false to use UpdateCharges().");

  bool do_adaptive_level_restriction;
  cmd.addBCmdOption("-do_adaptive_level_restriction", &do_adaptive_level_restriction, false,
    "(Advanced.) This flag means that we never try to prune a node after we have failed to compress it below nsmax columns in the simplified adaptive rank algorithm.");

  bool do_split_k;
  cmd.addBCmdOption("-do_split_k", &do_split_k, false,
    "(Advanced.) This flag splits k into two parts: one for pruning and one for sampling. The default split is half and half.");

  int pruning_num_neighbors;
  cmd.addICmdOption("-pruning_num_neighbors", &pruning_num_neighbors, 0,
    "(Advanced.) Number of neighbors used for pruning. The rest (up to -k) are used for sampling. Ignored if the -do_split_k flag is not set.");

  int neighbors_to_pass_up;
  cmd.addICmdOption("-neighbors_to_pass_up", &neighbors_to_pass_up, 4,
    "(Advanced.) Maximum number of neighbors to pass up to the parent in the upward pass. It is this times the maximum number of rows needed in a skeletonization.");

  ////////////////////////// Deprecated (bad) options ///////////////////////

  bool merge_aggressive;
  cmd.addBCmdOption("-merge_aggressive", &merge_aggressive, false, "(Deprecated.) Switches between the simple merge and full merge for FMM node-to-node interaction lists.");

  bool do_adaptive_rank;
  cmd.addBCmdOption("-do_adaptive_rank", &do_adaptive_rank, false,
    "(Advanced.) If true, adaptively chooses the skeleton rank using id_rank as the max and id_tol as the absolute error cutoff. (default: false)");

  char* comp_option;
  cmd.addSCmdOption("-comp", &comp_option, "all",
    "(Deprecated.) Select which results to compute.  all: compute all potentials.");



  ////////////////////////////////////////////////////////////////////////////

  // Read from the command line
  cmd.read(argc, argv);
 
  string data_filename(filename);
  string charge_filename(charge_file);

  // Read the data and charges, set up the global IDs 
  fksData* refData = ReadDistData(data_filename, charge_filename, N, d, is_binary_file);

  // Inputs to the kernel function
  KernelInputs inputs;
  inputs.bandwidth = bandwidth;
  inputs.do_variable_bandwidth = do_variable_bandwidth;
  if (strcmp(kernel_type, "gaussian") == 0) 
  {
    inputs.type = ASKIT_GAUSSIAN;
  }
  else if (strcmp(kernel_type, "laplace") == 0)
  {
    inputs.type = ASKIT_LAPLACE;
  }
  else if (strcmp(kernel_type, "polynomial") == 0)
  {
    inputs.type = ASKIT_POLYNOMIAL;
    inputs.power = poly_degree;
    inputs.constant = poly_constant;
  }
  else 
  {
    cout << "Kernel " << kernel_type << " not supported!\n";
    exit(1);
  }

  AskitInputs askit_inputs(num_skel_targets, id_rank, max_points_per_node, max_tree_level,
    num_neighbors_in, min_skeleton_level, oversampling_factor, num_uniform_required, knn_filename, is_binary_file);

  // Set up flags
  askit_inputs.compress_self = compress_self;

  askit_inputs.skeletonize_self = skeletonize_self;

  if (do_adaptive_rank)
  {
    askit_inputs.use_adaptive_id = true;
    askit_inputs.id_tol = id_tol;
  }

  if (do_simplified_adaptive_rank)
  {
    askit_inputs.use_simplified_adaptive_id = true;
    askit_inputs.use_adaptive_id = false;
    askit_inputs.id_tol = id_tol;
  }
  
  askit_inputs.do_absolute_id_cutoff = do_absolute_id_cutoff;
  
  askit_inputs.do_scale_near_adaptive = do_scale_near_adaptive;

  askit_inputs.traverse_to_self_only = traverse_to_self_only;

  askit_inputs.do_absolute_num_targets = do_absolute_num_targets;
  askit_inputs.dont_store_proj = dont_store_proj;

  // FMM flags
  askit_inputs.do_fmm = do_fmm;
  askit_inputs.merge_aggressive = merge_aggressive;
  
  // Save potentials output
  askit_inputs.save_training_potentials = save_training_potentials;
  askit_inputs.training_potentials_filename = training_potentials_filename;
  
  askit_inputs.neighbors_to_pass_up = neighbors_to_pass_up;
  
  askit_inputs.do_adaptive_level_restriction = do_adaptive_level_restriction;
  
  askit_inputs.do_split_k = do_split_k;
  askit_inputs.pruning_num_neighbors = pruning_num_neighbors;

  // Currently, if we don't store P then UpdateCharges won't work
  if (dont_store_proj && num_error_repeats > 1)
  {
    cout << "WARNING: can't do num_error_repeats > 1 with dont_store_proj flag. Resetting num_error_repeats to 1.\n";
    num_error_repeats = 1;
  }
  
  
  vector<double> error_point_coordinates;
  vector<long> my_error_gids;
  vector<int> num_error_gids;
  vector<long> error_gids;
  
  CollectErrorCheckCoordinates(error_point_coordinates, my_error_gids, 
    num_error_gids, error_gids, refData, N, num_error_checks);
    
  
  /////////////////////////////////////////////////////////////////////
  
  // Do the same for test points, if needed
  // In parallel, it's easier to gather the coordinates of the test points now
  // than after tree building, so we'll do that and pass them to RunAlg
  vector<double> test_error_point_coordinates;
  vector<long> my_test_error_gids;
  vector<int> num_test_error_gids;
  vector<long> test_error_gids;
  vector<int> test_displ(size);
  
  fksData* testData;

  if (do_test_evaluation)
  {

    string test_data_string(test_data_file);
    string test_charge_file("zeros");
    testData = ReadDistData(test_data_string, test_charge_file, 
      num_test_points, d, is_binary_file);

    CollectErrorCheckCoordinates(test_error_point_coordinates, 
      my_test_error_gids, num_test_error_gids, test_error_gids,
      testData, num_test_points, num_error_checks);

  }
  
  if (strcmp(kernel_type, "gaussian") == 0)
  {
    // Construct the alg

    // cout << "rank " << rank << " constructing alg\n";
    AskitAlg<GaussianKernel> alg(refData, inputs, askit_inputs);

    int newN = alg.tree->numof_points_of_dist_leaf;

    RunAlg(alg, kernel_type, comp_option, output_filename, num_error_checks, 
      inputs, askit_inputs, filename, knn_filename, charge_file, newN, d, 
      N, num_error_repeats, error_gids, 
      error_point_coordinates);

    if (do_test_evaluation)
    {

      if (rank == 0)
      {
        cout << "\n\n\nDOING TEST SET EVALUATION\n\n\n";
      }

      alg.AddTestPoints(num_test_points, testData, test_knn_file, is_binary_file);

      vector<double> test_potentials = alg.ComputeAllTestPotentials();
      
      // save results -- if needed
      if (save_test_potentials)
      {
        vector<double> yest(testData->numof_points);
        alg.tree->shuffle_back(alg.tree->inProcTestData->numof_points, test_potentials.data(),
                               alg.tree->inProcTestData->gids.data(),
                               testData->numof_points, yest.data(), MPI_COMM_WORLD);
        knn::mpi_binwrite(test_potentials_filename, testData->numof_points, 1, 
          yest.data(), MPI_COMM_WORLD);
      } // save the test potentials
      
      if (num_error_checks > 0)
      {
      
        // figure out where the gids got moved to
        vector<int> my_test_error_lids;
        vector<int> num_test_error_lids(size, 0);
        int my_num_test_error_lids = 0;
    
        vector<long> my_test_error_gids;
        // vector<int> num_test_gids // this is the same as num_test_lids
        // int my_num_test_lids = 0; // also the same
        for (int i = 0; i < num_error_checks; i++)
        {
          int lid = alg.tree->test_pos(test_error_gids[i]);
          if (lid >= 0 && lid < alg.tree->inProcTestData->numof_points) // this means it's one of our tree points
          {
            my_test_error_gids.push_back(test_error_gids[i]);
            my_test_error_lids.push_back(lid);
            my_num_test_error_lids++;
          } 
        }

        // now, collect the number for each process
        MPI_Allgather(&my_num_test_error_lids, 1, MPI_INT, num_test_error_lids.data(), 
          1, MPI_INT, MPI_COMM_WORLD);
    
        vector<int> displ(size);
        displ[0] = 0;
        for (int i = 1; i < size; i++)
        {
          displ[i] = displ[i-1] + num_test_error_lids[i-1];
        }
        
        // for (int r = 0; r < size; r++)
        // {
        //   if (r == rank)
        //     cout << "Rank " << rank << " handling " << my_test_error_lids.size() << " test points.\n";
        //   MPI_Barrier(MPI_COMM_WORLD);
        // }
 
        ErrorCounters test_errors(test_error_point_coordinates, 
          my_test_error_gids, my_test_error_lids, num_test_error_lids, displ, true);

        // This should print the errors for now          
        // true is for do_all
        EstimateErrors(test_potentials, num_error_checks, test_error_gids, alg, 
          true, test_errors);
          
      } // if num error checks > 0

    } // if we're doing test points

  }
  else if (strcmp(kernel_type, "laplace") == 0)
  {
    // Construct the alg

    // cout << "rank " << rank << " constructing alg\n";
    AskitAlg<LaplaceKernel> alg(refData, inputs, askit_inputs);

    int newN = alg.tree->numof_points_of_dist_leaf;

    RunAlg(alg, kernel_type, comp_option, output_filename, num_error_checks, 
      inputs, askit_inputs, filename, knn_filename, charge_file, newN, d, 
      N, num_error_repeats, error_gids, 
      error_point_coordinates);

    if (do_test_evaluation)
    {

      if (rank == 0)
      {
        cout << "\n\n\nDOING TEST SET EVALUATION\n\n\n";
      }

      alg.AddTestPoints(num_test_points, testData, test_knn_file, is_binary_file);

      vector<double> test_potentials = alg.ComputeAllTestPotentials();
      
      // save results -- if needed
      if (save_test_potentials)
      {
        vector<double> yest(testData->numof_points);
        alg.tree->shuffle_back(alg.tree->inProcTestData->numof_points, test_potentials.data(),
                               alg.tree->inProcTestData->gids.data(),
                               testData->numof_points, yest.data(), MPI_COMM_WORLD);
        knn::mpi_binwrite(test_potentials_filename, testData->numof_points, 1, 
          yest.data(), MPI_COMM_WORLD);
      } // save the test potentials
      
      if (num_error_checks > 0)
      {
      
        // figure out where the gids got moved to
        vector<int> my_test_error_lids;
        vector<int> num_test_error_lids(size, 0);
        int my_num_test_error_lids = 0;
    
        vector<long> my_test_error_gids;
        // vector<int> num_test_gids // this is the same as num_test_lids
        // int my_num_test_lids = 0; // also the same
        for (int i = 0; i < num_error_checks; i++)
        {
          int lid = alg.tree->test_pos(test_error_gids[i]);
          if (lid >= 0 && lid < alg.tree->inProcTestData->numof_points) // this means it's one of our tree points
          {
            my_test_error_gids.push_back(test_error_gids[i]);
            my_test_error_lids.push_back(lid);
            my_num_test_error_lids++;
          } 
        }

        // now, collect the number for each process
        MPI_Allgather(&my_num_test_error_lids, 1, MPI_INT, num_test_error_lids.data(), 1, 
          MPI_INT, MPI_COMM_WORLD);
    
        vector<int> displ(size);
        displ[0] = 0;
        for (int i = 1; i < size; i++)
        {
          displ[i] = displ[i-1] + num_test_error_lids[i-1];
        }
        
        for (int r = 0; r < size; r++)
        {
          if (r == rank)
            cout << "Rank " << rank << " handling " << my_test_error_lids.size() << " test points.\n"; 
          MPI_Barrier(MPI_COMM_WORLD);
        }
 
        ErrorCounters test_errors(test_error_point_coordinates, 
          my_test_error_gids, my_test_error_lids, num_test_error_lids, displ, true);


        // This should print the errors for now          
        // true is for do_all
        EstimateErrors(test_potentials, num_error_checks, test_error_gids, alg, 
          true, test_errors);
          
      } // if num error checks > 0

    } // if we're doing test points
  }
  else if (strcmp(kernel_type, "polynomial") == 0)
  {
    // Construct the alg

    // cout << "rank " << rank << " constructing alg\n";
    AskitAlg<PolynomialKernel> alg(refData, inputs, askit_inputs);

    int newN = alg.tree->numof_points_of_dist_leaf;

    RunAlg(alg, kernel_type, comp_option, output_filename, num_error_checks, 
      inputs, askit_inputs, filename, knn_filename, charge_file, newN, d, 
      N, num_error_repeats, error_gids, 
      error_point_coordinates);

    if (do_test_evaluation)
    {

      if (rank == 0)
      {
        cout << "\n\n\nDOING TEST SET EVALUATION\n\n\n";
      }

      alg.AddTestPoints(num_test_points, testData, test_knn_file, is_binary_file);

      vector<double> test_potentials = alg.ComputeAllTestPotentials();
      
      // save results -- if needed
      if (save_test_potentials)
      {
        vector<double> yest(testData->numof_points);
        alg.tree->shuffle_back(alg.tree->inProcTestData->numof_points, test_potentials.data(),
                               alg.tree->inProcTestData->gids.data(),
                               testData->numof_points, yest.data(), MPI_COMM_WORLD);
        knn::mpi_binwrite(test_potentials_filename, testData->numof_points, 1, 
          yest.data(), MPI_COMM_WORLD);
      } // save the test potentials
      
      if (num_error_checks > 0)
      {
      
        // figure out where the gids got moved to
        vector<int> my_test_error_lids;
        vector<int> num_test_error_lids(size, 0);
        int my_num_test_error_lids = 0;
    
        vector<long> my_test_error_gids;
        // vector<int> num_test_gids // this is the same as num_test_lids
        // int my_num_test_lids = 0; // also the same
        for (int i = 0; i < num_error_checks; i++)
        {
          int lid = alg.tree->test_pos(test_error_gids[i]);
          if (lid >= 0 && lid < alg.tree->inProcTestData->numof_points) // this means it's one of our tree points
          {
            my_test_error_gids.push_back(test_error_gids[i]);
            my_test_error_lids.push_back(lid);
            my_num_test_error_lids++;
          } 
        }

        // now, collect the number for each process
        MPI_Allgather(&my_num_test_error_lids, 1, MPI_INT, num_test_error_lids.data(), 1, 
          MPI_INT, MPI_COMM_WORLD);
    
        vector<int> displ(size);
        displ[0] = 0;
        for (int i = 1; i < size; i++)
        {
          displ[i] = displ[i-1] + num_test_error_lids[i-1];
        }
        
        for (int r = 0; r < size; r++)
        {
          if (r == rank)
            cout << "Rank " << rank << " handling " << my_test_error_lids.size() << " test points.\n"; 
          MPI_Barrier(MPI_COMM_WORLD);
        }
 
        ErrorCounters test_errors(test_error_point_coordinates, 
          my_test_error_gids, my_test_error_lids, num_test_error_lids, displ, true);


        // This should print the errors for now          
        // true is for do_all
        EstimateErrors(test_potentials, num_error_checks, test_error_gids, alg, 
          true, test_errors);
          
      } // if num error checks > 0

    } // if we're doing test points

  } // polynomial kernel type 
  else {

    std::cerr << "Invalid Kernel type " << kernel_type << "\n";
    return 1;

  }

  MPI::Finalize();

  return 0;

}



