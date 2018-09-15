
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

#include "generator.h"
#include "power_method.hpp"

using namespace Torch;
using namespace askit;

// The only reason this exists in a function is to keep from duplicating code
// for the different kernel types
template<class AlgClass>
void RunAlg(AlgClass& alg, char* kernel_type, char* comp_option, char* output_filename, int num_error_checks,
  KernelInputs& kernel_inputs, AskitInputs& askit_inputs, char* filename, char* knn_filename, int N, int d, int intrinsic_d)
{
  
  std::cout << "Computing results\n";
  
  if (strcmp(comp_option, "all") == 0)
  {
    std::vector<double> potentials;
    potentials = alg.ComputeAll();

    std::cout << "Computing Exact Error estimate.\n";
    
    std::vector<int> query_lids(num_error_checks);
    double avg_fmm_error = 0.0;
    double avg_nn_error = 0.0;
    double max_abs_error = 0.0;
    double avg_abs_error = 0.0;
    double max_fmm_error = 0.0;
    for (int i = 0; i < num_error_checks; i++)
    {
      
      int query_ind = rand() % N;
      //int query_ind = i;
      query_lids[i] = query_ind;
      
      double exact_potential = alg.ComputeDirect(query_ind);
      double nn_potential = alg.ComputeNN(query_ind);
      double approx_potential = potentials[query_ind];

      //printf("\nquery_ind: %d\n", query_ind);
      //printf("Exact potential: %.15e, approx: %.15e\n", exact_potential, approx_potential);
      
      max_fmm_error = max(max_fmm_error, fabs(approx_potential - exact_potential) / exact_potential);
      avg_fmm_error += fabs(approx_potential - exact_potential) / exact_potential;
      avg_nn_error += fabs(nn_potential - exact_potential) / exact_potential;
      
      avg_abs_error += fabs(approx_potential - exact_potential);
      max_abs_error = max(max_abs_error, fabs(approx_potential - exact_potential));
      
    } // loop over queries to check accuracy
    
    /*
    std::vector<double> est_direct_potentials = alg.ComputeEstDirect(query_lids);
    for (int i = 0; i < num_error_checks; i++)
    {
      printf("Est direct potential: %.15e\n", est_direct_potentials[i]);
    }
    */
    if (num_error_checks > 0)
    {
      
      avg_fmm_error /= num_error_checks;
      avg_nn_error /= num_error_checks;
      avg_abs_error /= num_error_checks;
      
      std::cout << "\nERRORS:\t (Averaged over " << num_error_checks << " queries).\n";
      std::cout << "FMM Avg Relative error: " << avg_fmm_error << "\n";
      std::cout << "FMM Max Relative error: " << max_fmm_error << "\n";
      std::cout << "FMM Avg Absolute error: " << avg_abs_error << "\n";
      std::cout << "FMM Max Absolute error: " << max_abs_error << "\n";
      std::cout << "NN Relative error: " << avg_nn_error << "\n";
     
    }
    
    // compute the average skeleton size
    // TODO: stop counting nodes that don't exist (or are near the root)
    long total_skeleton_size = 0;
#pragma omp parallel for reduction(+:total_skeleton_size)
    for (int i = 0; i < alg.skeleton_sizes.size(); i++)
    {
      total_skeleton_size += alg.skeleton_sizes[i];
    }
    
    double avg_skeleton_size = (double)total_skeleton_size / alg.skeleton_sizes.size();
    
    std::cout << "\nRUNTIMES:\n";
    std::cout << "Tree construction and NN finding: " << alg.build_time << " s.\n";
    std::cout << "Skeleton construction: " << alg.skeletonization_time << " s.\n";
    std::cout << "Near-Field List Construction: " << alg.list_building_time << " s.\n";
    std::cout << "Near-Field Evaluation: " << alg.near_field_time << " s.\n";
    std::cout << "Far-Field Evaluation: " << alg.far_field_time << " s.\n";
    
    std::cout << "\nSTATS: \n";
    std::cout << "Number of base cases: " << alg.num_base_cases << "\n";
    std::cout << "Number of prunes: " << alg.num_prunes << "\n";
    std::cout << "Kernel evaluations in downward pass: " << (double)alg.num_downward_kernel_evals / N << "\n";
    std::cout << "Total kernel evaluations: " << (double)alg.num_kernel_evals / N << "\n";
    std::cout << "Average skeleton size: " << avg_skeleton_size << "\n";
    
    
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
    fprintf(file, "-min_skeleton_level %d\n", askit_inputs.min_skeleton_level);
    fprintf(file, "-err %d\n", num_error_checks);
    fprintf(file, "-data %s\n", filename);
    fprintf(file, "-knn_filename %s\n", knn_filename);
    fprintf(file, "-N %d\n", N);
    fprintf(file, "-d %d\n", d);
    fprintf(file, "-int_d %d\n", intrinsic_d);
    fprintf(file, "-kernel_type %s\n", kernel_type);
    // including to keep from breaking table making script
    fprintf(file, "-num_procs %d\n", 1);
    fprintf(file, "-compress_self %d\n", askit_inputs.compress_self);
    fprintf(file, "-skeletonize_self %d\n", askit_inputs.skeletonize_self);
    fprintf(file, "-use_adaptive_id %d\n", askit_inputs.use_adaptive_id);
    fprintf(file, "-id_tol %g\n", askit_inputs.id_tol);

    fprintf(file, "\n\nERRORS: \n");
    fprintf(file, "-avg_rel_err %g\n", avg_fmm_error);
    fprintf(file, "-max_rel_err %g\n", max_fmm_error);
    fprintf(file, "-avg_abs_err %g\n", avg_abs_error);
    fprintf(file, "-max_abs_err %g\n", max_abs_error);
    fprintf(file, "-nn_err %g\n", avg_nn_error);
    
    fprintf(file, "\n\nTIMINGS: \n");
    fprintf(file, "-build_time %g\n", alg.build_time);
    fprintf(file, "-skeletonization_time %g\n", alg.skeletonization_time);
    fprintf(file, "-list_building_time %g\n", alg.list_building_time);
    fprintf(file, "-near_field_time %g\n", alg.near_field_time);
    fprintf(file, "-far_field_time %g\n", alg.far_field_time);
    fprintf(file, "-eval_time %g\n", alg.list_building_time + alg.near_field_time + alg.far_field_time);
    fprintf(file, "-total_time %g\n", alg.build_time + alg.skeletonization_time + alg.list_building_time + alg.near_field_time + alg.far_field_time);
  
    fprintf(file, "\n\nCOUNTS: \n");
    fprintf(file, "-direct_evals %g\n", (double)alg.num_base_cases/N);
    fprintf(file, "-skeleton_evals %g\n", (double)alg.num_prunes/N);
    fprintf(file, "-downward_kernel_evals %g\n", (double)alg.num_downward_kernel_evals/N);
    //fprintf(file, "-total_kernel_evals %g\n", (double)alg.num_kernel_evals/N); 
    // including these to keep from breaking table making scripts
    fprintf(file, "-num_let_leaves %d\n", 0);
    fprintf(file, "-num_let_internal %d\n", 0);
    fprintf(file, "-avg_skeleton_size %g\n", avg_skeleton_size);

    fclose(file);
     
  }
  else if (strcmp(comp_option, "estnaive") == 0)
  {
    
    int num_naive = std::min(10000, N);
    std::vector<int> query_lids(num_naive);
    for (int i = 0; i < num_naive; i++)
    {
      query_lids[i] = i;
    }
    
    std::vector<double> potentials = alg.ComputeEstDirect(query_lids);
    
    double est_time = alg.exact_comp_time * (N / (double)num_naive);
    
    std::cout << "\n ESTIMATED Direct RUNTIME: " << est_time << "\n";
    
  }
  else if (strcmp(comp_option, "power") == 0)
  {
    
    double acc = 0.01;
    int max_iterations = 10;
    
    // always initializes with a vector of all ones
    
    double norm = PowerMethod(alg, acc, max_iterations);
    
    std::cout << "\nEstimated Power Method Norm: " << norm << "\n\n";
    
    
  }
  else {
    std::cerr << "Invalid comp option " << comp_option << "\n";
  }

} // RunAlg


int main(int argc, char* argv[])
{
  
  // needed for tree code, this main only works for one process, though
  MPI::Init(argc, argv);
  
  // Set up and read in the parameters
  CmdLine cmd;
  const char *phelp = "Help";
  cmd.addInfo(phelp);
  
  int num_neighbors_in;
  cmd.addICmdOption("-k", &num_neighbors_in, 10, "Number of nearest neighbors to find (10).");
  
  int max_points_per_node;
  cmd.addICmdOption("-fks_mppn", &max_points_per_node, 100, "Maximum number of points per kernel summation tree node [fks] (100)");
  
  int max_tree_level;
  cmd.addICmdOption("-fks_mtl", &max_tree_level, 10, "Maximum kernel summation tree depth [fks] (default = 10)");
  
  char* kernel_type;
  cmd.addSCmdOption("-kernel_type", &kernel_type, "gaussian",
    "Type of kernel to use. Options are gaussian, laplace, polynomial. Default: gaussian.");

  double bandwidth;
  cmd.addRCmdOption("-h", &bandwidth, 1.0, "Bandwidth of the Gaussian and polynomial kernels (Default 1.0).");
  
  double poly_constant;
  cmd.addRCmdOption("-c", &poly_constant, 1.0, 
    "Additive constant (tradeoff between high and low order terms) for polynomial kernel. (Default 1.0)");

  double poly_degree;
  cmd.addRCmdOption("-p", &poly_degree, 2.0, 
    "Degree of the polynomial kernel.  (Default 2)");
  
  int num_skel_targets;
  cmd.addICmdOption("-num_skel_targets", &num_skel_targets, 50, "Minimum number of targets needed to construct a skeleton (50). Will sample far field uniformly at random for extra targets. If using adaptive rank, then the number of targets is the number of sources plus this number.");
  
  int id_rank;
  cmd.addICmdOption("-id_rank", &id_rank, 10, "Rank of ID to compute for skeletons (10).");

  int oversampling_factor;
  cmd.addICmdOption("-oversampling_fac", &oversampling_factor, 5, "Amount to oversample for far-field uniform sampling. (5)");

  // don't construct any skeletons yet
  int min_skeleton_level;
  cmd.addICmdOption("-min_skeleton_level", &min_skeleton_level, 2, 
    "Minimum tree level at which to construct a skeleton.  Above this, we never prune (2).");

  char* comp_option;
  cmd.addSCmdOption("-comp", &comp_option, "all", 
    "Select which results to compute.  all: compute all potentials and save to potentials.out. estnaive: Estimate the naive running time by computing a subset of 10K results. power: Estimate the 2-norm of the approximation of K via the power method.");  
  
  int num_error_checks;
  cmd.addICmdOption("-err", &num_error_checks, 0,
    "Chooses this number of queries, computes the NN and exact results, and reports average error. Default: 0.");  
  
  char* filename;
  cmd.addSCmdOption("-data", &filename, "uniform", 
    "File containing data set. Set to 'uniform' (default), 'gaussian', or 'hypersphere' for randomly generated data.\tIMPORTANT: need to specify a file for now, along with a file of neighbors. Can't randomly generate data.");
  
  char* knn_filename;
  cmd.addSCmdOption("-knn_file", &knn_filename, "neighbors.txt",
    "File containing the neighbor info. TODO: add ability to compute them if they don't exist.");
  
  int N;
  cmd.addICmdOption("-N", &N, 0, "Number of data points.");

  int d;
  cmd.addICmdOption("-d", &d, 0, "Dimension of data points.");
  
  int intrinsic_d;
  cmd.addICmdOption("-int_d", &intrinsic_d, 0, "Intrinsic dimension of randomly generated data points. Must be less than or equal to d.");
  
  bool do_variable_bandwidth;
  cmd.addBCmdOption("-do_variable_bandwidth", &do_variable_bandwidth, false,
    "Turns on the variable bandwidth Gaussian kernel.");
  
  char* charge_file;
  cmd.addSCmdOption("-charges", &charge_file, "ones", 
    "File containing charges.  'ones' -- all charges are 1/sqrt(N) (default). Set to 'rand' for uniformly distributed charges in the unit interval.");

  char* output_filename;
  cmd.addSCmdOption("-output", &output_filename, "output.out",
    "File for output of timings, errors, etc.");
  
  bool compress_self;
  cmd.addBCmdOption("-compress_self", &compress_self, false, 
    "If true, tries to compress the near-field interactions as well. (default: false)");
  
  bool skeletonize_self;
  cmd.addBCmdOption("-skeletonize_self", &skeletonize_self, false, 
    "If true, includes the subsampled self-interactions in the target list when computing skeletons. (default: false)");
  
  bool do_adaptive_rank;
  cmd.addBCmdOption("-do_adaptive_rank", &do_adaptive_rank, false,
    "If true, adaptively chooses the skeleton rank using id_rank as the max and id_tol as the absolute error cutoff. (default: false)");
  
  double id_tol;
  cmd.addRCmdOption("-id_tol", &id_tol, 1.0, "Absolute error tolerance in estimated singular values in skeletonization. (default: 1.0)");
  
  
  // Read from the command line
  cmd.read(argc, argv);
  
  // Generate or read the data  
  std::vector<double> data_ptr(N * d);
  
  std::string data_filename(filename);
  
  std::string binary_ending(".bin");
	
  
  cout << "WARNING: askit_shared_main is now deprecated. Start using askit_distributed_main (which can accept the same arguments.\n";
  
  
  // make sure intrinsic dimension was set to a reasonable value
  if (intrinsic_d > d || intrinsic_d < 1)
  {
    intrinsic_d = d;
  }
  
  if (data_filename == "uniform") {
    generateUniformEmbedding(N, d, intrinsic_d, data_ptr.data(), MPI_COMM_WORLD);
  }
  else if (data_filename == "gaussian") 
  {
		generateNormalEmbedding(N, d, intrinsic_d, data_ptr.data(), MPI_COMM_WORLD);
  }
  else if (data_filename == "hypersphere")
  {
    // IMPORTANT: the intrinsic and ambient arguments are swapped here
    generateUnitHypersphereEmbedded(N, intrinsic_d, d, data_ptr.data(), MPI_COMM_WORLD);    
  }
  // If the file extension is .bin, use Bo's IO code
  else if(data_filename.compare(data_filename.length() - binary_ending.length(), 
    binary_ending.length(), binary_ending) == 0)
  {
    
    bool res = knn::binread(data_filename.c_str(), N, d, data_ptr);
    
    // Check that it worked
    if (!res)
    {
      std::cerr << "Couldn't read binary file " << data_filename << "\n";
      return 1;
    }
    
  }
  else {

    // mpi_dlmread needs transposed version
    // It also requires whitespace separated -- not comma separated 
    const char* ptrInputFile = data_filename.c_str();
    long np = N;
    knn::mpi_dlmread(ptrInputFile, np, d, data_ptr, MPI_COMM_WORLD, false);



    //my_num_of_points = np;
    //refData->dim = d;
    //refData->numof_points = my_num_of_points;
    
  } // reading in data
  
  // Set up the data
  fksData data;

  data.dim = d;
  data.numof_points = N;
  data.X.assign(data_ptr.begin(), data_ptr.end());
  data.charges.resize(N);
  
  std::cout << "first row of data: \n";
  for (int id = 0; id < data.dim; id++)
  {
    std::cout << data.X[id] << ", ";
  }
  std::cout << "\n";
  
  // Read in or generate the charges
  if (strcmp(charge_file, "ones") == 0)
  {

    double oosqrt_n = 1.0/sqrt(N);

#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
      data.charges[i] = oosqrt_n;
    }

  }
  else if (strcmp(charge_file, "rand") == 0) {
    
    for (int i = 0; i < N; i++)
    {
      data.charges[i] = rand() / (double)RAND_MAX;
    }
    
  }
  else {
    
    std::ifstream charge_stream(charge_file);

    if (!charge_stream.is_open())
    {
      std::cerr << "Couldn't read charge file" << charge_file << ".\n";
      return 1;
    }

    for (int i = 0; i < N; i++)
    {
      charge_stream >> data.charges[i];
    }
    
  } // reading charges
  
  data.gids.resize(N);
  for (int i = 0; i < N; i++)
  {
    data.gids[i] = (long)i;
  }
  
  // inputs for the kernel class
  KernelInputs inputs;
  inputs.bandwidth = bandwidth;
  inputs.do_variable_bandwidth = do_variable_bandwidth;
  inputs.power = poly_degree;
  inputs.constant = poly_constant;
  
  int num_uniform_required = 0;
  
  AskitInputs askit_inputs(num_skel_targets, id_rank, max_points_per_node, max_tree_level, 
      num_neighbors_in, min_skeleton_level, oversampling_factor, num_uniform_required, knn_filename);
      
  if (compress_self)
    askit_inputs.compress_self = true;
  
  if (skeletonize_self)
    askit_inputs.skeletonize_self = true;
  
  if (do_adaptive_rank)
  {
    askit_inputs.use_adaptive_id = true;
    askit_inputs.id_tol = id_tol;
  }
  
  std::cout << "Building algorithm class.\n";
  
  if (strcmp(kernel_type, "gaussian") == 0) 
  {
    // Construct the alg
    AskitAlg<GaussianKernel> alg(&data, inputs, askit_inputs);  
  
    RunAlg(alg, kernel_type, comp_option, output_filename, num_error_checks, inputs, askit_inputs, filename, knn_filename, N, d, intrinsic_d);

  }
  else if (strcmp(kernel_type, "laplace") == 0)
  {
    // Construct the alg
    AskitAlg<LaplaceKernel> alg(&data, inputs, askit_inputs);  
  
    RunAlg(alg, kernel_type, comp_option, output_filename, num_error_checks, inputs, askit_inputs, filename, knn_filename, N, d, intrinsic_d);

  }
  else if (strcmp(kernel_type, "polynomial") == 0)
  {
    // Construct the alg
    AskitAlg<PolynomialKernel> alg(&data, inputs, askit_inputs);  
  
    RunAlg(alg, kernel_type, comp_option, output_filename, num_error_checks, inputs, askit_inputs, filename, knn_filename, N, d, intrinsic_d);

  }
  else {
    std::cerr << "Invalid Kernel type " << kernel_type << "\n";
    return 1;
  }
  
  MPI::Finalize();
	
  return 0;
  
} // main




