
#include <mpi.h>

#include "askit_alg.hpp"
#include "gaussian_kernel.hpp"
#include "kernel_inputs.hpp"

#include <CmdLine.h>
#include <ompUtils.h>
#include <ctime>
#include <omp.h>
#include <float.h>
#include <parallelIO.h>

#include "generator.h"

using namespace askit;


int main(int argc, char* argv[])
{
  
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

  fksData* refData = new fksData();

  int d;
  int N = 1024;
  
  // mpi_dlmread needs transposed version
  char* ptrInputFile = "matlab_points_trans.txt";

  long np = N;
  knn::mpi_dlmread(ptrInputFile, np, d, refData->X, comm, false);
  int numof_ref_points = np;
  refData->dim = d;
  refData->numof_points = numof_ref_points;
  
  int charge_d;
  np = N;
  char* charge_file = "matlab_charges.txt";
  knn::mpi_dlmread(charge_file, np, charge_d, refData->charges, comm, false);

  long nref = numof_ref_points;
  long glb_numof_ref_points, refid_offset;

  MPI_Allreduce( &nref, &glb_numof_ref_points, 1, MPI_LONG, MPI_SUM, comm );
  MPI_Scan( &nref, &refid_offset, 1, MPI_LONG, MPI_SUM, comm ); 
  refid_offset -= nref;
  refData->gids.resize(numof_ref_points);
#pragma omp parallel for
  for(int i = 0; i < numof_ref_points; i++) {
      refData->gids[i] = refid_offset + (long)i;
  }
  
  //print(refData, comm);

  int num_neighbors_in = 32;
  int max_points_per_node = 8;
  int max_tree_level = 30;
  int num_neighbor_iterations = 1;
  int rkdt_mppn = 1024;
  double h = 0.353553390593274;
  int num_skel_targets = 48; 
  int id_rank = 16;
  int min_skeleton_level = 1;
  int oversampling_factor = 5;
  int num_uniform_required = 0;
  
  // k = 32, results for query 0 (1st line of file)
  //double fmm_res_matlab = 0.034671556684851;
  //double nn_res_matlab = 0.034671512661305;
  //double exact_res_matlab = 0.034671557465992;
  
  // Note: these are the results for the second line in the file. This is 
  // because this is the first local id on rank 0 when running with two ranks
  
  double fmm_res_matlab = 0.046094411914086;
  double nn_res_matlab = 0.045974542836211;
  double exact_res_matlab = 0.046094748475001;
  
  // Do the exact computation here
  
  std::cout << "\n\nMatlab test case: \n";
  
  
  KernelInputs inputs;
  inputs.bandwidth = h;
  inputs.do_variable_bandwidth = false;
  
  const char* knn_file = NULL;
  
  AskitInputs askit_inputs(num_skel_targets, id_rank, max_points_per_node, max_tree_level, 
    num_neighbors_in, min_skeleton_level, oversampling_factor, num_uniform_required,
    knn_file);
  
  // just do exact neighbor search like in the matlab case
  AskitAlg<GaussianKernel> matlab_driver(refData, inputs, askit_inputs);

  //std::cout << "Using " << num_neighbors_in << " neighbors on " << N << " points in " << d << " dimensions.\n";
  
  std::vector<double> approx;
  approx = matlab_driver.ComputeAll();
  if (rank == 0)
  {
    //std::cout << "Approx result 0: " << approx[0] << "\n";
  
    std::cout << "\nERRORS RELATIVE TO MATLAB IMPLEMENTATION:\n";
    std::cout << "FMM output: " << fabs(fmm_res_matlab - approx[0]) / fmm_res_matlab << "\n";
    
  } 
  
  // Now, test the distributed computation of the exact potential
  
  std::vector<int> query_lids;
  query_lids.push_back(0);
  std::vector<double> exact;
  exact = matlab_driver.ComputeEstDirect(query_lids);
  
  if (rank == 0)
  {
    std::cout << "Exact result 0: " << exact[0] << "\n";
    std::cout << "Exact output: " << fabs(exact_res_matlab - exact[0])/ exact_res_matlab << "\n\n\n";
  }
  
  MPI_Finalize();
	return 0;
}


