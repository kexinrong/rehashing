
#include "treecode_driver.hpp"
#include <mpi.h>

#include <CmdLine.h>
#include <ompUtils.h>
#include <ctime>
#include <omp.h>
#include <float.h>

#include "generator.h"

int main(int argc, char* argv[])
{
  
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);


  // Create a small data set
  int dim = 2;
  int N = 3;

  std::cout << "generating data\n";

  std::vector<double> charge_vec(N, 1.0);
	
  fksData data;
  
  data.dim = dim;
  data.numof_points = N;
  data.X.resize(N*dim);
  data.gids.resize(N);
  
  /*
  for (int i = 0; i < N * dim; i++)
  {
    data.X[i] = rand();
  }
  */
  data.X[0] = 1.0;
  data.X[1] = 0.0;
  
  data.X[2] = 0.0;
  data.X[3] = 1.0;
  
  data.X[4] = 2.0;
  data.X[5] = 3.0;

  for (int i = 0; i < N; i++)
  {
    data.gids[i] = (long)i;
  }
  
  data.charges = charge_vec;
  
  // .2 build tree
  
  
  std::cout << "constructing tree and skeletons\n";
  double h = 1.0;
  int num_skel_targets = 5;
  int id_rank = 2;
  int max_points_per_node = 10;
  int max_tree_level = 10;
  int min_comm_size_per_node = 1;
  int num_neighbors_in = 1;
  int num_neighbor_iterations = 1;
  int oversampling_factor = 3;
  // don't construct any skeletons yet
  int min_skeleton_level = 5;
  
  TreecodeDriver driver(&data, h, num_skel_targets, id_rank, max_points_per_node, max_tree_level, 
    min_comm_size_per_node, num_neighbors_in, num_neighbor_iterations, min_skeleton_level, oversampling_factor);


  std::cout << "Computing results\n";
  double exact = driver.ComputeNaive(0);
  std::cout << "Exact result: " << exact << "\n";
  
  double approx = driver.Compute(0);
  std::cout << "Approx result: " << approx << "\n";
  
  double nn_only = driver.ComputeNN(0);
  std::cout << "NN result: " << nn_only << "\n";
  
  // Now, run a bigger one
  
  dim = 5;
  N = 100;
  
  num_skel_targets = 10;
  id_rank = 10;
  num_neighbors_in = 10;
  min_skeleton_level = 2;
  
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

  TreecodeDriver big_driver(&big_data, h, num_skel_targets, id_rank, max_points_per_node, max_tree_level, 
    min_comm_size_per_node, num_neighbors_in, num_neighbor_iterations, min_skeleton_level, oversampling_factor);

  std::cout << "Computing results\n";
  exact = big_driver.ComputeNaive(0);
  std::cout << "Exact result: " << exact << "\n";

  approx = big_driver.Compute(0);
  std::cout << "Approx result: " << approx << "\n";

  nn_only = big_driver.ComputeNN(0);
  std::cout << "NN result: " << nn_only << "\n";
  
  
  
  ////////////////////////////////////////////////////////////
  
  // Comparison to matlab test code
  
  int d = 2;
  N = 1000;
  
  num_neighbors_in = 4;
  num_neighbor_iterations = 5 + d;
  id_rank = 32;
  h = 0.2;
  num_skel_targets = 64;
  max_points_per_node = 8;
  
  int query_id = 0;
  
  double fmm_rel_error_matlab = 0.002203;
  double nn_rel_error_matlab = 0.3566;
  
  std::ifstream in("id_test_mat.out", std::ifstream::in);

  double* matlab_ptr = new double[d * N];

  for (int i = 0; i < d; i++) {
    for (int j = 0; j < N; j++) {
        in >> matlab_ptr[i + j * d];
    }
  }
  
  in.close();  

  fksData matlab_data;
  
  matlab_data.dim = d;
  matlab_data.numof_points = N;
  matlab_data.X.assign(matlab_ptr, matlab_ptr + d * N);
  
  matlab_data.gids.resize(N);
  for (int i = 0; i < N; i++)
  {
    matlab_data.gids[i] = (long)i;
  }
  
  std::vector<double> matlab_charges(N, 1.0/sqrt(N));
  matlab_data.charges = matlab_charges;
  
  TreecodeDriver matlab_driver(&matlab_data, h, num_skel_targets, id_rank, max_points_per_node, max_tree_level, 
    min_comm_size_per_node, num_neighbors_in, num_neighbor_iterations, min_skeleton_level, oversampling_factor);

  std::cout << "Computing results on matlab data:\n";
  exact = matlab_driver.ComputeNaive(query_id);
  std::cout << "Exact result: " << exact << "\n";

  approx = matlab_driver.Compute(query_id);
  std::cout << "Approx result: " << approx << "\n";

  nn_only = matlab_driver.ComputeNN(query_id);
  std::cout << "NN result: " << nn_only << "\n";

  std::cout << "FMM relative error: " << fabs(exact - approx) / exact << ", matlab gives " << fmm_rel_error_matlab << "\n";
  std::cout << "NN relative error: " << fabs(exact - nn_only) / exact << ", matlab gives " << nn_rel_error_matlab << "\n";
  
  
  
	MPI_Finalize();
	return 0;
}
