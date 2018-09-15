
#include "treecode_driver.hpp"

#include <CmdLine.h>
#include <ompUtils.h>
#include <ctime>
#include <omp.h>
#include <float.h>
#include <parallelIO.h>

#include "generator.h"

using namespace Torch;


int main(int argc, char* argv[])
{
  
  // needed for tree code, only works for one process, though
  MPI::Init(argc, argv);
  
  // For debugging purposes
  srand(0);
  
  // Read in the parameters
  CmdLine cmd;
  const char *phelp = "Help";
  cmd.addInfo(phelp);
  
  int num_neighbors_in;
  cmd.addICmdOption("-k", &num_neighbors_in, 10, "Number of nearest neighbors to find (10).");
  
  int max_points_per_node;
  cmd.addICmdOption("-fks_mppn", &max_points_per_node, 1000, "maximum number of points per kernel summation tree node [fks] (1000)");
  
  int max_tree_level;
  cmd.addICmdOption("-fks_mtl", &max_tree_level, 10, "maximum kernel summation tree depth [fks] (default = 10)");
  
  int num_neighbor_iterations;
  cmd.addICmdOption("-rkdt_iter", &num_neighbor_iterations, 4, "number of random projection trees used [knn] (4)");

  int rkdt_maxLevel;
  cmd.addICmdOption("-rkdt_mtl", &rkdt_maxLevel, 10, "maximum random projection tree depth [knn] (default = 10)");
  
  int rkdt_mppn;
  cmd.addICmdOption("-rkdt_mppn", &rkdt_mppn, 1000, "maximum number of points per random projection tree node [knn] (1000)");

  int debug;
  cmd.addICmdOption("-debug", &debug, 0, "output debug informaiton (1) or not (*0*)");

  double bandwidth;
  cmd.addRCmdOption("-h", &bandwidth, 1.0, "Bandwidth of the Gaussian kernel (Default 1.0).");
  
  int num_skel_targets;
  cmd.addICmdOption("-num_skel_targets", &num_skel_targets, 50, "Minimum number of targets needed to construct a skeleton (50).");
  
  int id_rank;
  cmd.addICmdOption("-id_rank", &id_rank, 10, "Rank of ID to compute for skeletons (10).");

  int oversampling_factor;
  cmd.addICmdOption("-oversampling_fac", &oversampling_factor, 3, "Amount to oversample for far-field uniform sampling.");

  int min_comm_size_per_node = 1;

  // don't construct any skeletons yet
  int min_skeleton_level;
  cmd.addICmdOption("-min_skeleton_level", &min_skeleton_level, 5, 
    "Minimum tree level at which to construct a skeleton.  Above this, we never prune (5).");

  int comp_option;
  cmd.addICmdOption("-comp_option", &comp_option, 0, 
    "Select which results to compute.  0: compute and report all potentials.  1: randomly select a query, report exact, NN, and approximate results.");  


  // Read in the data set: 
  char* filename;
  cmd.addSCmdOption("-data", &filename, "rand", 
    "File containing data set. Set to 'rand' (default) for randomly generated data.");
  
  int N;
  cmd.addICmdOption("-N", &N, 0, "Number of data points.");

  int d;
  cmd.addICmdOption("-d", &d, 0, "Dimension of data points.");
  
  char* charge_file;
  cmd.addSCmdOption("-charges", &charge_file, "ones", 
    "File containing charges.  If left unspecified, all charges are one. Set to 'rand' for uniformly distributed charges in the unit interval.");


  cmd.read(argc, argv);
  
  
  int start_id = 0;
  double* data_ptr;
  
  if (strcmp(filename, "rand") == 0) {
    
    data_ptr = new double[N * d];
    for (int i = 0; i < N * d; i++)
    {
      // uniform in (0,1)
      // IMPORTANT: this may not be thread safe
      data_ptr[i] = rand() / (double)RAND_MAX;
    }
    
  }
  else {

    bool fileres = knn::seqBinRead(filename, start_id, N, d, data_ptr);

    if (!fileres)
    {
      std::cerr << "Couldn't read data file.\n";
      return 1;
    }
    
  }
  
  // Set up the data
  fksData data;

  data.dim = d;
  data.numof_points = N;
  data.X.assign(data_ptr, data_ptr + d*N);
  
  // Read in or generate the charges
  if (strcmp(charge_file, "ones") == 0)
  {

    data.charges.resize(N);
    for (int i = 0; i < N; i++)
    {
      data.charges[i] = 1.0;
    }

  }
  else if (strcmp(charge_file, "rand") == 0) {
    
    data.charges.resize(N);
    for (int i = 0; i < N; i++)
    {
      data.charges[i] = rand() / (double)RAND_MAX;
    }
    
  }
  else {
    
    double* charge_ptr;
    int charge_d = 1;
    bool charge_file_res = knn::seqBinRead(charge_file, start_id, N, charge_d, charge_ptr);
    
    if (!charge_file_res)
    {
      std::cerr << "Couldn't read charge file.\n";
      return 1;
    }
    
    data.charges.assign(charge_ptr, charge_ptr + N);
    
  } // reading charges
  
  data.gids.resize(N);
  for (int i = 0; i < N; i++)
  {
    data.gids[i] = (long)i;
  }
  
  // Construct the driver
  TreecodeDriver driver(&data, bandwidth, num_skel_targets, id_rank, max_points_per_node, max_tree_level, 
    min_comm_size_per_node, num_neighbors_in, num_neighbor_iterations, min_skeleton_level, oversampling_factor);


  if (comp_option == 0)
  {
  std::vector<double> potentials(N, 0.0);

#pragma omp parallel for
  for (long query_id = 0; query_id < N; query_id++)
  {
    
    potentials[query_id] = driver.Compute(query_id);
    
  }

  // output the results
  
  std::cout << "Potentials: \n";
  for (int i = 0; i < N; i++)
  {
    std::cout << "u[" << i << "] = " << potentials[i] << "\n";
  }

  }
  // only compute one result
  else {
  
    int query_ind = rand() % N;

    std::cout << "For query: " << query_ind << ":\n";
    
    double exact_potential = driver.ComputeNaive(query_ind);

    std::cout << "Exact potential: " << exact_potential << "\n";

    double approx_potential = driver.Compute(query_ind);

    std::cout << "Approx potential: " << approx_potential << "\n";

    double nn_potential = driver.ComputeNN(query_ind);
    
    std::cout << "NN potential: " << nn_potential << "\n";    
  
  }
  
  MPI::Finalize();
	
  return 0;
  
}



