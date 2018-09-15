#include <mpi.h>

#include "askit_alg.hpp"
#include "gaussian_kernel.hpp"
#include "laplace_kernel.hpp"
#include "polynomial_kernel.hpp"
#include "kernel_inputs.hpp"

#include <CmdLine.h>
#include <ompUtils.h>
#include <ctime>
#include <omp.h>
#include <float.h>
#include <limits>

#include "generator.h"


using namespace askit;

int main(int argc, char* argv[])
{

	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);


  int d = 4;
  int N = 10000;
  const char* data_file = "gaussian_train_10k_4d.txt";
  const char* knn_file = "gaussian_train_knn.txt";
  int N_test = 10000;
  const char* test_file = "gaussian_test_10k_4d.txt";
  const char* test_knn_file = "gaussian_test_knn.txt";


  //int d = 8;
  //int N = 1000;
  //const char* data_file = "normal-n1000-d8-trn.txt";
  //const char* knn_file = "normal-n1000-k128-trn.txt";
  //int N_test = 1000;
  //const char* test_file = "normal-n1000-d8-tst.txt";
  //const char* test_knn_file = "normal-n1000-k128-tst.txt";


  // int d = 8;
  // int N = 100;
  // const char* data_file = "normal-n100-d8-trn.txt";
  // const char* knn_file = "normal-n100-k32-trn.txt";
  // int N_test = 100;
  // const char* test_file = "normal-n100-d8-tst.txt";
  // const char* test_knn_file = "normal-n100-k32-tst.txt";


  // Generate or read the data
  std::string data_filename(data_file);

  fksData* refData = new fksData();

  int my_num_of_points;

  const char* ptrInputFile = data_filename.c_str();
  long glb_numof_points = N;

  knn::mpi_dlmread(ptrInputFile, glb_numof_points, d, refData->X, comm, false);
  my_num_of_points = glb_numof_points;
  refData->dim = d;
  refData->numof_points = my_num_of_points;
  cout << "Rank " << rank << " read " << my_num_of_points << " with " << d << " features.\n";

  // set up global ids
  long nref = my_num_of_points;
  long glb_numof_ref_points, refid_offset;

  MPI_Allreduce( &nref, &glb_numof_ref_points, 1, MPI_LONG, MPI_SUM, comm );

  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0) {
      cout.flush();
      cout<<"rank "<<rank<<": mpi all reduce done!"<<endl;
  }

  MPI_Scan( &nref, &refid_offset, 1, MPI_LONG, MPI_SUM, comm );
  refid_offset -= nref;
  refData->gids.resize(my_num_of_points);

  MPI_Barrier(MPI_COMM_WORLD);
  if(rank == 0) {
      cout.flush();
      cout<<"rank "<<rank<<": mpi scan done!"<<endl;
  }

  #pragma omp parallel for
  for(int i = 0; i < my_num_of_points; i++) {
      refData->gids[i] = refid_offset + (long)i;
  }

  double charge = 1.0 / sqrt(N);
  std::vector<double> charges(N, charge);
  refData->charges.resize(my_num_of_points, charge);


  fksData* test_data = new fksData();

  int my_num_test_points;
  long glb_num_test_points = N_test;
  knn::mpi_dlmread(test_file, glb_num_test_points, d, test_data->X, comm, false);
  my_num_test_points = glb_num_test_points; test_data->dim = d;

  cout << "Rank " << rank << " read " << my_num_test_points << " points.\n";
  cout << test_data->X[0] << ", " << test_data->X[1] << ", " << test_data->X[2] << ", " << test_data->X[3] << "\n";
  test_data->numof_points = my_num_test_points;

  long test_nref = my_num_test_points;
  long test_refid_offset;
  
  MPI_Allreduce(&test_nref, &glb_num_test_points, 1, MPI_LONG, MPI_SUM, comm);
  
  MPI_Scan(&test_nref, &test_refid_offset, 1, MPI_LONG, MPI_SUM, comm);
  test_refid_offset -= test_nref;
  test_data->gids.resize(test_nref);

  for (long i = 0; i < test_nref; i++)
  {
    test_data->gids[i] = test_refid_offset + i;
  }

  /*
  int num_neighbors_in = 32;
  int max_points_per_node = 8;
  int max_tree_level = 30;
  double h = 10;

  int num_skel_targets = 84;
  int oversampling_factor = 5;
  int num_uniform_required = 0;

  int id_rank = 64;
  int min_skeleton_level = 2;
  */


  int num_neighbors_in = 16;
  int max_points_per_node = 8;
  int max_tree_level = 30;
    

  int num_skel_targets = 22;
  int oversampling_factor = 5;
  int num_uniform_required = 0;

  int id_rank = 1;
  int min_skeleton_level = 2;

  double h = 100000;



  KernelInputs inputs;
  inputs.bandwidth = h;

  AskitInputs askit_inputs(num_skel_targets, id_rank, max_points_per_node, max_tree_level, 
    num_neighbors_in, min_skeleton_level, oversampling_factor, num_uniform_required,
    knn_file);

  AskitAlg<GaussianKernel> update_driver(refData, inputs, askit_inputs);

  std::vector<double> u = update_driver.ComputeAll();
  
  // set up the charges we'll double
  vector<double> q(u.size(), charge);
  
  // double all of the charges
#pragma omp parallel for
  for (int i = 0; i < u.size(); i++)
  {
    q[i] *= 2.0;
  }
  
  cout << "Rank " << rank << " calling update charges.\n";
  update_driver.UpdateCharges(q);
  
  std::vector<double> u2 = update_driver.ComputeAll();
  
  std::cout << "Rank " << rank << ", u2[0]: " << u2[0] << ", u[0]: " << u[0] << ", ratio: " << u2[0]/u[0] << "\n";

  // double all of the charges again
#pragma omp parallel for
  for (int i = 0; i < u.size(); i++)
  {
    q[i] *= 2.0;
  }
  
  cout << "Rank " << rank << " calling update charges again.\n";
  update_driver.UpdateCharges(q);
  
  std::vector<double> u4 = update_driver.ComputeAll();
  
  std::cout << "Rank " << rank << ", u4[0]: " << u4[0] << ", u[0]: " << u[0] << ", ratio: " << u4[0]/u[0] << "\n";

  //////////////////////////////////////////////////////////////////////
  
  
  // cout << "\n\n\n";
//   cout << "Charge table: \n";
//
  
  // cout << "Printing data\n";
  // print_data(update_driver.tree->inProcData, MPI_COMM_WORLD);
  // false because files are not binary
  update_driver.AddTestPoints(test_nref, test_data, test_knn_file, false);

  // cout << "Printing data\n";

  // print_data(update_driver.tree->inProcData, MPI_COMM_WORLD);

  vector<double> u_test = update_driver.ComputeAllTestPotentials();

  // cout << "Printing data\n";

  // print_data(update_driver.tree->inProcData, MPI_COMM_WORLD);
  
  cout << "Rank " << rank << " u_test[0]: " << u_test[0] << "\n";



  // Now, we want to call update charges again
  
#pragma omp parallel for
  for (int i = 0; i < u.size(); i++)
  {
    q[i] *= 2.0;
  }
  
  
  update_driver.UpdateCharges(q);
  
  vector<double> u_train_post_test = update_driver.ComputeAll();
  
  vector<double> u_test_post_test = update_driver.ComputeAllTestPotentials();
  
  
  vector<double> test_coords(d, 0.0);
  for (int i = 0; i < d; i++)
  {
    test_coords[i] = update_driver.tree->inProcData->X[i];
  }
  vector<double> u_exact = update_driver.ComputeDirect(test_coords);
  
  if (rank == 0)
    cout << "Direct eval: " << u_exact[0] << "\n";
  
  cout << "Approx training: " << u_train_post_test[0] << "\n";
  
  //
  // for (int r = 0; r < size; r++)
  // {
  //   MPI_Barrier(MPI_COMM_WORLD);
  //   if (r == rank)
  //   {
  //
  //     cout << "\n Rank " << rank << " TRAINING AFTER TEST:\n";
  //
  //     for (int i = 0; i < update_driver.N; i++)
  //     {
  //       cout << "(" << update_driver.tree->inProcData->gids[i] << ", ";
  //       cout << u_train_post_test[i] << "), ";
  //     }
  //     cout << "\n\n";
  //
  //   }
  // } // loop over ranks
  //
  //
  // for (int r = 0; r < size; r++)
  // {
  //   MPI_Barrier(MPI_COMM_WORLD);
  //   if (r == rank)
  //   {
  //
  //     cout << "\n Rank " << rank << " TEST AFTER TRAINING AFTER TEST:\n";
  //
  //     for (int i = 0; i < update_driver.tree->inProcTestData->numof_points; i++)
  //     {
  //       cout << "(" << update_driver.tree->inProcTestData->gids[i] << ", ";
  //       cout << u_test_post_test[i] << "), ";
  //     }
  //     cout << "\n\n";
  //
  //   }
  // } // loop over ranks
  //
  
  ////////////////////////// Scaled Adaptive Rank Test /////////////////////
  
  // askit_inputs.rank = 100;
  // askit_inputs.do_absolute_num_targets = false;
  // askit_inputs.num_skel_targets = 2;
  // askit_inputs.use_simplified_adaptive_id = true;
  // askit_inputs.do_scale_near_adaptive = true;
  // askit_inputs.id_tol = 0.1;
  // AskitAlg<GaussianKernel> adaptive_driver(refData, inputs, askit_inputs);
  //
  //
  // vector<double> u_adapt = adaptive_driver.ComputeAll();
  //
  // for (int r = 0; r < size; r++)
  // {
  //   MPI_Barrier(MPI_COMM_WORLD);
  //   if (r == rank)
  //   {
  //
  //     cout << "\n Rank " << rank << " ADAPTIVE TEST:\n";
  //
  //     for (int i = 0; i < u_adapt.size(); i++)
  //     {
  //       cout << "(" << adaptive_driver.tree->inProcData->gids[i] << ", ";
  //       cout << u_adapt[i] << "), ";
  //     }
  //     cout << "\n\n";
  //
  //   }
  // } // loop over ranks

  
  ////////////////////////// Scaled Adaptive Rank Test /////////////////////
  
  // askit_inputs.rank = 100;
//   askit_inputs.do_absolute_num_targets = false;
//   askit_inputs.num_skel_targets = 2;
//   askit_inputs.use_simplified_adaptive_id = true;
//   askit_inputs.do_scale_near_adaptive = true;
//   askit_inputs.id_tol = 0.1;
//   AskitAlg<GaussianKernel> adaptive_driver(refData, inputs, askit_inputs);
//
//
//   vector<double> u_adapt = adaptive_driver.ComputeAll();
//
//   for (int r = 0; r < size; r++)
//   {
//     MPI_Barrier(MPI_COMM_WORLD);
//     if (r == rank)
//     {
//
//       cout << "\n Rank " << rank << " ADAPTIVE TEST:\n";
//
//       for (int i = 0; i < u_adapt.size(); i++)
//       {
//         cout << "(" << adaptive_driver.tree->inProcData->gids[i] << ", ";
//         cout << u_adapt[i] << "), ";
//       }
//       cout << "\n\n";
//
//     }
//   } // loop over ranks

  ///////////////////////// Simple FMM Test ///////////////////////////////




  AskitInputs fmm_inputs(num_skel_targets, id_rank, max_points_per_node, max_tree_level,
    num_neighbors_in, min_skeleton_level, oversampling_factor, num_uniform_required,
    knn_file);

  AskitAlg<GaussianKernel> fmm_driver(refData, inputs, fmm_inputs);

  vector<double> u_fmm = fmm_driver.ComputeFMM();

  cout << "Rank " << rank << " u_fmm[0] " << u_fmm[0] << "\n";

  // AskitInputs fmm_agg_inputs(num_skel_targets, id_rank, max_points_per_node, max_tree_level,
  //   num_neighbors_in, min_skeleton_level, oversampling_factor, num_uniform_required,
  //   knn_file);
  //
  // fmm_agg_inputs.merge_aggressive = true;
  //
  // AskitAlg<GaussianKernel> fmm_agg_driver(refData, inputs, fmm_agg_inputs);
  //
  // vector<double> u_fmm_agg = fmm_agg_driver.ComputeFMM();
  //
  // cout << "Rank " << rank << " u_fmm_agg[0] " << u_fmm_agg[0] << "\n";


  MPI_Finalize();
	return 0;
  
}
