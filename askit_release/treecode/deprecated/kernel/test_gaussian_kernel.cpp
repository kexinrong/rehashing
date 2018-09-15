
#include "gaussian_kernel.hpp"
#include <float.h>
#include <mpi.h>

int main(int argc, char* argv[])
{

  MPI::Init(argc, argv);

  double bandwidth = 1.0;

  int num_points = 4;
  int dim = 5;

  double A[20] = {1, 0, 0, 0, 0,
    0, 1, 0, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 0, 1, 0};


  GaussianKernel kernel(bandwidth, num_points*num_points);

  double* K = new double[num_points * num_points];

  std::cout << "Computing kernel matrix\n";
  kernel.Compute(A, num_points, A, num_points, dim, K);

  std::cout << "kernel result: \n";
  for (int i = 0; i < num_points; i++)
  {
    for (int j = 0; j < num_points; j++)
    {
      std::cout << K[i + j * num_points] << ", ";
    }
    std::cout << "\n";
  }

  delete K;
  
  int large_num_points = 1000;
  
  double* B = new double[large_num_points * dim];
  
  for (int i = 0; i < large_num_points * dim; i++)
  {
    B[i] = rand() / (double)RAND_MAX;
  }
  
  K = new double[large_num_points * num_points];

  GaussianKernel big_kernel(bandwidth, num_points * large_num_points);
  big_kernel.Compute(A, num_points, B, large_num_points, dim, K);
  
  double* potential = new double[num_points];
  std::vector<double> charges(large_num_points, 1.0);
  
  cblas_dgemv(CblasColMajor, CblasNoTrans, num_points, large_num_points, 1.0, K, num_points, charges.data(), 1, 0.0, potential, 1);
  
  for (int i = 0; i < num_points; i++)
  {
    std::cout << "u[" << i << "] = " << potential[i] << "\n";
  }

  delete K;
  delete potential;

  MPI::Finalize();

  return 0;
}


