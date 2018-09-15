
#include <mpi.h>

#include "polynomial_kernel.hpp"
#include "kernel_inputs.hpp"
#include <float.h>
#include <stdlib.h>

using namespace askit;

int main(int argc, char* argv[])
{

  MPI::Init(argc, argv);

  double bandwidth = 0.5;
  double c = 1.0;
  
  int num_points = 4;
  int large_num_points = 1000;
  
  int dim = 5;

  double A[20] = {1, 0, 0, 0, 0,
    0, 1, 0, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 0, 1, 0};
    
  std::vector<double> A_vec(A, A+20);
    
  std::vector<double> B_vec(large_num_points * dim);

  for (int i = 0; i < large_num_points * dim; i++)
  {
    B_vec[i] = rand() / (double)RAND_MAX;
  }
  std::vector<double> potential(num_points);
  std::vector<double> charges(large_num_points, 1.0);

  
  KernelInputs inputs;
  inputs.bandwidth = bandwidth;
  inputs.constant = c;
  
  std::vector<double> K(num_points * num_points);

  int max_pow = 3;
  // Test the kernel for different values of d
  for (int pow = 1; pow <= max_pow; pow++)
  {

    inputs.power = (double)pow;
    PolynomialKernel kernel(inputs);

    std::cout << "Computing kernel matrix for p = " << pow << "\n";
    kernel.Compute(A_vec.begin(), A_vec.end(), A_vec.begin(), A_vec.end(), dim, K);

    std::cout << "kernel result: \n";
    for (int i = 0; i < num_points; i++)
    {
      for (int j = 0; j < num_points; j++)
      {
        std::cout << K[i + j * num_points] << ", ";
      }
      std::cout << "\n";
    }
  
    // reuse the kernel to test resizing code
    kernel.Compute(A_vec.begin(), A_vec.end(), B_vec.begin(), B_vec.end(), dim, K);
  
    /*
    cblas_dgemv(CblasColMajor, CblasNoTrans, num_points, large_num_points, 1.0, 
      K.data(), num_points, charges.data(), 1, 0.0, potential.data(), 1);
  
    for (int i = 0; i < num_points; i++)
    {
      std::cout << "u[" << i << "] = " << potential[i] << "\n";
    }
  */
  }
    
  MPI::Finalize();

  return 0;
}


