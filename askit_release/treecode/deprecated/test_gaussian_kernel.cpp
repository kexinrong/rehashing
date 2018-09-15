
#include <mpi.h>

#include "gaussian_kernel.hpp"
#include "kernel_inputs.hpp"
#include <float.h>

using namespace askit;

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
    
  std::vector<double> A_vec(A, A+20);
    
  KernelInputs inputs;
  inputs.bandwidth = bandwidth;
  inputs.do_variable_bandwidth = false;

  GaussianKernel kernel(inputs);

  std::vector<double> K(num_points * num_points);

  std::cout << "Computing kernel matrix\n";
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

  std::cout << "Computing variable bandwidth matrix.\n";
  KernelInputs var_inputs;
  var_inputs.bandwidth = bandwidth;
  var_inputs.do_variable_bandwidth = true;
  GaussianKernel var_kern(var_inputs);
  var_kern.Compute(A_vec.begin(), A_vec.end(), A_vec.begin(), A_vec.end(), dim, K);
  
  std::cout << "Variable kernel result: \n";
  for (int i = 0; i < num_points; i++)
  {
    for (int j = 0; j < num_points; j++)
    {
      std::cout << K[i + j * num_points] << ", ";
    }
    std::cout << "\n";
  }
  
  
  int large_num_points = 1000;
  
  double* B = new double[large_num_points * dim];
  
  for (int i = 0; i < large_num_points * dim; i++)
  {
    B[i] = rand() / (double)RAND_MAX;
  }
  
  std::vector<double> B_vec(B, B+large_num_points*dim);
  
  // reuse the kernel to test resizing code
  kernel.Compute(A_vec.begin(), A_vec.end(), B_vec.begin(), B_vec.end(), dim, K);
  
  std::vector<double> potential(num_points);
  std::vector<double> charges(large_num_points, 1.0);
  
  /*
  cblas_dgemv(CblasColMajor, CblasNoTrans, num_points, large_num_points, 1.0, 
    K.data(), num_points, charges.data(), 1, 0.0, potential.data(), 1);
  
  for (int i = 0; i < num_points; i++)
  {
    std::cout << "u[" << i << "] = " << potential[i] << "\n";
  }
  
  // reuse the kernel to test resizing code
  var_kern.Compute(A_vec.begin(), A_vec.end(), B_vec.begin(), B_vec.end(), dim, K);
  
  cblas_dgemv(CblasColMajor, CblasNoTrans, num_points, large_num_points, 1.0, 
    K.data(), num_points, charges.data(), 1, 0.0, potential.data(), 1);
  
  for (int i = 0; i < num_points; i++)
  {
    std::cout << "u[" << i << "] = " << potential[i] << "\n";
  }
*/
  
  MPI::Finalize();

  return 0;
}


