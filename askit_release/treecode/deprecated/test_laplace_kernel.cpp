
#include <mpi.h>

#include "laplace_kernel.hpp"
#include "kernel_inputs.hpp"
#include <float.h>


#ifdef USE_KS
extern "C" {
#include <ks.h>
}
#include <omp_dgsks_list.hpp>
#endif
  
using namespace std;

using namespace askit;

int main(int argc, char* argv[])
{

  MPI::Init(argc, argv);

  double bandwidth = 1.0;

  int num_points = 5;
  int large_num_points = 1000;
  
  int max_dim = 4;

  double A[20] = {1, 0, 0, 0, 0,
    0, 1, 0, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 0, 1, 0};
    
  std::vector<double> A_vec(A, A+20);
    
  std::vector<double> B_vec(large_num_points * max_dim);
    
  for (int i = 0; i < large_num_points * max_dim; i++)
  {
    B_vec[i] = rand() / (double)RAND_MAX;
  }
  
  vector<double> charges(large_num_points);
  for (int i = 0; i < large_num_points; i++)
  {
    charges[i] = rand() / (double)RAND_MAX;
  }
    
  std::vector<double> potential(num_points);
  vector<double> fast_potential(num_points, 0.0);
  
  // Set up interaction lists for fast kernel evaluation code
  vector<vector<int> > target_inds(1);
  target_inds[0].resize(num_points);
  for(int i = 0; i < num_points; i++)
  {
    target_inds[0][i] = i;
  }
  
  vector<vector<int> > source_inds(1);
  source_inds[0].resize(large_num_points);
  for (int i = 0; i < large_num_points; i++)
  {
    source_inds[0][i] = i;
  }
  
  
  KernelInputs inputs;
  inputs.bandwidth = bandwidth;
  inputs.do_variable_bandwidth = false;

  LaplaceKernel kernel(inputs);

  std::vector<double> K(num_points * num_points);

  // Test the kernel for different values of d
  for (int dim = max_dim; dim <= max_dim; dim++)
  {
  
    std::cout << "Computing kernel matrix for d = " << dim << "\n";
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
    
    double oned = 1.0;
    double zerod = 0.0;
    int onei = 1;
    cblas_dgemv("N", &num_points, &large_num_points, &oned, 
        K.data(), &num_points, charges.data(), &onei, &zerod, potential.data(), &onei);

    for (int i = 0; i < num_points; i++)
    {
      std::cout << "u[" << i << "] = " << potential[i] << "\n";
    }
  
#ifdef USE_KS
    // now, compute using the fast kernel code
    ks_t ker;
    ker.type = KS_LAPLACE;
  
    omp_dgsks_list_separated_u_unsymmetric(
        &ker,
        dim,
        fast_potential,
        target_inds, 
        num_points, 
        A,
        target_inds,
        large_num_points, 
        B_vec.data(),
        source_inds,
        charges.data(),
        source_inds
        );

    cout << "\n\n";
    for (int i = 0; i < num_points; i++)
    {
      cout << "u_fast[" << i << "] = " << fast_potential[i] << "\n";
    }
#endif    
    

  }
    
  MPI::Finalize();

  return 0;
}


