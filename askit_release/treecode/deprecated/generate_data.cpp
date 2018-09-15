
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

using namespace Torch;
using namespace askit;


int main(int argc, char* argv[])
{
  
  // needed for tree code, this main only works for one process, though
  MPI::Init(argc, argv);
  
  {
    std::cout << "Doing 32 d, 100K\n";
    
  int N = 100000;
  int d = 32;
  int intrinsic_d = 32;
  
  std::vector<double> data(N * d);
  
  
  
	generateNormalEmbedding(N, d, intrinsic_d, data.data(), MPI_COMM_WORLD);


  const char* g32100k = "gaussian_32d_100K.txt";

  ofstream outfile;
  outfile.open(g32100k);
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < d-1; j++)
    {
      outfile << data[j + i*d] << ",";
    }
    outfile << data[(d-1) + i*d] << "\n";
  }
  outfile.close();
  } 

  {
    
    std::cout << "Doing 32 d, 1M\n";
    
  int N = 1000000;
  int d = 32;
  int intrinsic_d = 32;
  
  std::vector<double> data(N * d);
  
  
  
	generateNormalEmbedding(N, d, intrinsic_d, data.data(), MPI_COMM_WORLD);


  const char* g32100k = "gaussian_32d_1M.txt";

  ofstream outfile;
  outfile.open(g32100k);
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < d-1; j++)
    {
      outfile << data[j + i*d] << ",";
    }
    outfile << data[(d-1) + i*d] << "\n";
  }
  outfile.close();
  } 


  {
    
    std::cout << "Doing 16 d, 100K\n";
    
  int N = 100000;
  int d = 16;
  int intrinsic_d = 16;
  
  std::vector<double> data(N * d);
  
  
  
	generateNormalEmbedding(N, d, intrinsic_d, data.data(), MPI_COMM_WORLD);


  const char* g32100k = "gaussian_16d_100K.txt";

  ofstream outfile;
  outfile.open(g32100k);
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < d-1; j++)
    {
      outfile << data[j + i*d] << ",";
    }
    outfile << data[(d-1) + i*d] << "\n";
  }
  outfile.close();
  } 


  //generateUnitHypersphereEmbedded(N, intrinsic_d, d, data_ptr.data(), MPI_COMM_WORLD);    

  {
    
    std::cout << "Doing 16 d, 1M\n";
    
  int N = 1000000;
  int d = 16;
  int intrinsic_d = 16;
  
  std::vector<double> data(N * d);
  
  
  
	generateNormalEmbedding(N, d, intrinsic_d, data.data(), MPI_COMM_WORLD);


  const char* g32100k = "gaussian_16d_1M.txt";

  ofstream outfile;
  outfile.open(g32100k);
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < d-1; j++)
    {
      outfile << data[j + i*d] << ",";
    }
    outfile << data[(d-1) + i*d] << "\n";
  }
  outfile.close();
  } 
  
  {
    std::cout << "Doing 4 d, 100K\n";
    
  int N = 100000;
  int d = 1000;
  int intrinsic_d = 4;
  
  std::vector<double> data(N * d);
  
  
  
	generateNormalEmbedding(N, d, intrinsic_d, data.data(), MPI_COMM_WORLD);


  const char* g32100k = "hypersphere_4d_100K.txt";

  ofstream outfile;
  outfile.open(g32100k);
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < d-1; j++)
    {
      outfile << data[j + i*d] << ",";
    }
    outfile << data[(d-1) + i*d] << "\n";
  }
  outfile.close();
  } 


  {
    std::cout << "Doing 4 d, 1M\n";
    
  int N = 1000000;
  int d = 1000;
  int intrinsic_d = 4;
  
  std::vector<double> data(N * d);
  
  
  
	generateNormalEmbedding(N, d, intrinsic_d, data.data(), MPI_COMM_WORLD);


  const char* g32100k = "hypersphere_4d_1M.txt";

  ofstream outfile;
  outfile.open(g32100k);
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < d-1; j++)
    {
      outfile << data[j + i*d] << ",";
    }
    outfile << data[(d-1) + i*d] << "\n";
  }
  outfile.close();
  } 


  MPI::Finalize();
	
  return 0;
  
  
  
} // main




