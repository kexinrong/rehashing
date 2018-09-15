#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <utility>
#include <omp.h>
#include "direct_knn.h" 


using namespace std;
int main(int argc, char **argv)
{
  int	n, m, dim, k;	
  int 	rank, size;
  double *dataPts, *queryPts;
  int threads;

  n = 64000;
  m = 100;
  dim = 4;
  k = 1;
  queryPts = new double[m*dim];
  dataPts = new double[n*dim];	

  srand48(1);	
  for( int i = 0; i < n * dim; i++ )
    dataPts[i] = drand48();

  for( int i = 0; i < m * dim; i++ )
    queryPts[i] = drand48();

  double tstart, tstop, telapsed;
  std::pair<double, long> *nn;

  for( threads = 1; threads < 16; threads++ ) {
    omp_set_num_threads(threads);      
   
   // tstart = omp_get_wtime();
    nn = knn::directKQuery( dataPts, queryPts, n, m, k,  dim );	
   // tstop = omp_get_wtime();
   // telapsed = tstop - tstart;
  
    //double flops = 2*m*n*dim + 2*n + 2*m + 2*m*n;
    // output: threads, total time, distance calc time, distance calc gflops/s, overall gflops/s
    //cout << threads << "," << telapsed << "," << flops/telapsed << endl; 
    cout << threads << ", "<< perf->getTotalTime() << ", " << perf->getTime(0) << ", "  
         << perf->getFlops(0)/perf->getTime(0)/1.0e9 << ", " << perf->getTotalFlops()/perf->getTotalTime()/1.0e9 << endl;

    delete [] nn;
    delete perf;
  }

  delete [] dataPts;
  delete [] queryPts;

  return 0;
}
