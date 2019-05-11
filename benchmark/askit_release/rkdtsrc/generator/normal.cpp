#include <direct_knn.h>
#include <omp.h>
#include <mpi.h>
#include <cmath>
#include <cstdlib>
//#include "random123wrapper.h"

#ifndef __APPLE__

void generateNormal(int n, int d, double *x, MPI_Comm comm) {
 
   int rank, size;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &size);

   struct drand48_data *seeds;
 
   //Fill x with random numbers from the N(0,1) distribution
   #pragma omp parallel
   {  
      int t = omp_get_thread_num();
      int p = omp_get_num_threads();

      //Seed each thread's prng
      #pragma omp master
      {
         seeds = new struct drand48_data[p];
         srand48((long)time(NULL) + rank);
         for(int i = 0; i < p; i++)
            srand48_r(lrand48(), &(seeds[i]));
      }
      #pragma omp barrier

      #pragma omp for
      for (int j=0; j<n*d ; j++){
         double r1, r2;
         drand48_r(&(seeds[t]), &r1);
         drand48_r(&(seeds[t]), &r2);
         //Box muller transform to genereate a gaussian distribution
         x[j]= sqrt(-2.0* log(r1))* cos(2*M_PI*r2);
      }
   }

   delete[] seeds;
}



#else

void generateNormal(int n, int d, double *x, MPI_Comm comm)
{
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	srand48((unsigned)time(NULL)+rank);

    for (int j=0; j<n*d ; j++){
		double r1 = drand48();
		double r2 = drand48() ;
		//Box muller transform to genereate a gaussian distribution
		x[j]= sqrt(-2.0* log(r1))* cos(2*M_PI*r2);
	}

}

#endif


