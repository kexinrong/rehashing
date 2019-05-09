#include <direct_knn.h>
#include <omp.h>
#include <mpi.h>
#include <cmath>
#include <cstdlib>


#ifndef __APPLE__

void generateUniform(int n, int d, double *x, MPI_Comm comm) {
 
   int rank, size;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &size);

   struct drand48_data *seeds;
 
   #pragma omp parallel
   {  
      int t = omp_get_thread_num();
      int p = omp_get_num_threads();

      //Seed each thread's prng
      #pragma omp master
      {
         seeds = new struct drand48_data[p];
         srand48((long)time(NULL) + rank*size);
         for(int i = 0; i < p; i++)
            srand48_r(lrand48(), &(seeds[i]));
      }
      #pragma omp barrier

      #pragma omp for
      for (int j=0; j<n*d ; j++){
         drand48_r(&(seeds[t]), &(x[j]));
      }
   }

   delete[] seeds;
}

#else

void generateUniform(int n, int d, double *x, MPI_Comm comm) {
	
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	srand48( (unsigned)time(NULL) + rank );

	for(int i = 0; i < n*d; i++)
		x[i] = drand48();

}
#endif




