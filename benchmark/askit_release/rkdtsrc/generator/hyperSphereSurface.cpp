#include <direct_knn.h>
#include <omp.h>
#include <mpi.h>
#include <cmath>
#include <cstdlib>
#include <cassert>

#ifndef __APPLE__
void generateUnitHypersphere(int n, int d, double *x, MPI_Comm comm) {
 
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

         seeds = (struct drand48_data*)malloc(p*sizeof(struct drand48_data));
         srand48((long)time(NULL) + rank*size);
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

   double *norms = new double[n];
   knn::sqnorm ( x, n, d, norms );
   #pragma omp parallel for
   for(int i = 0; i < n; i++) norms[i] = 1/sqrt(norms[i]);

   //project points to surface by dividing by their radius
   #pragma omp parallel for
   for(int i = 0; i < n; i++) {
      for(int j = 0; j < d; j++) x[d*i+j] *= norms[i];
   }

   delete[] norms;
   free(seeds);
}



void generateUnitHypersphereEmbedded(int n, int d, int spatialD, double *x, MPI_Comm comm) {
  
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
 
  assert( spatialD > d);
  d += 1; //Want data points to have d "real" dimensions
 
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
      srand48((long)time(NULL) + rank*size);
      for(int i = 0; i < p; i++)
        srand48_r(lrand48(), &(seeds[i]));
    }
#pragma omp barrier
    
#pragma omp for
    for( int i = 0; i < n; i++ ) {
      for (int j=0; j<d ; j++){
        double r1, r2;
        drand48_r(&(seeds[t]), &r1);
        drand48_r(&(seeds[t]), &r2);
        //Box muller transform to genereate a gaussian distribution
        x[i*spatialD+j]= sqrt(-2.0* log(r1))* cos(2*M_PI*r2);
      }
      for (int j=d; j<spatialD ; j++){
        x[i*spatialD+j]= 0.0;
      }
    }
  }
  
  double *norms = new double[n];
  knn::sqnorm ( x, n, spatialD, norms );
#pragma omp parallel for
  for(int i = 0; i < n; i++) norms[i] = 1/sqrt(norms[i]);
  
  //project points to surface by dividing by their radius
#pragma omp parallel for
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < d; j++) x[spatialD*i+j] *= norms[i];
  }
  
  //Shift hyperphere to random center.
  double *c = new double[spatialD];
  for(int i = 0; i < spatialD; i++) 
    c[i] = drand48();
  
#pragma omp parallel for
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < spatialD; j++)
      x[i*spatialD+j] += c[j];
  }
  
  
  delete[] norms;
  delete[] seeds;
}
#else

void generateUnitHypersphere(int n, int d, double *x, MPI_Comm comm) {
 
   int rank, size;
   MPI_Comm_rank(comm, &rank);
   MPI_Comm_size(comm, &size);

	 srand48((long)time(NULL) + rank*size);
	 
	 for (int j=0; j<n*d ; j++){
		 double r1 = drand48();
		 double r2 = drand48();
		 //Box muller transform to genereate a gaussian distribution
		 x[j]= sqrt(-2.0* log(r1))* cos(2*M_PI*r2);
	 }

   double *norms = new double[n];
   knn::sqnorm ( x, n, d, norms );
#pragma omp parallel for
   for(int i = 0; i < n; i++) norms[i] = 1/sqrt(norms[i]);

   //project points to surface by dividing by their radius
#pragma omp parallel for
   for(int i = 0; i < n; i++) {
      for(int j = 0; j < d; j++) x[d*i+j] *= norms[i];
   }

   delete[] norms;
}


void generateUnitHypersphereEmbedded(int n, int d, int spatialD, double *x, MPI_Comm comm) {
  
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
 
  assert( spatialD > d);
  d += 1; //Want data points to have d "real" dimensions
	srand48((long)time(NULL) + rank*size);
	
	for( int i = 0; i < n; i++ ) {
		for (int j=0; j<d ; j++){
			double r1 = drand48();	
			double r2 = drand48();
			//Box muller transform to genereate a gaussian distribution
			x[i*spatialD+j]= sqrt(-2.0* log(r1))* cos(2*M_PI*r2);
		}
		for (int j=d; j<spatialD ; j++){
			x[i*spatialD+j]= 0.0;
		}	
	}		
  
  
  double *norms = new double[n];
  knn::sqnorm ( x, n, spatialD, norms );
#pragma omp parallel for
  for(int i = 0; i < n; i++) norms[i] = 1/sqrt(norms[i]);
  
  //project points to surface by dividing by their radius
#pragma omp parallel for
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < d; j++) x[spatialD*i+j] *= norms[i];
  }
  
  //Shift hyperphere to random center.
  double *c = new double[spatialD];
  for(int i = 0; i < spatialD; i++) 
    c[i] = drand48();
  
#pragma omp parallel for
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < spatialD; j++)
      x[i*spatialD+j] += c[j];
  }
  
  
  delete[] norms;
}


#endif

