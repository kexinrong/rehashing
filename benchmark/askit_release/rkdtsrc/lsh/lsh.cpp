# include "lsh.h"
# include "direct_knn.h"
//#include "random123wrapper.h"
# include <sys/time.h>
# include <cstdio>
# include <iostream>
# include <fstream>
# include <cstdlib>
# include <cstring>
# include <cmath>
# include <vector>
#include <algorithm>
#include <direct_knn.h>
#include <ompUtils.h>

# include <blas.h>
# include <omp.h>

using namespace std;
using namespace knn::lsh;







void knn::lsh::setup (int dim, double R, long K, long L, 
                      double *&rPP, double *& a, double *& b) {
                    
   double W= 4.0 * R; //Scale W by search radius, rather than scaling points.
   generateHashFunctions (a, b, K, L,W, dim);
   generateRandomNos(K, rPP);

}




void knn::lsh::generateRandomNos (long K, double *& rPP){
   rPP= new double [K];
   //srand (time(NULL)); //init random number generator

   for (unsigned long i=0; i< K; i++){
      rPP[i]= round((double)rand()/10000.0);
   }
}



void knn::lsh::generateHashFunctions (double *& a, double *& b, long K, long L, double W, long nthDimension){

   //init a and b, L rows and K colums
   a = new double[L*K*nthDimension];
   b = new double[L*K];

   double Winverse = 1.0/W;

   //generate b-which is uniformly distributed
   for (unsigned long j=0; j<L*K;j++){
      b[j] = (double)rand()/(double)RAND_MAX * W;
   }

   for (unsigned long j=0; j<L * K* nthDimension;j++){
      double rand_no = (double)rand()/(double)RAND_MAX;
      //Box muller transform to generate a gaussian distribution
      a[j]= sqrt(-2.0* log(rand_no))* cos(2*M_PI*rand_no) * Winverse; 
   }

}




void knn::lsh::compute_hash_values  (point_type *points, int num_points, 
                double *a, double *b, long K, long L, int dim, 
                double *rPP, unsigned int* bucketIDs){

   double *hashVals= new double [num_points * K * L];
   unsigned int bucketID;


   int iD = dim;
   int iN = num_points;
   int iK = K;
   int iL = L;
   int kL = K*L;
   int one = 1;
   double alpha = 1.0;
   double beta = 0.0;

   #pragma omp parallel for schedule(static)
   for(int i = 0; i < K*L*num_points; i++) {
      hashVals[i] = b[i%kL];
   }


   #pragma omp parallel 
   {
      int omp_num_points, last_omp_num_points;
      int i = omp_get_thread_num();
      int numt = omp_get_num_threads();
      omp_num_points = num_points / numt;
      last_omp_num_points = num_points - (omp_num_points * (numt -1));
         
      //This thread's number of points
      int npoints = (i == numt-1) ? last_omp_num_points : omp_num_points;       

      dgemm("T", "N", &kL, &npoints, &iD, &alpha, a, &iD,
                 points + (iD * i*omp_num_points), &iD,
                 &alpha, hashVals+ (i*omp_num_points*(K*L)), &kL);
   }

   #pragma omp parallel for schedule(static)
   for (long m=0; m< num_points * K * L; m++){
      hashVals[m] = floor (hashVals[m]);
   }

   double *hashIDs = new double[L*num_points];
   #pragma omp parallel 
   {
      int omp_num_points, last_omp_num_points;
      int t = omp_get_thread_num();
      int numt = omp_get_num_threads();
      omp_num_points = num_points / numt;
      last_omp_num_points = num_points - (omp_num_points * (numt -1));
         
      //This thread's number of points
      int npoints = (t == numt-1) ? last_omp_num_points : omp_num_points;       

      if( L > 1 ) {
         for(int i = 0; i < L; i++)
            dgemv("T", &iK, &npoints, &alpha, hashVals + K*i + (kL*t*omp_num_points), &kL, rPP, &one, &beta,
                 (hashIDs+i*iN) + (t*omp_num_points), &one);
      } else {
         dgemv("T", &iK, &npoints, &alpha, hashVals + (kL*t*omp_num_points), &kL, rPP, &one, &beta,
                 hashIDs + (t*omp_num_points), &one);
      }

   }

   if( L > 1 ) {
      #pragma omp parallel for schedule(static)
      for (long i =0; i < num_points; i++){
         //for every point
         for ( long j=0; j< L; j++){
            bucketIDs[i*L+j] = (unsigned int)(((unsigned long)(hashIDs[j*num_points+i])) % LSH_KEY_MAX);
         }
      }
   } else {
      #pragma omp parallel for schedule(static)
      for (long i =0; i < num_points; i++){
         bucketIDs[i] = (unsigned int)(((unsigned long)(hashIDs[i])) % LSH_KEY_MAX);
      }
   }

   delete[] hashVals;
   delete[] hashIDs;

}









