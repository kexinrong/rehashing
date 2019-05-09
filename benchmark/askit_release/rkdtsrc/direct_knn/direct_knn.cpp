#include <mpi.h>
#include <iostream>
#include <utility>
#include <vector>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <numeric>
#include <blas.h>
#include <omp.h>
#include <ompUtils.h>
#include "direct_knn.h"
#include "parallelIO.h"
#include <cassert>
#include "knnreduce.h"
#include <queue>

#include "verbose.h"

double dgemmtime;
double sqnormtime;
double addtime;

using namespace std;


void knn::sqnorm ( double *a, long n, int dim, double *b) {
  int one = 1;
  bool omptest = n*(long)dim > 10000L;

  #pragma omp parallel if(omptest)
  {
    #pragma omp for schedule(static)
    for(int i = 0; i < n; i++) {
       b[i] = ddot(&dim, &(a[dim*i]), &one, &(a[dim*i]), &one);
    }
  }
}


pair<double, long> *knn::directKQuery
           ( double *ref, double *query, long n, long m, long k, int dim )
{
   double *dist = new double[m*n];
   pair<double, long> *pdists =NULL;
   pair<double, long> *result = new pair<double, long>[m*k];
   int num_neighbors = (k<n) ? k : n;

   knn::compute_distances( ref, query, n, m, dim, dist );

   pdists = new pair<double, long>[m*n];
   //Copy each distance into a pair along with its index for sorting
   #pragma omp parallel for
   for( int i = 0; i < m; i++ )
     for( int j = 0; j < n; j++ )
       pdists[i*n+j] = pair<double, long>(dist[i*n+j], j);


   //Find nearest neighbors and copy to result.
   #pragma omp parallel for
   for( int h = 0; h < m; h++ ) {
     int curr_idx = 0;
     double curr_min = DBL_MAX;
     int swaploc = 0;
     for(int j = 0; j < num_neighbors; j++) {
       curr_min = DBL_MAX;
       for(int a = curr_idx; a < n; a++) {
         if(pdists[h*n+a].first < curr_min) {
            swaploc = a;
            curr_min = pdists[h*n+a].first;
         }
       }
       result[h*k+j] = pdists[h*n+swaploc];
       pdists[h*n+swaploc] = pdists[h*n+curr_idx];
       curr_idx++;
     }
   }

   if(num_neighbors < k) {
     //Pad the k-min matrix with bogus values that will always be higher than real values
     #pragma omp parallel for
     for( int i = 0; i < m; i++ )
       for( int j = num_neighbors; j < k; j++ )
         result[i*n+j] = pair<double, long>(DBL_MAX, -1L);
   }

   delete [] dist;
   if(pdists)
     delete [] pdists;

   return result;
}


// n should be small, specially designed for tree leaf search
void knn::directKQuery_small_a2a( double *ref, long n, int dim, int k,
                                  std::pair<double, long> *result,
                                  double *dist, double *sqnormr, double *sqnormq )
{
    bool dealloc_dist = false;
    bool dealloc_sqnormr = false;
    bool dealloc_sqnormq = false;

    int maxt = omp_get_max_threads();

    if(!dist) {
        dist = new double[n*n];
        dealloc_dist = true;
    }
    if(!sqnormr) {
        sqnormr = new double[n];
        dealloc_sqnormr = true;
    }
    if(!sqnormq) {
        sqnormq = new double[n];
        dealloc_sqnormq = true;
    }

    int num_neighbors = (k<n) ? k : n;
    knn::compute_distances( ref, ref, n, n, dim, dist, sqnormr, sqnormq );

    #pragma omp parallel
    {
        priority_queue<pair<double, long>, vector< pair<double, long> >, maxheap_comp> maxheap;
        #pragma omp for
        for(int i = 0; i < n; i++) {
            while(!maxheap.empty()) maxheap.pop();
            for(int j = 0; j < num_neighbors; j++) {
                double unique_dist = min(dist[i*n+j], dist[j*n+i]);
                maxheap.push(make_pair<double, long>(unique_dist, j));
            }
            for(int j = num_neighbors; j < n; j++) {
                double unique_dist = min(dist[i*n+j], dist[j*n+i]);
                maxheap.push(make_pair<double, long>(unique_dist, j));
                maxheap.pop();
            }
            for(int j = num_neighbors-1; j <= 0; j--) {
                result[i*k+j] = maxheap.top();
                maxheap.pop();
            }
        }   // end for i
   }    // end parallel section

    if( num_neighbors < k ) {
        //Pad the k-min matrix with bogus values that will always be higher than real values
        #pragma omp parallel if( n > 128 * maxt )
        {
            #pragma omp for schedule(static)
            for( int i = 0; i < n; i++ )
            {
                for( int j = num_neighbors; j < k; j++ ) {
                    result[i*k+j].first = DBL_MAX;
                    result[i*k+j].second =  -1L;
                }
            }
        }
    }

    if(dealloc_dist) {
        delete[] dist;
        dist = NULL;
    }
    if(dealloc_sqnormr) {
        delete [] sqnormr;
        sqnormr = NULL;
    }
    if(dealloc_sqnormq) {
        delete [] sqnormq;
        sqnormq = NULL;
    }
}



/*
void knn::directKQueryLowMem
           ( double *ref, double *query, long n, long m, long k, int dim, pair<double, long> *result,
             double *dist, double* sqnormr, double* sqnormq )
{

   dgemmtime = sqnormtime = addtime = 0.0;


   register int num_neighbors = (k<n) ? k : n;

   //If performing the entire query at once will require too much memory, split it into smaller
     //pieces.
   int blocksize = getBlockSize(n, m);

   bool dealloc_dist = false;
   bool dealloc_sqnormr = false;
   bool dealloc_sqnormq = false;

   int maxt = omp_get_max_threads();

   assert(blocksize > 0);
   int nblocks = (int) m / blocksize;
   int iters = (int) ceil((double)m/(double)blocksize);

   if(!dist) {
      dist = new double[n*blocksize];
      dealloc_dist = true;
   }
   if(!sqnormr) {
     sqnormr = new double[n];      
     dealloc_sqnormr = true;
   }
   if(!sqnormq) {
     sqnormq = new double[blocksize];      
     dealloc_sqnormq = true;
   }
 
   bool useSqnormrInput = false;

   for(int i = 0; i < iters; i++) {
      double *currquery = query + i*blocksize*dim;
      if( (i == iters-1) && (m % blocksize) ) {
         int lastblocksize = m%blocksize;
         knn::compute_distances( ref, currquery, n, lastblocksize, dim, dist, sqnormr, sqnormq, useSqnormrInput );
         bool omptest = lastblocksize > 50*maxt || n > 1024;
         #pragma omp parallel if( omptest )
         {
            int hnpj;
            double nearest_dist[num_neighbors]; 
            long nearest_ids[num_neighbors]; 
            //Find nearest neighbors and copy to result.
            #pragma omp for 
            for( int h = 0; h < lastblocksize; h++ ) {
               int querynum = i*blocksize + h;
               int j;
               for(j = 0; j < num_neighbors; j++) {
                  nearest_dist[j] = DBL_MAX;
                  nearest_ids[j] = -1L;
               }
               for(j = 0; j < n; j++) {
                  int curr = 0;
                  hnpj = h*n + j;
                  if( nearest_dist[curr] > dist[hnpj] ) {
                     while( curr < num_neighbors-1 && nearest_dist[curr+1] > dist[hnpj])
                        curr++;
                     int a;
                     for(a = 0; a < curr; a++) { 
                        nearest_dist[a] = nearest_dist[a+1];
                        nearest_ids[a] = nearest_ids[a+1];
                     }
                     nearest_dist[curr] = dist[hnpj];
                     nearest_ids[curr] = j;
                  }
               }

               int qnk = querynum*k;
               int qnkpj;
               for(j = 0; j < num_neighbors; j++) {
                 qnkpj = qnk + j;
                 result[qnkpj].first = nearest_dist[num_neighbors-1 - j];
                 result[qnkpj].second = nearest_ids[num_neighbors-1 - j];
               }
            }
         } 

      } else {
         knn::compute_distances( ref, currquery, n, blocksize, dim, dist, sqnormr, sqnormq, useSqnormrInput);
         bool omptest = blocksize > 50*maxt || n > 1024;
         #pragma omp parallel if( omptest )
         {
            int hnpj;
            double nearest_dist[num_neighbors]; 
            long nearest_ids[num_neighbors]; 
            //Find nearest neighbors and copy to result.
            #pragma omp for 
            for( int h = 0; h < blocksize; h++ ) {
               int querynum = i*blocksize + h;
               int j;
               for(j = 0; j < num_neighbors; j++) {
                  nearest_dist[j] = DBL_MAX;
                  nearest_ids[j] = -1L;
               }
               for(j = 0; j < n; j++) {
                  int curr = 0;
                  hnpj = h*n + j;
                  if( nearest_dist[curr] > dist[hnpj] ) {
                     while( curr < num_neighbors-1 && nearest_dist[curr+1] > dist[hnpj])
                        curr++;
                     int a;
                     for(a = 0; a < curr; a++) { 
                        nearest_dist[a] = nearest_dist[a+1];
                        nearest_ids[a] = nearest_ids[a+1];
                     }
                     nearest_dist[curr] = dist[hnpj];
                     nearest_ids[curr] = j;
                  }
               }
 
               int qnk = querynum*k;
               int qnkpj;
               for(j = 0; j < num_neighbors; j++) {
                 qnkpj = qnk + j;
                 result[qnkpj].first = nearest_dist[num_neighbors-1 - j];
                 result[qnkpj].second = nearest_ids[num_neighbors-1 - j];
               }
            }
         } 
      }
      useSqnormrInput = true; //Already computed sqnormr

   }


   if( num_neighbors < k ) {
      //Pad the k-min matrix with bogus values that will always be higher than real values
      #pragma omp parallel if( m > 128 * maxt )
      { 
         #pragma omp for schedule(static)
         for( int i = 0; i < m; i++ )
         {
            for( int j = num_neighbors; j < k; j++ ) {
               result[i*k+j].first = DBL_MAX;
               result[i*k+j].second =  -1L;
            }
         }
      }
   }

   if(dealloc_dist)
     delete[] dist;
   if(dealloc_sqnormr)
     delete [] sqnormr;
   if(dealloc_sqnormq)
     delete [] sqnormq;
}
*/


void knn::directKQueryLowMem
           ( double *ref, double *query, long n, long m, long k, int dim, pair<double, long> *result,
             double *dist, double* sqnormr, double* sqnormq )
{
    dgemmtime = sqnormtime = addtime = 0.0;

    register int num_neighbors = (k<n) ? k : n;

    //If performing the entire query at once will require too much memory,
    //split it into smaller pieces.
    int blocksize = getBlockSize(n, m);

    bool dealloc_dist = false;
    bool dealloc_sqnormr = false;
    bool dealloc_sqnormq = false;

    int maxt = omp_get_max_threads();

    assert(blocksize > 0);
    int nblocks = (int) m / blocksize;
    int iters = (int) ceil((double)m/(double)blocksize);

    if(!dist) {
        dist = new double[n*blocksize];
        dealloc_dist = true;
    }
    if(!sqnormr) {
        sqnormr = new double[n];
        dealloc_sqnormr = true;
    }
    if(!sqnormq) {
        sqnormq = new double[blocksize];
        dealloc_sqnormq = true;
    }

    bool useSqnormrInput = false;

    for(int i = 0; i < iters; i++) {
        double *currquery = query + i*blocksize*dim;
        if( (i == iters-1) && (m % blocksize) ) {
            int lastblocksize = m%blocksize;
            double dummy_t = omp_get_wtime();
            knn::compute_distances(ref, currquery, n, lastblocksize, dim, dist,
                                    sqnormr, sqnormq, useSqnormrInput);
            COMPUTE_DIST_T_ += omp_get_wtime()-dummy_t;
            dummy_t = omp_get_wtime();
            #pragma omp parallel
            {
                priority_queue<pair<double, long>, vector<pair<double, long> >, maxheap_comp> maxheap;
                #pragma omp for
                for(int h = 0; h < lastblocksize; h++) {
                    while(!maxheap.empty()) maxheap.pop();
                    int querynum = i*blocksize + h;
                    for(int j = 0; j < num_neighbors; j++)
                        maxheap.push(make_pair<double, long>(dist[h*n+j], j));
                    for(int j = num_neighbors; j < n; j++) {
                        maxheap.push(make_pair<double, long>(dist[h*n+j], j));
                        maxheap.pop();
                    }
                    for(int j = num_neighbors-1; j >= 0; j--) {
                        result[querynum*k+j] = maxheap.top();
                        maxheap.pop();
                    }
                }   // end for h
            }
            MAX_HEAP_T_ += omp_get_wtime()-dummy_t;
        } // end if last block
        else {
            double dummy_t = omp_get_wtime();
            knn::compute_distances( ref, currquery, n, blocksize, dim, dist,
                                    sqnormr, sqnormq, useSqnormrInput);
            COMPUTE_DIST_T_ += omp_get_wtime()-dummy_t;
            dummy_t = omp_get_wtime();
            #pragma omp parallel
            {
                priority_queue<pair<double, long>, vector<pair<double, long> >, maxheap_comp> maxheap;
                #pragma omp for
                for(int h = 0; h < blocksize; h++) {
                    while(!maxheap.empty()) maxheap.pop();
                    int querynum = i*blocksize + h;
                    for(int j = 0; j < num_neighbors; j++)
                        maxheap.push(make_pair<double, long>(dist[h*n+j], j));
                    for(int j = num_neighbors; j < n; j++) {
                        maxheap.push(make_pair<double, long>(dist[h*n+j], j));
                        maxheap.pop();
                    }
                    for(int j = num_neighbors-1; j >= 0; j--) {
                        result[querynum*k+j] = maxheap.top();
                        maxheap.pop();
                    }
                }   // end for h
            } // end pragma omp parallel
            MAX_HEAP_T_ += omp_get_wtime()-dummy_t;
        }
        useSqnormrInput = true; //Already computed sqnormr
    }

    if( num_neighbors < k ) {
        //Pad the k-min matrix with bogus values that will always be higher than real values
        #pragma omp parallel if( m > 128 * maxt )
        {
            #pragma omp for schedule(static)
            for( int i = 0; i < m; i++ )
            {
                for( int j = num_neighbors; j < k; j++ ) {
                    result[i*k+j].first = DBL_MAX;
                    result[i*k+j].second =  -1L;
                }
            }
        }
    }

    if(dealloc_dist) delete[] dist;
    if(dealloc_sqnormr) delete [] sqnormr;
    if(dealloc_sqnormq) delete [] sqnormq;
}



/*
void knn::directKQueryLowMem
           ( double *ref, double *query, long n, long m, long k, int dim, pair<double, long> *result,
             double *dist, double* sqnormr, double* sqnormq ) {

    dgemmtime = sqnormtime = addtime = 0.0;

    register int num_neighbors = (k<n) ? k : n;

    //If performing the entire query at once will require too much memory,
    //split it into smaller pieces.
    int blocksize = getBlockSize(n, m);

    bool dealloc_dist = false;
    bool dealloc_sqnormr = false;
    bool dealloc_sqnormq = false;

    int maxt = omp_get_max_threads();

    assert(blocksize > 0);
    int nblocks = (int) m / blocksize;
    int iters = (int) ceil((double)m/(double)blocksize);

    if(!dist) {
        dist = new double[n*blocksize];
        dealloc_dist = true;
    }
    if(!sqnormr) {
        sqnormr = new double[n];
        dealloc_sqnormr = true;
    }
    if(!sqnormq) {
        sqnormq = new double[blocksize];
        dealloc_sqnormq = true;
    }

    bool useSqnormrInput = false;

    for(int i = 0; i < iters; i++) {
        double *currquery = query + i*blocksize*dim;
        if( (i == iters-1) && (m % blocksize) ) {
            int lastblocksize = m%blocksize;
            knn::compute_distances(ref, currquery, n, lastblocksize, dim, dist,
                                    sqnormr, sqnormq, useSqnormrInput);
            #pragma omp parallel
            {
                vector< pair<double, long> > tmpvec(n);
                #pragma omp for
                for(int h = 0; h < lastblocksize; h++) {
                    int querynum = i*blocksize + h;
                    for(int j = 0; j < n; j++)
                        tmpvec[j] = make_pair<double, long>(dist[h*n+j], j);
                    std::sort(tmpvec.begin(), tmpvec.end());
                    for(int j = 0; j < num_neighbors; j++)
                        result[querynum*k+j] = tmpvec[j];
                }   // end for h
            }
        } // end if last block
        else {
            knn::compute_distances( ref, currquery, n, blocksize, dim, dist,
                                    sqnormr, sqnormq, useSqnormrInput);
            #pragma omp parallel
            {
                vector< pair<double, long> > tmpvec(n);
                #pragma omp for
                for(int h = 0; h < blocksize; h++) {
                    int querynum = i*blocksize + h;
                    for(int j = 0; j < n; j++)
                        tmpvec[j] = make_pair<double, long>(dist[h*n+j], j);
                    std::sort(tmpvec.begin(), tmpvec.end());
                    for(int j = num_neighbors-1; j >= 0; j--)
                        result[querynum*k+j] = tmpvec[j];
                }   // end for h
            } // end pragma omp parallel
        }
        useSqnormrInput = true; //Already computed sqnormr
    }

    if( num_neighbors < k ) {
        //Pad the k-min matrix with bogus values that will always be higher than real values
        #pragma omp parallel if( m > 128 * maxt )
        {
            #pragma omp for schedule(static)
            for( int i = 0; i < m; i++ )
            {
                for( int j = num_neighbors; j < k; j++ ) {
                    result[i*k+j].first = DBL_MAX;
                    result[i*k+j].second =  -1L;
                }
            }
        }
    }

    if(dealloc_dist) delete[] dist;
    if(dealloc_sqnormr) delete [] sqnormr;
    if(dealloc_sqnormq) delete [] sqnormq;
}
*/

void knn::directRQuery
           ( double *ref, double *query, long n, long m, double R, int dim, long *glob_ids,
             int **neighbor_count, pair<double, long> **neighbors  ) {

   double *dist = NULL;
   pair<double, long> *pdists =NULL;
   vector< pair<double, long> >*temp_neighbors = NULL;
   int *neighbor_prefix = new int[m];
   *neighbor_count = new int[m];
   int total_neighbors = 0;
   #pragma omp parallel for schedule(static)
   for( int i = 0; i < m; i++ )
      (*neighbor_count)[i] = 0;
      


   //If performing the entire query at once will require too much memory, split it into smaller
     //pieces.
   int blocksize = getBlockSize(n, m);

   assert(blocksize > 0);
   int nblocks = (int) m / blocksize;
   int iters = (int) ceil((double)m/(double)blocksize);

   dist = new double[n*blocksize];
   temp_neighbors = new vector< pair<double, long> >[m];

   for(int i = 0; i < iters; i++) {
      double *currquery = query + i*blocksize*dim;
      if( (i == iters-1) && (m % blocksize) ) {
         int lastblocksize = m%blocksize;
       
         knn::compute_distances( ref, currquery, n, lastblocksize, dim, dist );
   
         //Copy each distance < R into a pair along with its index
         #pragma omp parallel for schedule(dynamic) reduction(+:total_neighbors)
         for( int l = 0; l < lastblocksize; l++ )
         {
            for( int j = 0; j < n; j++ ) {
               if( dist[l*n+j] <= R ) {
                   int querynum = i*blocksize + l;
                   int s = temp_neighbors[querynum].size();
                   temp_neighbors[querynum].resize(s+1);
                   temp_neighbors[querynum][s].first = dist[l*n + j];   
                   temp_neighbors[querynum][s].second =  glob_ids[j];
                   total_neighbors++; 
                }
             }
         }
      } else {
         knn::compute_distances( ref, currquery, n, blocksize, dim, dist );
   
         //Copy each distance < R into a pair along with its index
         #pragma omp parallel for schedule(dynamic) reduction(+:total_neighbors)
         for( int l = 0; l < blocksize; l++ )
         {
            for( int j = 0; j < n; j++ ) {
               if( dist[l*n+j] <= R ) {
                   int querynum = i*blocksize + l;
                   int s = temp_neighbors[querynum].size();
                   temp_neighbors[querynum].resize(s+1);
                   temp_neighbors[querynum][s].first = dist[l*n + j];   
                   temp_neighbors[querynum][s].second =  glob_ids[j];
                   total_neighbors++; 
               }
            }
         }
      }
   }

   #pragma omp parallel for schedule(static)
   for( int i = 0; i < m; i++ ) (*neighbor_count)[i] = temp_neighbors[i].size();

   *neighbors = new pair<double, long>[total_neighbors];
   //Compute prefix sum of neighbor count for array indices
   omp_par::scan(*neighbor_count, neighbor_prefix, m);
   #pragma omp parallel
   {
     #pragma omp for schedule(static) 
     for( int i = 0; i < m; i++ ) {
        for( int j = 0; j < (*neighbor_count)[i]; j++ ) {
            (*neighbors)[neighbor_prefix[i]+j].first = temp_neighbors[i][j].first;
            (*neighbors)[neighbor_prefix[i]+j].second = temp_neighbors[i][j].second;
        }
     }
   }


   delete [] temp_neighbors;
   delete [] neighbor_prefix;   
   delete [] dist;
} 



void knn::directRQueryIndividual 
( double *ref, double *query, long n, long m, double *R, int dim, long *glob_ids,
 int **neighbor_count, pair<double, long> **neighbors  ) {
  
  double *dist = NULL;
  pair<double, long> *pdists =NULL;
  vector< pair<double, long> >*temp_neighbors = NULL;
  int *neighbor_prefix = new int[m];
  
  *neighbor_count = new int[m];
  int total_neighbors = 0;
#pragma omp parallel for schedule(static)
  for( int i = 0; i < m; i++ )
    (*neighbor_count)[i] = 0;
  
  
  
  //If performing the entire query at once will require too much memory, split it into smaller
  //pieces.
   if( m > KNN_MAX_BLOCK_SIZE ) { 
    int blocksize = std::min((long)KNN_MAX_BLOCK_SIZE, m); //number of query points handled in a given iteration
    assert(blocksize > 0);
    int nblocks = (int) m / blocksize;
    int iters = (int) ceil((double)m/(double)blocksize);
    temp_neighbors = new vector< pair<double, long> >[m];
    dist = new double[n*blocksize];
    
    for(int i = 0; i < iters; i++) {
      double *currquery = query + i*blocksize*dim;
      if( (i == iters-1) && (m % blocksize) ) {
        int lastblocksize = m%blocksize;
        
        knn::compute_distances( ref, currquery, n, lastblocksize, dim, dist );
        
        //Copy each distance < R into a pair along with its index
#pragma omp parallel for schedule(static) reduction(+:total_neighbors)
        for( int l = 0; l < lastblocksize; l++ )
        {
          for( int j = 0; j < n; j++ ) {
            int querynum = i*blocksize + l;
            if( dist[l*n+j] <= R[querynum] ) {
              int s = temp_neighbors[querynum].size();
              temp_neighbors[querynum].resize(s+1);
              temp_neighbors[querynum][s].first = dist[l*n + j];   
              temp_neighbors[querynum][s].second =  glob_ids[j];
              total_neighbors++; 
            }
          }
        }
      } else {
        knn::compute_distances( ref, currquery, n, blocksize, dim, dist );
        
        //Copy each distance < R into a pair along with its index
#pragma omp parallel for schedule(static) reduction(+:total_neighbors)
        for( int l = 0; l < blocksize; l++ )
        {
          for( int j = 0; j < n; j++ ) {
            int querynum = i*blocksize + l;
            if( dist[l*n+j] <= R[querynum] ) {
              int s = temp_neighbors[querynum].size();
              temp_neighbors[querynum].resize(s+1);
              temp_neighbors[querynum][s].first = dist[l*n + j];   
              temp_neighbors[querynum][s].second =  glob_ids[j];
              total_neighbors++; 
            }
          }
        }
      }
    }
    
#pragma omp parallel for schedule(static)
    for( int i = 0; i < m; i++ ) (*neighbor_count)[i] = temp_neighbors[i].size();
    
    *neighbors = new pair<double, long>[total_neighbors];
    //Compute prefix sum of neighbor count for array indices
    omp_par::scan(*neighbor_count, neighbor_prefix, m);
#pragma omp parallel
    {
#pragma omp for schedule(static) 
      for( int i = 0; i < m; i++ ) {
        for( int j = 0; j < (*neighbor_count)[i]; j++ ) {
          (*neighbors)[neighbor_prefix[i]+j].first = temp_neighbors[i][j].first;
          (*neighbors)[neighbor_prefix[i]+j].second = temp_neighbors[i][j].second;
        }
      }
    }
    delete [] temp_neighbors;
    
  } else {
    
    dist = new double[m*n];
    knn::compute_distances( ref, query, n, m, dim, dist );
    
    pdists = new pair<double, long>[m*n];
    
    //Copy each distance < R into a pair along with its index
#pragma omp parallel for schedule(static) reduction(+:total_neighbors)
    for( int i = 0; i < m; i++ )
    {
      for( int j = 0; j < n; j++ ) {
        if( dist[i*n+j] <= R[i] ) {
          pdists[i*n+(*neighbor_count)[i]].first = dist[i*n+j];
          pdists[i*n+(*neighbor_count)[i]++].second =  glob_ids[j];
          total_neighbors++;
        }
      }
    }
    
    *neighbors = new pair<double, long>[total_neighbors];
    //Compute prefix sum of neighbor count for array indices
    omp_par::scan(*neighbor_count, neighbor_prefix, m);
#pragma omp parallel
    {
#pragma omp for schedule(static) 
      for( int i = 0; i < m; i++ )
      {
        for( int j = 0; j < (*neighbor_count)[i]; j++ ) {
          (*neighbors)[neighbor_prefix[i]+j].first = pdists[i*n+j].first;
          (*neighbors)[neighbor_prefix[i]+j].second = pdists[i*n+j].second;
        }
      }
    }
    
    delete [] pdists;
  }
  
  delete [] neighbor_prefix;   
  delete [] dist;
} 




void knn::directRQueryIndividualK
( double *ref, double *query, long n, long m, int k, double *R, int dim, long *glob_ids,
 int **neighbor_count, pair<double, long> **neighbors  ) {
  
  double *dist = NULL;
  pair<double, long> *pdists =NULL;
  vector< pair<double, long> >*temp_neighbors = NULL;
  int *neighbor_prefix = new int[m];
  
  *neighbor_count = new int[m];
  int total_neighbors = 0;
#pragma omp parallel for schedule(static)
  for( int i = 0; i < m; i++ )
    (*neighbor_count)[i] = 0;
  
  
  
  //If performing the entire query at once will require too much memory, split it into smaller
  //pieces.
  int blocksize;
  if( m > KNN_MAX_BLOCK_SIZE ) { 
     blocksize = std::min((long)KNN_MAX_BLOCK_SIZE, m); //number of query points handled in a given iteration
  } else {
    blocksize = m;
  }
    assert(blocksize > 0);
    int nblocks = (int) m / blocksize;
    int iters = (int) ceil((double)m/(double)blocksize);
    temp_neighbors = new vector< pair<double, long> >[m];
    dist = new double[n*blocksize];
    
    for(int i = 0; i < iters; i++) {
      double *currquery = query + i*blocksize*dim;
      if( (i == iters-1) && (m % blocksize) ) {
        int lastblocksize = m%blocksize;
        
        knn::compute_distances( ref, currquery, n, lastblocksize, dim, dist );
        
        //Copy each distance < R into a pair along with its index
#pragma omp parallel for schedule(static) reduction(+:total_neighbors)
        for( int l = 0; l < lastblocksize; l++ )
        {
          int querynum = i*blocksize + l;
          for( int j = 0; j < n; j++ ) {
            if( dist[l*n+j] <= R[querynum] ) {
              int s = temp_neighbors[querynum].size();
              temp_neighbors[querynum].resize(s+1);
              temp_neighbors[querynum][s].first = dist[l*n + j];   
              temp_neighbors[querynum][s].second =  glob_ids[j];
              total_neighbors++; 
            }
          }
          std::sort(temp_neighbors[querynum].begin(), temp_neighbors[querynum].end());
          temp_neighbors[querynum].resize( std::min(temp_neighbors[querynum].size(), (size_t)k) );
        }
      } else {
        knn::compute_distances( ref, currquery, n, blocksize, dim, dist );
        
        //Copy each distance < R into a pair along with its index
#pragma omp parallel for schedule(static) reduction(+:total_neighbors)
        for( int l = 0; l < blocksize; l++ )
        {
          int querynum = i*blocksize + l;
          for( int j = 0; j < n; j++ ) {
            if( dist[l*n+j] <= R[querynum] ) {
              int s = temp_neighbors[querynum].size();
              temp_neighbors[querynum].resize(s+1);
              temp_neighbors[querynum][s].first = dist[l*n + j];   
              temp_neighbors[querynum][s].second =  glob_ids[j];
              total_neighbors++; 
            }
          }
          std::sort(temp_neighbors[querynum].begin(), temp_neighbors[querynum].end());
          temp_neighbors[querynum].resize( std::min(temp_neighbors[querynum].size(), (size_t)k) );
        }
      }
    }
    
#pragma omp parallel for schedule(static)
    for( int i = 0; i < m; i++ ) (*neighbor_count)[i] = temp_neighbors[i].size();
    
    *neighbors = new pair<double, long>[total_neighbors];
    //Compute prefix sum of neighbor count for array indices
    omp_par::scan(*neighbor_count, neighbor_prefix, m);
#pragma omp parallel
    {
#pragma omp for schedule(static) 
      for( int i = 0; i < m; i++ ) {
        for( int j = 0; j < (*neighbor_count)[i]; j++ ) {
          (*neighbors)[neighbor_prefix[i]+j].first = temp_neighbors[i][j].first;
          (*neighbors)[neighbor_prefix[i]+j].second = temp_neighbors[i][j].second;
        }
      }
    }
    delete [] temp_neighbors;
    
  
  delete [] neighbor_prefix;   
  delete [] dist;
} 






void knn::compute_distances(double *ref, double *query, long n, long m, int dim, double *dist,
                            double* sqnormr, double* sqnormq, bool useSqnormrInput)
{
    double alpha = -2.0;
    double beta = 0.0;

    int iN = (int) n;
    int iM = (int) m;

    int maxt = omp_get_max_threads();
    bool omptest = (m > 4 * maxt || (m >= maxt && n > 128)) && n < 100000;

    double start = omp_get_wtime();
    #pragma omp parallel if( omptest )
    {
        int t = omp_get_thread_num();
        int numt = omp_get_num_threads();
        int npoints = getNumLocal(t, numt, m);

        int offset = 0;
        for(int i = 0; i < t; i++) offset += getNumLocal(i, numt, m);

        dgemm( "T", "N", &iN, &npoints, &dim, &alpha, ref, &dim, query + (dim*offset),
                &dim, &beta, dist + (offset*n), &iN );
    }
    dgemmtime += omp_get_wtime() - start;

    bool dealloc_sqnormr = false;
    bool dealloc_sqnormq = false;

    if(!sqnormr && !useSqnormrInput) {
        sqnormr = new double[n];
        dealloc_sqnormr = true;
    }
    if(!sqnormq) {
        sqnormq = new double[m];
        dealloc_sqnormq = true;
    }

    start = omp_get_wtime();
    if(!useSqnormrInput)  // Reuse if already computed for a previous block
        knn::sqnorm( ref, n, dim, sqnormr );
    knn::sqnorm( query, m, dim, sqnormq );
    sqnormtime += omp_get_wtime() - start;

    start = omp_get_wtime();
    if( m > maxt || n > 10000 ) {
        int blocksize = (n > 10000) ? m/maxt/2 : 128;
        #pragma omp parallel for //schedule(static) //schedule(dynamic,blocksize)
        for(int i = 0; i < m; i++ ) {
            int in = i*n;
            int j;
            #pragma ivdep
            for(j = 0; j < n; j ++ ) {
                int inpj = in + j;
                dist[inpj] += sqnormq[i] + sqnormr[j];
            }
        }
    } else {
        for(int i = 0; i < m; i++ ) {
            int in = i*n;
            int j;
            #pragma ivdep
            for(j = 0; j < n; j ++ ) {
                int inpj = in + j;
                dist[inpj] += sqnormq[i] + sqnormr[j];
            }
        }
    }
    addtime += omp_get_wtime() - start;

    if(dealloc_sqnormr)
        delete [] sqnormr;
    if(dealloc_sqnormq)
        delete [] sqnormq;

    #pragma omp parallel for
    for(int i = 0; i < m*n; i++) {
        if(dist[i] < 0.0) dist[i] = 0.0;
    }
}


std::pair<double, long> * knn::dist_directKQuery( double* &ref, double *query, 
						  long* &glob_ids,
						  int nlocal, int mlocal,
						  int k, 
						  int dim, 
						  MPI_Comm comm ) 
{
	
	int rank, size;
	
	double *ref_next; //receive buffer
	long *ids_next;
	
	pair<double, long> *kmin, *kmin_new; //k minimum 
	
	int recv_count, recvid_count, nlocal_next; //number of ref points received
	int sendrto, recvrfrom; //Communication partners
	
	//Request and status objects for sending and receiving stuff
	MPI_Request sendr, recvr, sendid, recvid;
	MPI_Status recvstat, recvidstat;
	
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);
	
	sendrto = (rank+1)%size;
	recvrfrom = (rank-1 >= 0 ) ? rank-1 : size-1;
	
	int max_recv; //= (int)std::ceil( (double)n / (double)size );
	MPI_Allreduce(&nlocal, &max_recv, 1, MPI_INT, MPI_MAX, comm);
	
	//Allocate receive buffer
	ref_next = (double*)malloc( max_recv*dim*sizeof(double));
	ids_next = (long*)malloc(max_recv*sizeof(long));
	//Ensure ref is big enough to use as a receive buffer.
	ref = (double *) realloc( (void *)ref, max_recv*dim*sizeof(double) );
	glob_ids = (long *) realloc( (void *)glob_ids, max_recv*sizeof(long) );

	bool kmin_set = false; //Don't merge unless kmin has been initialized.
	
	for( int i = 0; i < size; i++ ) {
		MPI_Isend( (void *) ref, nlocal*dim, MPI_DOUBLE, sendrto, TAG_R, comm, &sendr );
		MPI_Irecv( (void *) ref_next, max_recv*dim, MPI_DOUBLE, recvrfrom, TAG_R, comm, &recvr );
		MPI_Isend( (void *) glob_ids, nlocal, MPI_LONG, sendrto, TAG_ID, comm, &sendid );
		MPI_Irecv( (void *) ids_next, max_recv, MPI_LONG, recvrfrom, TAG_ID, comm, &recvid );
	
		if( kmin_set ) {
			kmin_new = new pair<double, long>[mlocal*k];
			knn::directKQueryLowMem( ref, query, nlocal, mlocal, k, dim, kmin_new );
			// Translate to global indices
			#pragma omp parallel for
			for( int j = 0; j < mlocal*k; j++ ) {
				if(kmin_new[j].second != -1) //make sure it's valid
					kmin_new[j].second = glob_ids[ kmin_new[j].second ];
			}
			
			std::pair<double, long> *temp = knn::kmin_merge( kmin, kmin_new, mlocal, k );
			delete [] kmin;
			delete [] kmin_new;
			kmin = temp;
		} else {
			kmin = new pair<double, long>[mlocal*k];
			knn::directKQueryLowMem( ref, query, nlocal, mlocal, k, dim, kmin );
			// Translate to global indices
			#pragma omp parallel for
			for( int j = 0; j < mlocal*k; j++ ) {
				if(kmin[j].second != -1) //make sure it's valid
					kmin[j].second = glob_ids[ kmin[j].second ];
			}

			kmin_set = true;
		}

		//Make sure send/receive have finished before swapping buffers 
		MPI_Wait( &sendr, MPI_STATUS_IGNORE );
		MPI_Wait( &sendid, MPI_STATUS_IGNORE);
		MPI_Wait( &recvr, &recvstat );
		MPI_Wait( &recvid, &recvidstat);
		
		MPI_Get_count( &recvstat, MPI_DOUBLE, &recv_count );
		MPI_Get_count( &recvidstat, MPI_LONG, &recvid_count );
		nlocal_next = recv_count/dim;
		assert( recvid_count == nlocal_next );
		
		double *tempref = ref;
		long *tempid = glob_ids;
		ref = ref_next;
		glob_ids = ids_next;
		ref_next = tempref;
		ids_next = tempid;
		nlocal = nlocal_next;
	}
	
	free(ref_next);
	free( ids_next);
	
	return kmin;
}





std::pair<double, long> *
   knn::kmin_merge( std::pair<double, long> *a, std::pair<double, long> *b, long m, long k ) {

   std::pair<double, long> *result = new std::pair<double, long>[m*k];   

   #pragma omp parallel for
   for( int i = 0; i < m; i++ ) {
     int aloc = i*k;
     int bloc = i*k;
     int resultloc = i*k;  

     for( int j = 0; j < k; j++ ) {
        if( a[aloc] <= b[bloc] ) {
           result[resultloc++] = a[aloc++];
        } else {
           result[resultloc++] = b[bloc++];
        }
     }

   }

   return result;
}



void knn::dist_directRQuery
              ( double* &ref, double *query, long n, long m, int nlocal, int mlocal, double R, 
                int dim, long* &glob_ids, vector< pair<double, long> >* rneighbors, 
                MPI_Comm comm  ) {

   int rank, size;

   double *ref_next; //receive buffers
   long *ids_next;   //

   int recv_count, recvid_count, nlocal_next; //number of ref points received

   int sendrto, recvrfrom; //Communication partners

   //Request and status objects for sending and receiving stuff
   MPI_Request sendr, recvr, sendid, recvid;
   MPI_Status recvstat, recvidstat;

   MPI_Comm_size(comm, &size);
   MPI_Comm_rank(comm, &rank);

   sendrto = (rank+1)%size;
   recvrfrom = (rank-1 >= 0 ) ? rank-1 : size-1;

   int max_recv;
   MPI_Allreduce(&nlocal, &max_recv, 1, MPI_INT, MPI_MAX, comm);

 
   //Allocate receive buffers
   ref_next = (double*) malloc(max_recv*dim*sizeof(double));
   ids_next = (long*) malloc(max_recv*dim*sizeof(long));

   //Ensure ref and glob_ids are big enough to use as receive buffers.
   ref = (double *) realloc( (void *)ref, max_recv*dim*sizeof(double) );
   glob_ids = (long *) realloc( (void *)glob_ids, max_recv*sizeof(long) );

   for( int i = 0; i < size; i++ ) {
      MPI_Isend( (void *) ref, nlocal*dim, MPI_DOUBLE, sendrto, TAG_R, comm, &sendr );
      MPI_Irecv( (void *) ref_next, max_recv*dim, MPI_DOUBLE, recvrfrom, TAG_R, comm, &recvr );
      MPI_Isend( (void *) glob_ids, nlocal, MPI_LONG, sendrto, TAG_ID, comm, &sendid );
      MPI_Irecv( (void *) ids_next, max_recv, MPI_LONG, recvrfrom, TAG_ID, comm, &recvid );

      int *neighbor_count;
      pair <double, long>* new_neighbors;

      directRQuery ( ref, query, nlocal, mlocal, R, dim, glob_ids, &neighbor_count, &new_neighbors );

      //Update current results with new neighbors found (if any).
      int *neighbor_scan = new int[mlocal+1];
      omp_par::scan(neighbor_count, neighbor_scan, mlocal);
      neighbor_scan[mlocal] = neighbor_scan[mlocal-1] + neighbor_count[mlocal-1];
      #pragma omp parallel for
      for( int j = 0; j < mlocal; j++ ) {
         rneighbors[j].reserve( rneighbors[j].size() + neighbor_count[j] );
         for( int l = neighbor_scan[j]; l < neighbor_scan[j+1]; l++ )  
            rneighbors[j].push_back(new_neighbors[l]); 
      }


      //Make sure send/receive have finished before swapping buffers 
      MPI_Wait( &sendr, MPI_STATUS_IGNORE );
      MPI_Wait( &sendid, MPI_STATUS_IGNORE );
      MPI_Wait( &recvr, &recvstat );
      MPI_Wait( &recvid, &recvidstat );

      MPI_Get_count( &recvstat, MPI_DOUBLE, &recv_count );
      MPI_Get_count( &recvidstat, MPI_LONG, &recvid_count );

      nlocal_next = recv_count/dim;
      assert( recvid_count == nlocal_next );     
 
      double *tempref = ref;
      long *tempid = glob_ids;
      ref = ref_next;
      glob_ids = ids_next;
      ref_next = tempref;
      ids_next = tempid;
      nlocal=nlocal_next;

      delete [] neighbor_count;
      delete [] neighbor_scan;
      delete [] new_neighbors; 
   }

   free(ref_next);
   free(ids_next);
}


void knn::dist_directRQueryIndividual
( double* &ref, double *query, long n, long m, int nlocal, int mlocal, double *R, 
 int dim, long* &glob_ids, vector< pair<double, long> >* rneighbors, 
 MPI_Comm comm  ) {
 

  int rank, size;
  
  double *ref_next; //receive buffers
  long *ids_next;   //
  
  int recv_count, recvid_count, nlocal_next; //number of ref points received
  
  int sendrto, recvrfrom; //Communication partners
  
  //Request and status objects for sending and receiving stuff
  MPI_Request sendr, recvr, sendid, recvid;
  MPI_Status recvstat, recvidstat;
  
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  
  sendrto = (rank+1)%size;
  recvrfrom = (rank-1 >= 0 ) ? rank-1 : size-1;
  
  int max_recv;
  MPI_Allreduce(&nlocal, &max_recv, 1, MPI_INT, MPI_MAX, comm);
  
  
  //Allocate receive buffers
  ref_next = (double*) malloc(max_recv*dim*sizeof(double));
  ids_next = (long*) malloc(max_recv*dim*sizeof(long));
  
  //Ensure ref and glob_ids are big enough to use as receive buffers.
  ref = (double *) realloc( (void *)ref, max_recv*dim*sizeof(double) );
  glob_ids = (long *) realloc( (void *)glob_ids, max_recv*sizeof(long) );
  
  for( int i = 0; i < size; i++ ) {
    MPI_Isend( (void *) ref, nlocal*dim, MPI_DOUBLE, sendrto, TAG_R, comm, &sendr );
    MPI_Irecv( (void *) ref_next, max_recv*dim, MPI_DOUBLE, recvrfrom, TAG_R, comm, &recvr );
    MPI_Isend( (void *) glob_ids, nlocal, MPI_LONG, sendrto, TAG_ID, comm, &sendid );
    MPI_Irecv( (void *) ids_next, max_recv, MPI_LONG, recvrfrom, TAG_ID, comm, &recvid );
    
    int *neighbor_count;
    pair <double, long>* new_neighbors;
    
    directRQueryIndividual ( ref, query, nlocal, mlocal, R, dim, glob_ids, &neighbor_count, &new_neighbors );
    
    //Update current results with new neighbors found (if any).
    int *neighbor_scan = new int[mlocal+1];
    omp_par::scan(neighbor_count, neighbor_scan, mlocal);
    neighbor_scan[mlocal] = neighbor_scan[mlocal-1] + neighbor_count[mlocal-1];
#pragma omp parallel for
    for( int j = 0; j < mlocal; j++ ) {
      rneighbors[j].reserve( rneighbors[j].size() + neighbor_count[j] );
      for( int l = neighbor_scan[j]; l < neighbor_scan[j+1]; l++ )  
        rneighbors[j].push_back(new_neighbors[l]); 
    }
    
    
    //Make sure send/receive have finished before swapping buffers 
    MPI_Wait( &sendr, MPI_STATUS_IGNORE );
    MPI_Wait( &sendid, MPI_STATUS_IGNORE );
    MPI_Wait( &recvr, &recvstat );
    MPI_Wait( &recvid, &recvidstat );
    
    MPI_Get_count( &recvstat, MPI_DOUBLE, &recv_count );
    MPI_Get_count( &recvidstat, MPI_LONG, &recvid_count );
    
    nlocal_next = recv_count/dim;
    assert( recvid_count == nlocal_next );     
    
    double *tempref = ref;
    long *tempid = glob_ids;
    ref = ref_next;
    glob_ids = ids_next;
    ref_next = tempref;
    ids_next = tempid;
    nlocal=nlocal_next;
    
    delete [] neighbor_count;
    delete [] neighbor_scan;
    delete [] new_neighbors; 
  }
  
  free(ref_next);
  free(ids_next);
}




void knn::dist_directRQueryIndividualK
( double* &ref, double *query, long n, long m, int nlocal, int mlocal, double *R, int k,
 int dim, long* &glob_ids, vector< pair<double, long> >* rneighbors, 
 MPI_Comm comm  ) {
 

  int rank, size;
  
  double *ref_next; //receive buffers
  long *ids_next;   //
  
  int recv_count, recvid_count, nlocal_next; //number of ref points received
  
  int sendrto, recvrfrom; //Communication partners
  
  //Request and status objects for sending and receiving stuff
  MPI_Request sendr, recvr, sendid, recvid;
  MPI_Status recvstat, recvidstat;
  
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  
  sendrto = (rank+1)%size;
  recvrfrom = (rank-1 >= 0 ) ? rank-1 : size-1;
  
  int max_recv;
  MPI_Allreduce(&nlocal, &max_recv, 1, MPI_INT, MPI_MAX, comm);
  
  
  //Allocate receive buffers
  ref_next = (double*) malloc(max_recv*dim*sizeof(double));
  ids_next = (long*) malloc(max_recv*dim*sizeof(long));
  
  //Ensure ref and glob_ids are big enough to use as receive buffers.
  ref = (double *) realloc( (void *)ref, max_recv*dim*sizeof(double) );
  glob_ids = (long *) realloc( (void *)glob_ids, max_recv*sizeof(long) );
  
  for( int i = 0; i < size; i++ ) {
    MPI_Isend( (void *) ref, nlocal*dim, MPI_DOUBLE, sendrto, TAG_R, comm, &sendr );
    MPI_Irecv( (void *) ref_next, max_recv*dim, MPI_DOUBLE, recvrfrom, TAG_R, comm, &recvr );
    MPI_Isend( (void *) glob_ids, nlocal, MPI_LONG, sendrto, TAG_ID, comm, &sendid );
    MPI_Irecv( (void *) ids_next, max_recv, MPI_LONG, recvrfrom, TAG_ID, comm, &recvid );
    
    int *neighbor_count;
    pair <double, long>* new_neighbors;
    
    directRQueryIndividual ( ref, query, nlocal, mlocal, R, dim, glob_ids, &neighbor_count, &new_neighbors );
    
    //Update current results with new neighbors found (if any).
    int *neighbor_scan = new int[mlocal+1];
    omp_par::scan(neighbor_count, neighbor_scan, mlocal);
    neighbor_scan[mlocal] = neighbor_scan[mlocal-1] + neighbor_count[mlocal-1];
#pragma omp parallel for
    for( int j = 0; j < mlocal; j++ ) {
      rneighbors[j].reserve( rneighbors[j].size() + neighbor_count[j] );
      for( int l = neighbor_scan[j]; l < neighbor_scan[j+1]; l++ )  
        rneighbors[j].push_back(new_neighbors[l]); 
      std::sort(rneighbors[j].begin(), rneighbors[j].end());
      rneighbors[j].resize( std::min(rneighbors[j].size(), (size_t) k) );
    }
    
    
    //Make sure send/receive have finished before swapping buffers 
    MPI_Wait( &sendr, MPI_STATUS_IGNORE );
    MPI_Wait( &sendid, MPI_STATUS_IGNORE );
    MPI_Wait( &recvr, &recvstat );
    MPI_Wait( &recvid, &recvidstat );
    
    MPI_Get_count( &recvstat, MPI_DOUBLE, &recv_count );
    MPI_Get_count( &recvidstat, MPI_LONG, &recvid_count );
    
    nlocal_next = recv_count/dim;
    assert( recvid_count == nlocal_next );     
    
    double *tempref = ref;
    long *tempid = glob_ids;
    ref = ref_next;
    glob_ids = ids_next;
    ref_next = tempref;
    ids_next = tempid;
    nlocal=nlocal_next;
    
    delete [] neighbor_count;
    delete [] neighbor_scan;
    delete [] new_neighbors; 
  }
  
  free(ref_next);
  free(ids_next);
}





//Only allows up to max_neighbors neigbors per query point
void knn::dist_directRQuery
( double* &ref, double *query, long n, long m, int nlocal, int mlocal, double R, 
 int dim, long* &glob_ids, vector< pair<double, long> >* rneighbors, int max_neighbors,
 MPI_Comm comm  ) {
  
  int rank, size;
  
  double *ref_next; //receive buffers
  long *ids_next;   //
  
  int recv_count, recvid_count, nlocal_next; //number of ref points received
  
  int sendrto, recvrfrom; //Communication partners
  
  //Request and status objects for sending and receiving stuff
  MPI_Request sendr, recvr, sendid, recvid;
  MPI_Status recvstat, recvidstat;
  
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  
  sendrto = (rank+1)%size;
  recvrfrom = (rank-1 >= 0 ) ? rank-1 : size-1;
  
  int max_recv;
  MPI_Allreduce(&nlocal, &max_recv, 1, MPI_INT, MPI_MAX, comm);
  
  
  //Allocate receive buffers
  ref_next = (double*) malloc(max_recv*dim*sizeof(double));
  ids_next = (long*) malloc(max_recv*dim*sizeof(long));
  
  //Ensure ref and glob_ids are big enough to use as receive buffers.
  ref = (double *) realloc( (void *)ref, max_recv*dim*sizeof(double) );
  glob_ids = (long *) realloc( (void *)glob_ids, max_recv*sizeof(long) );
  
  for( int i = 0; i < size; i++ ) {
    MPI_Isend( (void *) ref, nlocal*dim, MPI_DOUBLE, sendrto, TAG_R, comm, &sendr );
    MPI_Irecv( (void *) ref_next, max_recv*dim, MPI_DOUBLE, recvrfrom, TAG_R, comm, &recvr );
    MPI_Isend( (void *) glob_ids, nlocal, MPI_LONG, sendrto, TAG_ID, comm, &sendid );
    MPI_Irecv( (void *) ids_next, max_recv, MPI_LONG, recvrfrom, TAG_ID, comm, &recvid );
    
    int *neighbor_count;
    pair <double, long>* new_neighbors;
    
    directRQuery ( ref, query, nlocal, mlocal, R, dim, glob_ids, &neighbor_count, &new_neighbors );
    
    //Update current results with new neighbors found (if any).
    int *neighbor_scan = new int[mlocal+1];
    omp_par::scan(neighbor_count, neighbor_scan, mlocal);
    neighbor_scan[mlocal] = neighbor_scan[mlocal-1] + neighbor_count[mlocal-1];
#pragma omp parallel for
    for( int j = 0; j < mlocal; j++ ) {
      rneighbors[j].reserve( rneighbors[j].size() + neighbor_count[j] );
      for( int l = neighbor_scan[j]; l < neighbor_scan[j+1]; l++ )  
        rneighbors[j].push_back(new_neighbors[l]); 
    }
    
    
    //Make sure send/receive have finished before swapping buffers 
    MPI_Wait( &sendr, MPI_STATUS_IGNORE );
    MPI_Wait( &sendid, MPI_STATUS_IGNORE );
    MPI_Wait( &recvr, &recvstat );
    MPI_Wait( &recvid, &recvidstat );
    
    MPI_Get_count( &recvstat, MPI_DOUBLE, &recv_count );
    MPI_Get_count( &recvidstat, MPI_LONG, &recvid_count );
    
    nlocal_next = recv_count/dim;
    assert( recvid_count == nlocal_next );     
    
    double *tempref = ref;
    long *tempid = glob_ids;
    ref = ref_next;
    glob_ids = ids_next;
    ref_next = tempref;
    ids_next = tempid;
    nlocal=nlocal_next;
    
    delete [] neighbor_count;
    delete [] neighbor_scan;
    delete [] new_neighbors; 
  }
  
  free(ref_next);
  free(ids_next);
}


//Only allows up to max_neighbors neigbors per query point
void knn::dist_directRQueryIndividual
( double* &ref, double *query, long n, long m, int nlocal, int mlocal, double *R, 
 int dim, long* &glob_ids, vector< pair<double, long> >* rneighbors, int max_neighbors,
 MPI_Comm comm  ) {
  int rank, size;
  
  double *ref_next; //receive buffers
  long *ids_next;   //
  
  int recv_count, recvid_count, nlocal_next; //number of ref points received
  
  int sendrto, recvrfrom; //Communication partners
  
  //Request and status objects for sending and receiving stuff
  MPI_Request sendr, recvr, sendid, recvid;
  MPI_Status recvstat, recvidstat;
  
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  
  sendrto = (rank+1)%size;
  recvrfrom = (rank-1 >= 0 ) ? rank-1 : size-1;
  
  int max_recv;
  MPI_Allreduce(&nlocal, &max_recv, 1, MPI_INT, MPI_MAX, comm);
  
  
  //Allocate receive buffers
  ref_next = (double*) malloc(max_recv*dim*sizeof(double));
  ids_next = (long*) malloc(max_recv*dim*sizeof(long));
  
  //Ensure ref and glob_ids are big enough to use as receive buffers.
  ref = (double *) realloc( (void *)ref, max_recv*dim*sizeof(double) );
  glob_ids = (long *) realloc( (void *)glob_ids, max_recv*sizeof(long) );
  
  for( int i = 0; i < size; i++ ) {
    MPI_Isend( (void *) ref, nlocal*dim, MPI_DOUBLE, sendrto, TAG_R, comm, &sendr );
    MPI_Irecv( (void *) ref_next, max_recv*dim, MPI_DOUBLE, recvrfrom, TAG_R, comm, &recvr );
    MPI_Isend( (void *) glob_ids, nlocal, MPI_LONG, sendrto, TAG_ID, comm, &sendid );
    MPI_Irecv( (void *) ids_next, max_recv, MPI_LONG, recvrfrom, TAG_ID, comm, &recvid );
    
    int *neighbor_count;
    pair <double, long>* new_neighbors;
    
    directRQueryIndividual ( ref, query, nlocal, mlocal, R, dim, glob_ids, &neighbor_count, &new_neighbors );
    
    //Update current results with new neighbors found (if any).
    int *neighbor_scan = new int[mlocal+1];
    omp_par::scan(neighbor_count, neighbor_scan, mlocal);
    neighbor_scan[mlocal] = neighbor_scan[mlocal-1] + neighbor_count[mlocal-1];
#pragma omp parallel for
    for( int j = 0; j < mlocal; j++ ) {
      rneighbors[j].reserve( rneighbors[j].size() + neighbor_count[j] );
      for( int l = neighbor_scan[j]; l < neighbor_scan[j+1]; l++ )  
        rneighbors[j].push_back(new_neighbors[l]); 
    }
    
    
    //Make sure send/receive have finished before swapping buffers 
    MPI_Wait( &sendr, MPI_STATUS_IGNORE );
    MPI_Wait( &sendid, MPI_STATUS_IGNORE );
    MPI_Wait( &recvr, &recvstat );
    MPI_Wait( &recvid, &recvidstat );
    
    MPI_Get_count( &recvstat, MPI_DOUBLE, &recv_count );
    MPI_Get_count( &recvidstat, MPI_LONG, &recvid_count );
    
    nlocal_next = recv_count/dim;
    assert( recvid_count == nlocal_next );     
    
    double *tempref = ref;
    long *tempid = glob_ids;
    ref = ref_next;
    glob_ids = ids_next;
    ref_next = tempref;
    ids_next = tempid;
    nlocal=nlocal_next;
    
    delete [] neighbor_count;
    delete [] neighbor_scan;
    delete [] new_neighbors; 
  }
  
  free(ref_next);
  free(ids_next);
}




pair<int, int> knn::partitionPoints(std::string refFile, std::string queryFile, double *& ref, double *& query, long n, long m, int dim, int refParts, int queryParts, MPI_Comm comm)
{
	int rank;
	MPI_Comm_rank(comm, &rank);

	int *ranks = new int[refParts];
	for(int i = 0; i < refParts; i++)
	{
		ranks[i] = rank % queryParts + i * queryParts;
	}	
	MPI_Comm comm2;
	MPI_Group mainGroup;
	MPI_Comm_group(comm, &mainGroup);
	MPI_Group group;
	MPI_Group_incl(mainGroup, refParts, ranks, &group);
	delete[] ranks;
	MPI_Comm_create(comm, group, &comm2);
	//knn::parallelIO(refFile, n, dim, ref, comm2);
    //knn::mpi_binread(refFile.c_str(), n, dim, ref, comm2, false);
    int dummy_n;
    knn::mpi_binread(refFile.c_str(), n, dim, dummy_n, ref, comm2);

	ranks = new int[queryParts];
	for(int i = 0; i < queryParts; i++)
	{
		ranks[i] = rank / queryParts * queryParts + i;
	}
	MPI_Group_incl(mainGroup, queryParts, ranks, &group);
	delete[] ranks;
	MPI_Comm_create(comm, group, &comm2);
	//knn::parallelIO(queryFile, m, dim, query, comm2);
    //knn::mpi_binread(queryFile.c_str(), m, dim, query, comm2, false);
    int dummy_m;
    knn::mpi_binread(queryFile.c_str(), m, dim, dummy_m, query, comm2);

	return std::make_pair(dummy_n, dummy_m);
}

int knn::directRectRepartition(double *&points, int localPointCount, int dim, int *&globalIDs, int partitionCount, int direction, MPI_Comm comm)
{
	int rank;
	int size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	assert(size % partitionCount == 0);

	int replications = size / partitionCount;
	double *newPoints;
	int *newGlobalIDs;
	int newPointCount = 0;

	// Perform one entire replication per iteration
	for(int i = 0; i < replications; i++)
	{
		// Decide which process gets the data
		int root;
		if(direction == VERTICAL)
		{
			root = rank / replications * replications + i;
		}
		else if(direction == HORIZONTAL)
		{
			root = rank % replications + i * replications;
		}
		else
		{
			assert(false);
		}

		// Create a comm out of all processes sending to root
		int *ranks = new int[replications];
		if(direction == VERTICAL)
		{
			#pragma omp parallel for
			for(int j = 0; j < replications; j++)
			{	
				ranks[j] = rank / replications * replications + j;
			}
		}
		else
		{
			#pragma omp parallel for
			for(int j = 0; j < replications; j++)
			{
				ranks[j] = rank % partitionCount + j * partitionCount;
			}
		}

		MPI_Group mainGroup;
		MPI_Comm_group(comm, &mainGroup);
		MPI_Group tmpGroup;
		MPI_Comm tmpComm;
		MPI_Group_incl(mainGroup, replications, ranks, &tmpGroup);
		MPI_Comm_create(comm, tmpGroup, &tmpComm);
		delete[] ranks;

		int newRank;
		MPI_Comm_rank(tmpComm, &newRank);

		int *receiveCount;
		int *receiveIDCount;
		if(newRank == i)
		{
			receiveCount = new int[replications];
			receiveIDCount = new int[replications];
		}
		MPI_Gather(&localPointCount, 1, MPI_INT, receiveIDCount, 1, MPI_INT, i, tmpComm);
		if(newRank == i)
		{
			#pragma omp parallel for
			for(int j = 0; j < replications; j++)
			{
				receiveCount[j] = receiveIDCount[j] * dim;
			}
		}

		// Form displacement array and allocate enough space for receiving points
		int *displacement;
		int *IDdisplacement;
		if(newRank == i)
		{
			displacement = new int[replications];
			omp_par::scan(receiveCount, displacement, replications);

			IDdisplacement = new int[replications];
			#pragma omp parallel for
			for(int j = 0; j < replications; j++)
			{
				IDdisplacement[j] = displacement[j] / dim;
			}
		
			newPointCount = 0;
			#pragma omp parallel for reduction(+:newPointCount)
			for(int j = 0; j < replications; j++)
			{
				newPointCount += receiveCount[j];
			}	
			newPoints = new double[newPointCount];
			newPointCount /= dim;
			newGlobalIDs = new int[newPointCount];
		}

		// Gather the values
		MPI_Gatherv(points, localPointCount * dim, MPI_DOUBLE, newPoints, receiveCount, displacement, MPI_DOUBLE, i, tmpComm);
		MPI_Gatherv(globalIDs, localPointCount, MPI_INT, newGlobalIDs, receiveIDCount, IDdisplacement, MPI_INT, i, tmpComm);

		if(newRank == i)
		{
			delete[] receiveCount;
			delete[] receiveIDCount;
			delete[] displacement;
			delete[] IDdisplacement;
		}
		MPI_Comm_free(&tmpComm);
	}
	
	delete[] points;
	points = newPoints;
	delete[] globalIDs;
	globalIDs = newGlobalIDs;
	return newPointCount;
}


directQueryResults knn::directQueryRectK(double *refPoints, double *queryPoints, int localRefCount, int localQueryCount, int dim, int k, MPI_Comm queryComm)
{
	directQueryResults results;

	// Group distances with reference index and sort
	double *D = new double[localRefCount * localQueryCount];
	knn::compute_distances(refPoints, queryPoints, localRefCount, localQueryCount, dim, D);

	int size;
	int rank;
	MPI_Comm_rank(queryComm, &rank);
	MPI_Comm_size(queryComm, &size);
	int globalRefCount = 0;
	MPI_Allreduce(&localRefCount, &globalRefCount, 1, MPI_INT, MPI_SUM, queryComm);

	dist_t *inbuf = new dist_t[localRefCount * localQueryCount];
	long numPtsOffset = knn::getGlobalArrayOffset(rank, size, globalRefCount);
	#pragma omp parallel for
	for(int i = 0; i < localQueryCount; i++)
	{
		#pragma omp parallel for
		for(int j = 0; j < localRefCount; j++)
		{
			inbuf[i * localRefCount + j].first = D[i * localRefCount + j];
			inbuf[i * localRefCount + j].second = numPtsOffset + j;
		}
		omp_par::merge_sort(inbuf + i * localRefCount, inbuf + (i + 1) * localRefCount);
	}

	int globalQueryCount = 0;
	MPI_Allreduce(&localQueryCount, &globalQueryCount, 1, MPI_INT, MPI_SUM, queryComm);

	dist_t *rcvBufK = NULL;
	knn::query_k(queryComm, k, 0, inbuf, localQueryCount, localRefCount, rcvBufK);

	results.neighbors = rcvBufK;
	results.neighborCounts = new long[localQueryCount];
	#pragma omp parallel for
	for(int i = 0; i < localQueryCount; i++)
	{
		results.neighborCounts[i] = k;
	}

	delete[] D;
	delete[] inbuf;

	return results;
}

directQueryResults knn::directQueryRectR(double *refPoints, double *queryPoints, int localRefCount, int localQueryCount, int dim, double r, MPI_Comm queryComm)
{
	int size;
	int rank;
	MPI_Comm_size(queryComm, &size);
	MPI_Comm_rank(queryComm, &rank);
	
	directQueryResults results;

	// Calculate distances
	double *D = new double[localRefCount * localQueryCount];
	knn::compute_distances(refPoints, queryPoints, localRefCount, localQueryCount, dim, D);

	int globalRefCount = 0;
	MPI_Allreduce(&localRefCount, &globalRefCount, 1, MPI_INT, MPI_SUM, queryComm);

	// Group distances with reference index and sort
	dist_t *inbuf = new dist_t[localRefCount * localQueryCount];
	long numPtsOffset = knn::getGlobalArrayOffset(rank, size, globalRefCount);
	#pragma omp parallel for
	for(int i = 0; i < localQueryCount; i++)
	{
		#pragma omp parallel for
		for(int j = 0; j < localRefCount; j++)
		{
			inbuf[i * localRefCount + j].first = D[i * localRefCount + j];
			inbuf[i * localRefCount + j].second = numPtsOffset + j;
		}
		omp_par::merge_sort(inbuf + i * localRefCount, inbuf + (i + 1) * localRefCount);
	}

	// Perform the query
	dist_t *rcvBufR = NULL;
	long *rcvNumR = NULL;
	knn::query_r(queryComm, r * r, inbuf, localQueryCount, localRefCount, rcvBufR, rcvNumR);

	// Find out number of points
	long *receiveCounts = new long[localQueryCount];
	MPI_Allreduce(rcvNumR, receiveCounts, localQueryCount, MPI_LONG, MPI_SUM, queryComm);
	long totalPoints = 0;
	#pragma omp parallel for reduction(+:totalPoints)
	for(int i = 0; i < localQueryCount; i++)
	{
		totalPoints += receiveCounts[i];
	}

	// Distribute results
	long *indexResults = new long[totalPoints];
	double *distanceResults = new double[totalPoints];
	int offset = 0;
	int globalOffset = 0;
	for(int i = 0; i < localQueryCount; i++)
	{
		// Copy values to send
		long *tmpIndices = new long[rcvNumR[i]];
		double *tmpDistances = new double[rcvNumR[i]];
		for(int j = 0; j < rcvNumR[i]; j++)
		{
			tmpIndices[j] = rcvBufR[j + offset].second;
			tmpDistances[j] = rcvBufR[j + offset].first;
		}

		// Gather number of values each node is sending
		int *receive = new int[size];
		int *displ = new int[size];
		MPI_Gather(&rcvNumR[i], 1, MPI_INT, receive, 1, MPI_INT, 0, queryComm);
		omp_par::scan(receive, displ, size);

		MPI_Gatherv(tmpIndices, rcvNumR[i], MPI_LONG, indexResults + globalOffset, receive, displ, MPI_LONG, 0, queryComm);
		MPI_Gatherv(tmpDistances, rcvNumR[i], MPI_DOUBLE, distanceResults + globalOffset, receive, displ, MPI_DOUBLE, 0, queryComm);

		delete[] receive;
		delete[] displ;
		delete[] tmpIndices;
		delete[] tmpDistances;
		offset += rcvNumR[i];
		globalOffset += receiveCounts[i];
	}

	results.neighbors = new dist_t[totalPoints];
	#pragma omp parallel for
	for(int i = 0; i < totalPoints; i++)
	{
		results.neighbors[i].second = indexResults[i];
		results.neighbors[i].first = distanceResults[i];
	}

	results.neighborCounts = receiveCounts;

	delete[] D;
	delete[] inbuf;
	delete[] indexResults;
	delete[] distanceResults;

	return results;
}

directQueryResults knn::directRectRepartitionAndQuery(double *&refPoints, double *&queryPoints, long& localRefCount, long& localQueryCount, int dim, int *&refIDs, int *&queryIDs, directQueryParams params, MPI_Comm comm)
{	
	int rank;
	int size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	directQueryResults results;

	assert(size == params.queryParts * params.refParts);

	MPI_Group mainGroup;
	MPI_Comm_group(comm, &mainGroup);

	MPI_Group tmpGroup;

	// Construct comm of processes with same ref points
	MPI_Comm queryComm;
	int *ranks = new int[params.refParts];
	#pragma omp parallel for
	for(int i = 0; i < params.refParts; i++)
	{
		ranks[i] = rank % params.queryParts + i * params.queryParts;
	}
	MPI_Group_incl(mainGroup, params.refParts, ranks, &tmpGroup);
	MPI_Comm_create(comm, tmpGroup, &queryComm);
	delete[] ranks;

	// Construct comm of processes with same query points
	MPI_Comm refComm;
	ranks = new int[params.queryParts];
	#pragma omp parallel for
	for(int i = 0; i < params.queryParts; i++)
	{
		ranks[i] = rank / params.queryParts * params.queryParts + i;
	}
	MPI_Group_incl(mainGroup, params.queryParts, ranks, &tmpGroup);
	MPI_Comm_create(comm, tmpGroup, &refComm);

	// Do the partitioning
	localRefCount = knn::directRectRepartition(refPoints, localRefCount, dim, refIDs, params.refParts, VERTICAL, comm);
	localQueryCount = knn::directRectRepartition(queryPoints, localQueryCount, dim, queryIDs, params.queryParts, HORIZONTAL, comm);

	if(params.queryType == 'R')
	{
		results = knn::directQueryRectR(refPoints, queryPoints, localRefCount, localQueryCount, dim, params.r, queryComm);
	}
	else if(params.queryType == 'K')
	{
		results = knn::directQueryRectK(refPoints, queryPoints, localRefCount, localQueryCount, dim, params.k, queryComm);
	}

	if(rank >= params.queryParts)
	{
		#pragma omp parallel for
		for(int i = 0; i < localQueryCount; i++)
		{
			results.neighborCounts[i] = 0;
		}
	}

	MPI_Comm_free(&refComm);
	MPI_Comm_free(&queryComm);

	return results;
}

