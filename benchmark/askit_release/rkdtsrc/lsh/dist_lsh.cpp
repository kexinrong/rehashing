#include "lsh.h"
#include <direct_knn.h>
#include<knnreduce.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <climits>
#include <cfloat>
#include <vector>
#include <map>
#include <algorithm>
#include <direct_knn.h>
#include <ompUtils.h>
#include <blas.h>
#include <omp.h>
#include <mpi.h>
#include <repartition.h>
#include <util.h>
#include <eval.h>
#include <binQuery.h>
#include "stTreeSearch.h"

//Enable/disable minimal timing measurements and printing
#define TIMING 1
//Enable/disable detailed timing measurements and printing (requires additional syncronization)
#define TIMINGVERBOSE 1



using namespace std;
using namespace knn::lsh;
using namespace knn::repartition;



template<class T> inline void parcopy( T* a, T* b, size_t n ) {
  #pragma omp parallel for schedule(static)
  for(int i = 0; i < n; i++)
    b[i] = a[i];
}


/**
 * Computes the distance to the kth neighbor of a random sample of query points.
 * \param sampleSize The number of points to sample.
 * \param ref The reference point set.
 * \param refIDs The global IDs for the reference points.
 * \param query The query point set.
 * \param queryIDs The global IDs for the query points.
 * \param n The total number of reference points across all processes.
 * \param m The total number of query points across all processes.
 * \param nLocal The number of reference points in the local ref array.
 * \param mLocal The number of query points in the local query array.
 * \param dim The dimensionality of the points.
 * \param k The number of neighbors to find.
 * \param kthDist An array of sampleSize distances (allocated internally).
 * \param sampleIDs The IDs of the randomly sampled query point corresponding to each value in kthDist (allocated internally).
 * \param comm The MPI communicator.
 * \note This function is intended to be used internally only for computing the search radius for LSH queries.
 */
void sampleKthNearest( int sampleSize, double *ref, long *refIDs, double* query, long* queryIDs, long n, long m, 
                         int nLocal, int mLocal, 
                         int dim, int k, double **kthDist, long **sampleIDs, MPI_Comm comm) {
 
  int size, rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  //Generate new query point IDs contiguous on the interval [0,m)
  long *contiguousQueryIDs = new long[mLocal];  
  long lmlocal = mLocal;
  long firstID;
  MPI_Scan(&lmlocal, &firstID, 1, MPI_LONG, MPI_SUM, comm);
  firstID -= lmlocal; //need exclusive scan
  #pragma omp parallel for schedule(static)
  for(long i = 0; i < lmlocal; i++) contiguousQueryIDs[i] = firstID + i;

  double *directquery = new double[sampleSize*dim];

  //Sample log_2(mLocal) query points, and broadcast from rank 0.
  *sampleIDs = new long[sampleSize];
  for(int i = 0; i < sampleSize; i++) (*sampleIDs)[i] =  (long)((((double)rand())/(double)RAND_MAX) * (double)m);
  MPI_Bcast((*sampleIDs), sampleSize, MPI_LONG, 0, comm);
  
  //Search for selected query points in local query set.  Use linear search because of small number of samples.
  vector< pair<long, int> > found;
  for(int i = 0; i < sampleSize; i++) {
    #pragma omp parallel for schedule(static)
    for(int j = 0; j < mLocal; j++) {
      if( queryIDs[j] == (*sampleIDs)[i] ){
        #pragma omp critical
        {
          found.push_back( make_pair<long, int>(queryIDs[j], j) );
        }
      }
    }
  }

  int numFound = found.size();
  double *localSamples = new double[sampleSize*dim];
  for(int i = 0; i < numFound; i++) {
    for(int j = 0; j < dim; j++) {
      localSamples[i*dim+j] = query[found[i].second*dim+j];
    }
  }

  //Distribute sample query points to all processes.
  int sendcount = numFound*dim;
  int *rcvcounts = new int[size];
  int *rcvdisp = new int[size];
  MPI_Allgather(&sendcount, 1, MPI_INT, rcvcounts, 1, MPI_INT, comm);
  omp_par::scan(rcvcounts, rcvdisp, size);
  MPI_Allgatherv(localSamples, sendcount, MPI_DOUBLE, directquery, rcvcounts, rcvdisp, MPI_DOUBLE, comm); 

  //Perform a direct query on local reference points.
  pair<double, long> *localResult = new pair<double, long>[sampleSize*k];
  knn::directKQueryLowMem(ref, directquery, nLocal, sampleSize, k, dim, localResult);
  pair<double, long> *mergedResult;
  *kthDist = new double[sampleSize];
  knn::query_k(comm, k, 0, localResult, sampleSize, k, mergedResult); //Merge all results
     
  if(rank == 0) {
    for(int i = 0; i < sampleSize; i++) {
      (*kthDist)[i] = mergedResult[i*k+k-1].first;
    }
  }

  MPI_Bcast((*kthDist), sampleSize, MPI_DOUBLE, 0, comm);
  
  delete[] mergedResult;
  delete[] localResult;
  delete[] rcvcounts;
  delete[] rcvdisp;
  delete[] directquery;
  delete[] localSamples;
  delete[] contiguousQueryIDs;

}




/**
 * Calculates the number of discarded iterations that will occur for a given value of K; used for auto-tuner's bisection search.
 * \param localRef Local array of reference points.
 * \param nLocal Number of points in localRef.
 * \param nGlobal Sum of all processes' nLocal values.
 * \param localQuery Local array of query points.
 * \param mLocal Number of points in localQuery.
 * \param mGlobal Sum of all processes' mLocal values.
 * \param dim Dimensionality of points.
 * \param r Search radius.
 * \param K Number of random projection vectors to use for hashing.
 * \param L Number of times to hash points.
 * \param discardRatio Desired ratio of hash functions to be discarded (i.e., discardsWanted = discardRatio * L).
 * \param numBuckets The total number of hash buckets.
 * \param refKeys An array with length of at least nLocal * L for storing reference point hash keys (externally 
 * allocated for efficiency);
 * \param queryKeys An array with length of at least mLocal * L for storing query point hash keys (externally 
 * allocated for efficiency);
 * \param localKeyCount An array of length >= numBuckets for storing the number of local ref points with each
 * hash value (externally allocated).
 * \param globalKeyCount An array of length >= numBuckets for storing the number of global ref points with each
 * hash value (externally allocated).
 * \param localQueryKeyCount An array of length >= numBuckets for storing the number of local query points with each
 * hash value (externally allocated).
 * \param globalQueryKeyCount An array of length >= numBuckets for storing the number of global query points with each
 * hash value (externally allocated).
 * \param bucketWorkload An array of length >= numBuckets for storing the workload for each hash bucket (externally allocated).
 * \param comm The MPI communicator.
 * \param pointLimit The maximum number of reference points allowed in a single bucket; if 0 or unspecified, defaults to
 * 4*nGlobal/size.
 * \return A pair containing the (signed) difference between the number of discards encountered and the desired number, and the
 * number of discards encountered.
 */
pair<long,int> differenceFromTarget(double* localRef, int nLocal, long nGlobal, double *localQuery, 
                            int mLocal, long mGlobal, int dim, double r,
                            long K, long L, double discardRatio, int numBuckets, unsigned int *refKeys, unsigned int *queryKeys, 
                            int* localKeyCount, int *globalKeyCount, int *localQueryKeyCount, int *globalQueryKeyCount,
                            long *bucketWorkload, MPI_Comm comm, int pointLimit = 0 ) {

  int size, rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  //Generate and broadcast random vectors for projections
  double *rPP, *a, *b; 
  setup( dim, r, K, L, rPP, a, b );
  MPI_Bcast(rPP, K, MPI_DOUBLE, 0, comm);
  MPI_Bcast(a, K*L*dim, MPI_DOUBLE, 0, comm);
  MPI_Bcast(b, K*L, MPI_DOUBLE, 0, comm);


  int discards = 0;
  int discardTarget = discardRatio * (double)L;


  for(int tbl = 0; tbl < L; tbl++) {
    //Hash reference points, and compress keys to number of "outer table buckets"
    compute_hash_values( localRef, nLocal, a+tbl*K*dim, b+tbl*K, K, 1, dim,
                         rPP, refKeys );
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < nLocal*L; i++) {
       refKeys[i] = refKeys[i] % numBuckets;
    }
    #pragma omp parallel if(numBuckets > 2000)
    {
      #pragma omp for schedule(static)
      for(int i = 0; i < numBuckets; i++) localKeyCount[i] = 0;
    }
    //Calculate number of points with each compressed key
    for(int i = 0; i < nLocal; i++) localKeyCount[ refKeys[i] ]++;

    MPI_Reduce(localKeyCount, globalKeyCount, numBuckets, MPI_INT, MPI_SUM, 0, comm);


    if( localRef != localQuery ) {
      //Now, compute the query point hash keys.
      compute_hash_values( localQuery, mLocal, a+tbl*K*dim, b+tbl*K, K, 1, dim,
                         rPP, queryKeys );
      #pragma omp parallel for schedule(static)
      for(int i = 0; i < mLocal*L; i++) {
         queryKeys[i] = queryKeys[i] % numBuckets;
      }
      #pragma omp parallel if(numBuckets > 2000)
      {
        #pragma omp for schedule(static)
        for(int i = 0; i < numBuckets; i++)  localQueryKeyCount[i] = 0;
      }
      for(int i = 0; i < mLocal; i++) localQueryKeyCount[ queryKeys[i] ]++;

      MPI_Reduce(localQueryKeyCount, globalQueryKeyCount, numBuckets, MPI_INT, MPI_SUM, 0, comm);
    } else {
      globalQueryKeyCount = globalKeyCount;
    }


    if(rank == 0) {
      //Approximate the amount of work required to evaluate each bucket.
      long totalWork = 0;
      long workPerProc;
      long populated = 0;
      int excessSize = 0;
      long refLimit = pointLimit ? pointLimit : (nGlobal/(long)size)*4L;

      int maxBucketSize = 0; 
      for(int i = 0; i < numBuckets; i++)
        if(globalKeyCount[i] > maxBucketSize)
          maxBucketSize = globalKeyCount[i];
   
      #pragma omp parallel for schedule(static) reduction(+:totalWork,populated,excessSize)
      for(int i = 0; i < numBuckets; i++) {
        long qc = globalQueryKeyCount[i];
        long rc = globalKeyCount[i];
        bucketWorkload[i] = rc*qc;
        totalWork += bucketWorkload[i];
        if(qc || rc) populated++;
        excessSize += (int)((rc > refLimit)); //Can't safely handle buckets larger than this.
      }


      if( size > 1 ) {

        long largeBucketWork = 0;
        workPerProc = totalWork / size;
        #pragma omp parallel for reduction(+:largeBucketWork)
        for(int i = 0; i < numBuckets; i++) {
          if(bucketWorkload[i] > workPerProc) largeBucketWork += bucketWorkload[i];
        }
        if( (std::ceil((double)largeBucketWork / (double)workPerProc) >= size/2) || excessSize )
          discards ++;

      } else { //Single-process mode
        if( populated < numBuckets ) discards++;
      }


    }
  }


  long diffFromTarget = discards - discardTarget;

  delete[] a;
  delete[] b;
  delete[] rPP;


  MPI_Bcast(&diffFromTarget, 1, MPI_LONG, 0,comm);
  MPI_Bcast(&discards, 1, MPI_INT, 0,comm);
  return std::make_pair<long,int>(diffFromTarget, discards);

}





long autotuneK(long initialK, double *localRef, double *localQuery,
                    int nLocal, int mLocal, long n, long m, int dim, double r, int bf, MPI_Comm comm, int pointLimit = 0) {

  int size, rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  int numBuckets = bf*size;

  long minK = 1L;
  long midK;
  long maxMaxK = initialK*4L;
  long maxK = maxMaxK;
  long L = 20; //Number of trials for each K value

  double discardRatio = 0.25;

  unsigned int *refKeys = new unsigned int[nLocal*L];
  unsigned int *queryKeys = new unsigned int[mLocal*L];
  int* localKeyCount = new int[numBuckets];
  int* globalKeyCount = new int[numBuckets];
  int* localQueryKeyCount = new int[numBuckets];
  int* globalQueryKeyCount = new int[numBuckets];
  long *bucketWorkload = new long[numBuckets];


  pair<long,int> minKdiff;
  pair<long,int> midKdiff;
  pair<long,int> maxKdiff;
  bool maxMinUnset = true;
  int discardLimit = (double)discardRatio*(double)L;
  while( maxK - minK > 1 ) {
    midK = (maxK - minK)/2L + minK;
    
    if( maxMinUnset ) { //Only compute if we haven't already
      minKdiff = differenceFromTarget( localRef, nLocal, n, localQuery, mLocal, m, dim, r,
                            minK, L, discardRatio, numBuckets, refKeys, queryKeys,
                            localKeyCount, globalKeyCount, localQueryKeyCount, globalQueryKeyCount,
                            bucketWorkload, comm, pointLimit );

      maxKdiff = differenceFromTarget( localRef, nLocal, n,  localQuery, mLocal, m, dim, r,
                            maxK, L, discardRatio, numBuckets, refKeys, queryKeys,
                            localKeyCount, globalKeyCount, localQueryKeyCount, globalQueryKeyCount,
                            bucketWorkload, comm, pointLimit );

      assert( maxKdiff.second < discardLimit );
 
      maxMinUnset = false;
    }

    midKdiff = differenceFromTarget( localRef, nLocal, n, localQuery, mLocal, m, dim, r,
                            midK, L, discardRatio, numBuckets, refKeys, queryKeys,
                            localKeyCount, globalKeyCount, localQueryKeyCount, globalQueryKeyCount,
                            bucketWorkload, comm, pointLimit );


    if( (midKdiff.first < 0 && minKdiff.first > 0) || (midKdiff.first > 0 && minKdiff.first < 0) || midKdiff.first == 0 ) { //mid becomes new max
      maxK = midK;
      maxKdiff = midKdiff;
    } else { //mid becomes new min
      minK = midK;
      minKdiff = midKdiff;
    }

  }

  long newK; 
  if( minKdiff.second <= discardLimit ) 
    newK = minK;
  else if( midKdiff.second <= discardLimit )
    newK = midK;
  else if( maxKdiff.second <= discardLimit )
    newK = maxK;
  else {
    int currK = maxK;
    pair<long, int> currKdiff;
    do {
      currK++;
      currKdiff = differenceFromTarget( localRef, nLocal, n, localQuery, mLocal, m, dim, r,
                            currK, L, discardRatio, numBuckets, refKeys, queryKeys,
                            localKeyCount, globalKeyCount, localQueryKeyCount, globalQueryKeyCount,
                            bucketWorkload, comm, pointLimit );

      assert(currK < maxMaxK);
    } while (currKdiff.second > discardLimit );
    newK = currK;
  }

  MPI_Bcast(&newK, 1, MPI_LONG, 0,comm);

  delete[] refKeys;
  delete[] queryKeys;
  delete[] localKeyCount;
  delete[] globalKeyCount;
  delete[] localQueryKeyCount;
  delete[] globalQueryKeyCount;
  delete[] bucketWorkload;

  return newK;
}






void knn::lsh::rKSelect(double *ref, long *refIDs, double* query, long* queryIDs, long n, long m, int nLocal, int mLocal, 
                         int dim, int k, double mult, 
                         int bf, double &r, long &K, MPI_Comm comm, bool autotune, int pointLimit) {
 
  int size, rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  int logm = std::min((double)m/(double)size, 100.0*std::ceil( log10((double)m)/log10(2.0) ));

  double *globalKth;
  long *sampleIDs;

  sampleKthNearest( logm, ref, refIDs, query, queryIDs, n, m,
                         nLocal, mLocal,
                         dim, k, &globalKth, &sampleIDs, comm);

  delete[]  sampleIDs;

  for(int i = 0; i < logm; i++) {
    globalKth[i] = sqrt( abs(globalKth[i]) );
  }

  double sum = 0.0, mean, var, sigma;
  if( rank == 0) {
    //Compute the Mean, Variance, Std. Dev.
    #pragma omp parallel for reduction(+:sum) 
    for(int i = 0; i < logm; i++) {
      sum += globalKth[i];
    }
    mean = sum/(double)logm;

    sum = 0.0;
    #pragma omp parallel for reduction(+:sum) 
    for(int i = 0; i < logm; i++) {
      sum += (globalKth[i] - mean)*(globalKth[i] - mean);
    }

    var = sum/(double)logm;
    sigma = sqrt(var);
  }

  omp_par::merge_sort(globalKth, &(globalKth[logm]));
  double median = globalKth[logm/2];

  r = mean + mult*sigma;
  MPI_Bcast(&r, 1, MPI_DOUBLE, 0, comm);

  double dn = (double)n;
  double p1 = 0.8504;
  double p2 = 0.61;
  K = std::ceil( (std::log(dn) / std::log(1.0/p2)) );
  MPI_Bcast(&K, 1, MPI_LONG, 0, comm);

/*
  if(rank == 0) {
    cout << "=============================" << endl;
    cout << "r = " << r << endl;
    cout << "mean = " << mean << endl;
    cout << "median = " << median << endl;
    cout << "var = " << var << endl;
    cout << "sigma = " << sigma << endl;
    cout << "Initial K = " << K << endl;
    cout << "=============================" << endl;
  }
*/

  double tuningtime = omp_get_wtime();
  long tunedK = K;
  if(autotune)
    tunedK = autotuneK(K, ref, query, nLocal, mLocal, n, m, dim, r, bf, comm, pointLimit);
  tuningtime = omp_get_wtime() - tuningtime;

  delete [] globalKth;

/*
  if(rank == 0) {
    cout << "=============================" << endl;
    cout << "r = " << r << endl;
    cout << "mean = " << mean << endl;
    cout << "median = " << median << endl;
    cout << "var = " << var << endl;
    cout << "sigma = " << sigma << endl;
    cout << "Initial K = " << K << endl;
    cout << "Tuned K = " << tunedK << endl;
    cout << "K tuning time = " << tuningtime << endl;
    cout << "=============================" << endl;
  }
*/

  K = tunedK;

}



void balance(long *workload,
             int *globalRefKeyCount,
             int *globalQueryKeyCount,
             long workloadPerProc,
             int numBuckets,
             unsigned int *bucketMembership,
             int nLocal,
             long nGlobal,
             long mGlobal,
             int *process_id_per_point,
             int *process_id_per_bucket,
             int **send_count,
             MPI_Comm comm_in,
             MPI_Comm *comm_out)
{

  int size, rank;
  MPI_Comm_size(comm_in, &size);
  MPI_Comm_rank(comm_in, &rank);


  vector< pair<long,int> > smallBuckets;
  vector< pair<long,int> > largeBuckets;
  for(int i = 0; i < numBuckets; i++) {
    if( workload[i] > workloadPerProc )
      largeBuckets.push_back( std::make_pair(workload[i],i) );
    else
      smallBuckets.push_back( std::make_pair(workload[i],i) );
  }

  long largeBucketWork = 0L;
  int lbsize = largeBuckets.size();
  #pragma omp parallel if(lbsize > 1000)
  {
    #pragma omp for reduction(+:largeBucketWork)
    for(int i = 0; i < lbsize; i++) largeBucketWork += largeBuckets[i].first;
  }

  if( std::ceil((double)largeBucketWork / (double)workloadPerProc) > /*size/2*/ (double)size*0.5 ) {//Need too many processes for even balance
    if(rank == 0) cerr << "Warning: Too many points in large buckets.  Attempting to balance anwyay..." << endl;
    workloadPerProc = std::max((double)workloadPerProc,
          std::ceil( ((double)largeBucketWork/(double) /*(size/2)*/ (double)size*0.5)*1.5 )); //Adjust to fit large buckets within half of processes
  }


  if( largeBuckets.size() >= /*size/2*/(double)size*0.5 ) {  //Too many large buckets to allow splitting.  
    for(int i = 0; i < largeBuckets.size(); i++) smallBuckets.push_back( largeBuckets[i] );
    if(rank == 0) cerr << "Warning: Too many large buckets.  Attempting to balance anwyay..." << endl;
    largeBuckets.clear();
  }

  //Calculate how many processes each large bucket needs
  vector< pair<int,int> > largeBucketAssignments;  // <base process, number of processes>
  largeBucketAssignments.resize(largeBuckets.size());
  int currRank = size; //Last rank assigned
  for(int i = 0; i < largeBuckets.size(); i++) {
    int procs = round( (double)largeBuckets[i].first / (double)workloadPerProc );
    largeBucketAssignments[i].first = currRank - procs;
    largeBucketAssignments[i].second = procs;
    process_id_per_bucket[largeBuckets[i].second] = largeBucketAssignments[i].first;
    currRank = currRank - procs;
  }

  //Split the communicator appropriately
  vector<int> colors, keys;
  colors.resize(size);
  keys.resize(size);
  //Processes not assigned to large buckets will have a new comm with size=1
  for(int i = 0; i < currRank; i++) { 
    colors[i] = i;
    keys[i] = 0;
  }
  for(int i = 0; i < largeBucketAssignments.size(); i++){
     for(int j = 0; j < largeBucketAssignments[i].second; j++) { 
        colors[largeBucketAssignments[i].first+j] = currRank+i;
        keys[largeBucketAssignments[i].first+j] = j;
     }
  }
  MPI_Comm_split(comm_in, colors[rank], keys[rank], comm_out);



  //Priority queue for processes (min heap)
  std::priority_queue<pair<long,int>, std::vector<pair<long,int> >, std::greater<pair<long,int> > > process_queue;

  for( int i = 0; i < currRank; i++ ) process_queue.push( make_pair<long,int>(0L, i) );
  omp_par::merge_sort(smallBuckets.begin(), smallBuckets.end(), std::greater<pair<long,int> >());

  vector<int> refPointsPerProcess(currRank, 0);
  vector<int> queryPointsPerProcess(currRank, 0);

  int pointLimit = (nGlobal/(long)size)*4L;

  //Now assign small buckets to single-process communicators
  for(int i = 0; i < smallBuckets.size(); i++) {
    pair<long, int> emptiestProcess;
    emptiestProcess.first = process_queue.top().first;
    emptiestProcess.second = process_queue.top().second;
    process_id_per_bucket[smallBuckets[i].second] = emptiestProcess.second;

    //Update number of points assigned to this process
    refPointsPerProcess[ emptiestProcess.second ] +=  globalRefKeyCount[ smallBuckets[i].second ];
    queryPointsPerProcess[ emptiestProcess.second ] +=  globalQueryKeyCount[ smallBuckets[i].second ];

    //Update queue
    emptiestProcess.first += smallBuckets[i].first;
    process_queue.pop();

    //Only reinsert process into queue if it doesn't have too many points already.
    if( refPointsPerProcess[ emptiestProcess.second ] <= pointLimit )
      process_queue.push(emptiestProcess);
    else if( process_queue.empty() )
      process_queue.push(emptiestProcess);
  }

  *send_count = new int [size];
  #pragma omp parallel if(size > 1000)
  {
    #pragma omp for
    for(int i = 0; i < size; i++)
      (*send_count)[i] = 0;
  }

  #pragma omp parallel for
  for(int i = 0; i < nLocal; i++)
    process_id_per_point[i] = process_id_per_bucket[ bucketMembership[i] ];

  for(int i = 0; i < nLocal; i++)
    (*send_count)[ process_id_per_point[i] ]++;

}




void knn::lsh::distPartitionedKQuery
              ( double* ref, long *refIDs, double *query, long* queryIDs, long n, long m, int nLocal, 
                int mLocal, int dim, double range, int k, long K, long Lmax, int bucketFactor, double targetError,
                std::vector< pair<double,long> > &kNN, vector<long>& outQueryIDs, MPI_Comm comm, bool convAndTiming ) {

  int size, rank;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  int numBuckets = size*bucketFactor;
  int maxthreads = omp_get_max_threads();

  int iters = Lmax;
  Lmax = 1; //We will generate functions for only one table at a time.

  //performance timers and counters
  double refrepartition = 0.0;
  double refalltoall = 0.0;
  double refrehash = 0.0;
  double queryrepartition = 0.0;
  double queryalltoall = 0.0;
  double queryrehash = 0.0;
  double directeval = 0.0;
  double resultcollection = 0.0;
  double resultmerge = 0.0;
  double resultsort = 0.0;
  double loadbaltime = 0.0;
  double maxsingleeval = 0.0;
  double minsingleeval = 0.0;
  double querytime = 0.0;
  double groupingtime = 0.0;
  long flops;
  int maxbucket = 0; 
  int maxquery = 0;
  int minbucket = INT_MAX; 
  int minquery = INT_MAX;


  #if TIMING
  double overallstart = omp_get_wtime();
  #endif
  int nLocalOrig = nLocal; 
  int mLocalOrig = mLocal; 

  //Generate new query point IDs contiguous on the interval [0,m)
  long *contiguousQueryIDs = new long[mLocal];  
  long lmlocal = mLocal;
  long firstID;
  MPI_Scan(&lmlocal, &firstID, 1, MPI_LONG, MPI_SUM, comm);
  firstID -= lmlocal; //need exclusive scan
  #pragma omp parallel for schedule(static)
  for(long i = 0; i < lmlocal; i++) contiguousQueryIDs[i] = firstID + i;


  int ppn = m/size; //Number of query points per process
  int homepoints = (rank==size-1) ? ppn+m%ppn : ppn; //Number of query points "owned" by this process

  //Transmit original query IDs to home processes
  {
    pair<long, long> *idMap = new pair<long, long>[mLocal];
    pair<long, long> *homeidMap = new pair<long, long>[homepoints];
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < mLocal; i++) {
      idMap[i].first = contiguousQueryIDs[i];
      idMap[i].second = queryIDs[i];
    }
    omp_par::merge_sort(idMap, &(idMap[mLocal]));
    int *sendcounts = new int[size];
    int *recvcounts = new int[size];
    int *senddisp = new int[size];
    int *recvdisp = new int[size];

    MPI_Datatype pairdata;
    MPI_Type_contiguous(sizeof(pair<long, long>), MPI_BYTE, &pairdata);
    MPI_Type_commit(&pairdata);
  
    #pragma omp parallel if(size > 1000)
    {
      #pragma omp for 
      for(int i = 0; i < size; i++) sendcounts[i] = 0;
    }

    for(int i = 0; i < mLocal; i++) sendcounts[ idToHomeRank(idMap[i].first, ppn, size) ]++;
    MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, comm);
    omp_par::scan(sendcounts, senddisp, size);
    omp_par::scan(recvcounts, recvdisp, size);
    assert( recvdisp[size-1] + recvcounts[size-1] == homepoints );
    MPI_Alltoallv(idMap, sendcounts, senddisp, pairdata, homeidMap, recvcounts, recvdisp, pairdata, comm);

    omp_par::merge_sort(homeidMap, &(homeidMap[homepoints]));
    outQueryIDs.resize(homepoints);
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < homepoints; i++) outQueryIDs[i] = homeidMap[i].second;

    MPI_Type_free(&pairdata);
    delete[] sendcounts;
    delete[] recvcounts;
    delete[] senddisp;
    delete[] recvdisp;
    delete[] idMap;
    delete[] homeidMap;
  }

  //Fill result vector with padding for proper sorting/merging.
  kNN.resize(k*homepoints);
  #pragma omp parallel for schedule(static)
  for(int i = 0; i < k*homepoints; i++) {
    kNN[i].first = DBL_MAX;
    kNN[i].second = -1;
  }

  int* localKeyCount = new int[numBuckets];
  int* globalKeyCount = new int[numBuckets];
  int* localQueryKeyCount = new int[numBuckets];
  int* globalQueryKeyCount = new int[numBuckets];
  long *bucketWorkload = new long[numBuckets];
  int* bucketMapping = new int[numBuckets]; //Mapping of buckets to MPI ranks
  int *rcvcounts = new int[size];
  int *senddisp = new int[size];
  int *rcvdisp = new int[size];

  int qsize = mLocal/8; //Current size of query-size depdendent buffers
  int rsize = nLocal/8; //Current size of reference-size depdendent buffers
  int mLsize = (double)mLocal;  //Current size of mLocal-dependent buffers

  std::pair<double, long> *newNeighbors;
  double* sqnormr;
  double* sqnormq;

  newNeighbors = new std::pair<double,long>[qsize*k];
  sqnormr = new double[rsize];
  sqnormq = new double[KNN_MAX_BLOCK_SIZE];

  //Pad results lines to be multiples of cache line size (assume 64-bytes)
  int resultStride = ((int)(std::ceil( sizeof(triple<long, double, long>)*(double)k/64.0 ))*64) 
                          / sizeof(triple<long, double, long>) ;
  triple<long, double, long> *localResults = NULL; // = 
  assert( !posix_memalign((void**)&localResults, 4096, mLsize*resultStride*sizeof(triple<long, double, long>)) );
  assert(localResults);

  MPI_Datatype tripledata;
  MPI_Type_contiguous(sizeof(triple<long, double, long>), MPI_BYTE, &tripledata);
  MPI_Type_commit(&tripledata);


 
  //Sample O(log m) query points and compute exact kth-neighbor distance.
    //Compare with this value at each iteration until error converges to desired value.
  vector<long> sampleIDs;
  vector<double> globalKdist;
  vector<long> globalKid;
  double hit_rate, relative_error=DBL_MAX;
  int missingNeighbors = INT_MAX;

  if(convAndTiming)
     get_sample_info(ref, query, refIDs, contiguousQueryIDs, nLocal, mLocal, dim, k,
                   sampleIDs, globalKdist, globalKid);


  //Iterate over the "tables."  Although, we do not insert points into a hash table in the traditional sense.
  int curriter = 0;
  int badproj = 0;
  int tbl;
  while( curriter < iters && (relative_error != relative_error /*NaN check*/ || relative_error > targetError ||
                     missingNeighbors > 0) ) {

    //Generate and broadcast random vectors for projections
    double *rPP, *a, *b; 
    setup( dim, range, K, 1, rPP, a, b );
    MPI_Bcast(rPP, K, MPI_DOUBLE, 0, comm);
    MPI_Bcast(a, K*1*dim, MPI_DOUBLE, 0, comm);
    MPI_Bcast(b, K*1, MPI_DOUBLE, 0, comm);


    nLocal = nLocalOrig;
    mLocal = mLocalOrig;
    double *localRef = new double[nLocal*dim];
    double *localQuery = new double[mLocal*dim];
    long *localRefIDs = new long[nLocal];  
    long *localQueryIDs = new long[mLocal];  
 
    parcopy(ref, localRef, nLocal*dim); 
    parcopy(query, localQuery, mLocal*dim); 
    parcopy(refIDs, localRefIDs, nLocal); 
    parcopy(contiguousQueryIDs, localQueryIDs, mLocal); 


    unsigned int* refKeys = new unsigned int[nLocal];
    MPI_Comm subcomm;
    int subrank, subsize;
    #pragma omp parallel if( numBuckets > 1000 )
    {
      #pragma omp for schedule(static)
      for(int i = 0; i < numBuckets; i++)  localKeyCount[i] = 0;
    } 
       
    double start; 
    #if TIMINGVERBOSE
    MPI_Barrier(comm);
    start = omp_get_wtime();
    #endif
    //Hash reference points, and compress keys to number of "outer table buckets" 
    compute_hash_values( localRef, nLocal, a, b, K, 1, dim,
                         rPP, refKeys );
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < nLocal; i++) {
       refKeys[i] = refKeys[i] % numBuckets;
    }
    #if TIMINGVERBOSE
    MPI_Barrier(comm);
    refrehash += omp_get_wtime() - start;
    #endif

    #if TIMINGVERBOSE
    MPI_Barrier(comm);
    start = omp_get_wtime();
    #endif
    //Now, compute the query point hash keys.
    unsigned int* queryKeys = new unsigned int[mLocal];
    compute_hash_values( localQuery, mLocal, a, b, K, 1, dim,
                          rPP, queryKeys );
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < mLocal; i++) {
       queryKeys[i] = queryKeys[i] % numBuckets;
    }
    #if TIMINGVERBOSE
    MPI_Barrier(comm);
    queryrehash += omp_get_wtime() - start;
    #endif


    int *sendcounts;
    if( size > 1 ) {   // Can skip all of this for single-process execution
   
       //Calculate number of points with each compressed key
       for(int i = 0; i < nLocal; i++) localKeyCount[ refKeys[i] ]++; 
       MPI_Allreduce(localKeyCount, globalKeyCount, numBuckets, MPI_INT, MPI_SUM, comm);
  
       #pragma omp parallel for schedule(static)
       for(int i = 0; i < numBuckets; i++)  localQueryKeyCount[i] = 0;
       for(int i = 0; i < mLocal; i++) localQueryKeyCount[ queryKeys[i] ]++; 
       MPI_Allreduce(localQueryKeyCount, globalQueryKeyCount, numBuckets, MPI_INT, MPI_SUM, comm);

       //Approximate the amount of work required to evaluate each bucket.
       long totalWork = 0;
       long workPerProc;
       int excessSize = 0;
       #pragma omp parallel for schedule(static) reduction(+:totalWork,excessSize)
       for(int i = 0; i < numBuckets; i++) {
         long qc = globalQueryKeyCount[i];
         long rc = globalKeyCount[i];
         bucketWorkload[i] = rc*qc /*+ rc + qc*/;
         totalWork += bucketWorkload[i];
       } 
       workPerProc = totalWork / (long) size;


       //Check for excessive work in  large buckets.  If found, discard this set of projections.
       long largeBucketWork = 0L; 
       #pragma omp parallel for schedule(static) reduction(+:largeBucketWork)
       for(int i = 0; i < numBuckets; i++) {
         if(bucketWorkload[i] > workPerProc /* *0.75*/ ) largeBucketWork += bucketWorkload[i];
       } 
       if( ((largeBucketWork/workPerProc) >= (long)(size/2)) || excessSize ) { //Make sure we have enough projections left.
         if(rank == 0) { cerr << "Notice: Bad projection #" << ++badproj << " detected (Too many large buckets).  Skipping." << endl; cerr.flush(); }
         delete[] refKeys;
         delete[] queryKeys;
         delete[] rPP;
         delete[] a;
         delete[] b; 
         delete[] localRef;
         delete[] localQuery;
         delete[] localRefIDs;
         delete[] localQueryIDs;
         continue;
       }

       //Assign one or more bucket to each process.
       int* pointMapping = new int[nLocal];  //Mapping of reference points to MPI ranks
       #if TIMINGVERBOSE
       MPI_Barrier(comm);
       double start2 = omp_get_wtime();
       #endif
       balance( bucketWorkload, globalKeyCount, globalQueryKeyCount, workPerProc, 
                numBuckets, refKeys, nLocal, n, m,  pointMapping, 
                bucketMapping, &sendcounts, comm, &subcomm);


       #if TIMINGVERBOSE
       MPI_Barrier(comm);
       loadbaltime += omp_get_wtime() - start2;
       MPI_Barrier(comm);
       start = omp_get_wtime();
       #endif
       local_rearrange( &localRefIDs, &pointMapping, &refKeys, &localRef, nLocal, dim );
   
       
       //Transfer the reference points to their respective bucket's owner.
       double *newRef;
       long *newRefIDs;
       unsigned int *newRefKeys;
       long newnLocal;

       double commtime = omp_get_wtime();
       knn::repartition::repartition( localRefIDs, refKeys, localRef, nLocal, sendcounts, dim, 
                                      &newRefIDs, &newRefKeys, &newRef, &newnLocal, comm );
       refalltoall += omp_get_wtime() - commtime;

       delete[] localRef;
       delete[] localRefIDs;
       delete[] refKeys;
       delete[] pointMapping;
       localRef = newRef;
       localRefIDs = newRefIDs;
       nLocal = newnLocal;
       refKeys = newRefKeys;
   
       if(nLocal > 4.0 * n/size) 
         cerr << "Warning: Significant load imbalance detected.  Consider increasing parameter K." << endl;
   
       #if TIMINGVERBOSE
       refrepartition += omp_get_wtime() - start;
       #endif
   
       int *queryPointMapping = new int[mLocal]; //Rank where each query point will go
       #if TIMINGVERBOSE
       start = omp_get_wtime();
       #endif
       #pragma omp parallel for schedule(static)
       for(int i = 0; i < mLocal; i++)  queryPointMapping[i] = bucketMapping[ queryKeys[i] ];
   
       //Rearrange query points and transfer to appropriate processes
       local_rearrange( &localQueryIDs, &queryPointMapping, &queryKeys, &localQuery, mLocal, dim );
       #pragma omp parallel for schedule(static)
       for(int i = 0; i < size; i++) sendcounts[i] = 0;
       for(int i = 0; i < mLocal; i++) sendcounts[ queryPointMapping[i] ]++;
       double *newQuery;
       long *newQueryIDs;
       unsigned int *newQueryKeys;
       long newmLocal;
       commtime = omp_get_wtime();

       knn::repartition::repartition( localQueryIDs, queryKeys, localQuery, mLocal, 
                                      sendcounts, dim, &newQueryIDs, &newQueryKeys, &newQuery, &newmLocal, comm );
       queryalltoall += omp_get_wtime() - commtime;

       delete[] localQuery;
       delete[] localQueryIDs;
       delete[] queryKeys;
       delete[] queryPointMapping;
       localQuery = newQuery;
       localQueryIDs = newQueryIDs;
       mLocal = newmLocal;
       queryKeys = newQueryKeys;
      #if TIMING
       queryrepartition += omp_get_wtime() - start;
       #endif



      //Broadcast reference points to other processes in sub-communicator.
      MPI_Comm_rank(subcomm, &subrank);
      MPI_Comm_size(subcomm, &subsize);
      MPI_Bcast(&nLocal, 1, MPI_INT, 0, subcomm);
      if(subrank > 0) {
        delete[] localRef;
        delete[] localRefIDs;
        delete[] refKeys;
        localRef = new double[dim*nLocal];
        localRefIDs = new long[nLocal];
        refKeys = new unsigned int[nLocal];
      }
      MPI_Bcast(localRef, dim*nLocal, MPI_DOUBLE, 0, subcomm);
      MPI_Bcast(localRefIDs, nLocal, MPI_LONG, 0, subcomm);
      MPI_Bcast(refKeys, nLocal, MPI_UNSIGNED, 0, subcomm);
  
      //Scatter query points to other processes in sub-communicator.
      vector<int> scattercounts, scatterdispl;
      scattercounts.resize(subsize);
      scatterdispl.resize(subsize);
      int pointsPerProc = mLocal / subsize;
      for(int i = 0; i < subsize-1; i++) scattercounts[i] = pointsPerProc;
      scattercounts[subsize-1] = mLocal - (pointsPerProc * (subsize-1));
      omp_par::scan(&(scattercounts[0]), &(scatterdispl[0]), subsize);
      MPI_Scatter(&(scattercounts[0]), 1, MPI_INT, &newmLocal, 1, MPI_INT, 0, subcomm);
      newQuery = new double[dim*newmLocal];
      newQueryIDs = new long[newmLocal];
      newQueryKeys = new unsigned int[newmLocal];
  
      MPI_Scatterv(localQueryIDs, &(scattercounts[0]), &(scatterdispl[0]),
                   MPI_LONG, newQueryIDs, newmLocal, MPI_LONG, 0, subcomm);
      MPI_Scatterv(queryKeys, &(scattercounts[0]), &(scatterdispl[0]),
                   MPI_UNSIGNED, newQueryKeys, newmLocal, MPI_UNSIGNED, 0, subcomm);
      for(int i = 0; i < subsize; i++) {
        scattercounts[i] *= dim;
        scatterdispl[i] *= dim;
      }
      MPI_Scatterv(localQuery, &(scattercounts[0]), &(scatterdispl[0]),
                   MPI_DOUBLE, newQuery, newmLocal*dim, MPI_DOUBLE, 0, subcomm);
  
      delete[] localQuery;
      delete[] localQueryIDs;
      delete[] queryKeys;
      mLocal = newmLocal;
      localQuery = newQuery;
      localQueryIDs = newQueryIDs;
      queryKeys = newQueryKeys;
      MPI_Comm_free(&subcomm);

    } else {
       sendcounts = new int[size];
    }

    
    //rearrange point data/IDs to group equal keys contiguously.
    #if TIMINGVERBOSE
    MPI_Barrier(comm);
    double groupstart = omp_get_wtime();
    #endif
    local_rearrange( &localRefIDs, &refKeys, &localRef, nLocal, dim );
    local_rearrange( &localQueryIDs, &queryKeys, &localQuery, mLocal, dim );
    
    //Find location of each bucket in ref point array (so that we can perform lookups).
    vector<pair<unsigned int,int> > bucketOffsets;
    bucketOffsets.reserve(2*numBuckets/size);
    bucketOffsets.push_back(make_pair<unsigned int, int>(refKeys[0], 0));
    unsigned int currKey = refKeys[0];
    for(int i = 1; i < nLocal; i++) {
      if(refKeys[i] != currKey) {
        bucketOffsets.push_back(make_pair<unsigned int, int>(refKeys[i], i));
        currKey = refKeys[i];
      }
    }
    bucketOffsets.push_back(make_pair<unsigned int, int>(0xffffffff, nLocal)); //To find end of last bucket
    delete[] refKeys;


    vector<pair<unsigned int,int> > queryOffsets;
    queryOffsets.reserve(2*numBuckets/size);
    queryOffsets.push_back(make_pair<unsigned int, int>(queryKeys[0], 0));
    currKey = queryKeys[0];
    for(int i = 1; i < mLocal; i++) {
      if(queryKeys[i] != currKey) {
        queryOffsets.push_back(make_pair<unsigned int, int>(queryKeys[i], i));
        currKey = queryKeys[i];
      }
    }
    queryOffsets.push_back(make_pair<unsigned int, int>(0xffffffff, mLocal)); //To find end of last bucket
    delete[] queryKeys;
    #if TIMINGVERBOSE
    MPI_Barrier(comm);
    groupingtime += omp_get_wtime() - groupstart;
    #endif
    

    #if TIMING
    MPI_Barrier(comm);
    double querystart = omp_get_wtime();
    #endif


    maxquery = maxbucket = 0;
    for(int i = 0; i < queryOffsets.size() - 1; i++) {
       int numQ = queryOffsets[i+1].second - queryOffsets[i].second;
       if( numQ > maxquery ) maxquery = numQ;
    }
    int maxDistArraySize = 0;
    for(int i = 0; i < bucketOffsets.size() - 1; i++) {
      int numR = bucketOffsets[i+1].second - bucketOffsets[i].second;
      if( numR > maxbucket ) maxbucket = numR;
      if( getBlockSize(numR, maxquery)*numR > maxDistArraySize ) maxDistArraySize = getBlockSize(numR, maxquery)*numR;
    }

    int *refBucketIndex = new int[queryOffsets.size() - 1];
    int nQbuckets = queryOffsets.size() -1;
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < nQbuckets; i++) {
      unsigned int key = queryOffsets[i].first;
      //Find location of ref bucket with matching key, if it exists
      refBucketIndex[i] = -1;
      vector<pair<unsigned int,int> >::iterator it;
      pair<unsigned int, int> searchkey = std::make_pair(key, 0);
      it = std::lower_bound(bucketOffsets.begin(), bucketOffsets.end(), searchkey);
      if(it->first ==key) {
         refBucketIndex[i] = it - bucketOffsets.begin();
       }  
    }

    for(int i = 0; i < queryOffsets.size() - 1; i++) {
       int numQ = queryOffsets[i+1].second - queryOffsets[i].second;
       if( numQ > maxquery ) maxquery = numQ;
    }

    if( maxbucket > rsize ) {
       rsize = (double)maxbucket;
       delete[] sqnormr;
       sqnormr = new double[rsize];
    }

    if(maxquery > qsize){
       qsize = (double)maxquery;
       delete[] newNeighbors;
       newNeighbors = new std::pair<double,long>[qsize*k];
    }

    if( mLocal > mLsize ) {
       mLsize = (double)mLocal;
       free(localResults);
       assert( !posix_memalign((void**)&localResults, 4096, mLsize*resultStride*sizeof(triple<long, double, long>)) );
       assert(localResults);
    }

    double *dist = new double[ maxDistArraySize ];
   
    //Make sure that any padding goes to the end of the array when sorted.
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < mLocal*resultStride; i++) localResults[i].first = LONG_MAX;

    #if TIMING
    start = omp_get_wtime();
    #endif
    for(int i = 0; i < queryOffsets.size() - 1; i++) {
      int numQ = queryOffsets[i+1].second - queryOffsets[i].second;
      int numR;
      int qoffset = queryOffsets[i].second;
      int resultLoc = resultStride*qoffset;
      if( refBucketIndex[i] >= 0 ) { //Found matching bucket
        int roffset = bucketOffsets[refBucketIndex[i]].second;
        numR = bucketOffsets[refBucketIndex[i]+1].second - bucketOffsets[refBucketIndex[i]].second;
        directKQueryLowMem ( &(localRef[roffset*dim]), 
                              &(localQuery[qoffset*dim]), numR, numQ, k, dim, newNeighbors,
                              dist, sqnormr, sqnormq);
        #pragma omp parallel if(numQ > 1000)
        {
          #pragma omp for
          for(int j = 0; j < numQ; j++) {
            for(int l = 0; l < k; l++) {
              localResults[resultLoc+j*resultStride+l].first = localQueryIDs[qoffset+j];
              localResults[resultLoc+j*resultStride+l].second = newNeighbors[j*k+l].first;
              localResults[resultLoc+j*resultStride+l].third = 
                         (newNeighbors[j*resultStride+l].second < 0) ? newNeighbors[j*k+l].second : 
                                     localRefIDs[roffset + newNeighbors[j*k+l].second];
            }
          }
        }
      } else {  //Didn't find a matching bucket.  No neighbors.
        #pragma omp parallel if(numQ > 1000)
        {
          #pragma omp for schedule(static)
          for(int j = 0; j < numQ; j++) { //Fill with "padding" values
            for(int l = 0; l < k; l++) {
              localResults[resultLoc+j*resultStride+l].first = localQueryIDs[qoffset+j];
              localResults[resultLoc+j*resultStride+l].second = DBL_MAX;
              localResults[resultLoc+j*resultStride+l].third = -1;
            }
          }
        }
      }
    }

    #if TIMING
    directeval += omp_get_wtime() - start;
    #endif

    #if TIMINGVERBOSE
    MPI_Barrier(comm);
    start = omp_get_wtime();
    #endif
    double start1;
    //Finally, transfer results for each query point back to its "home" process,
     //and update existing results with newly found neighbors.
    int totalneighbors = k*mLocal;
    triple<long, double, long> *homeneighbors; 
    int rcvneighs; 

    if(size > 1) {
       #if TIMING
       start1 = omp_get_wtime();
       #endif
       omp_par::merge_sort(localResults, &(localResults[mLocal*resultStride]), triple<long, double, long>::firstLess);

       #if TIMING
       resultsort += omp_get_wtime() -start1;
       #endif
   
       #pragma omp parallel for schedule(static)
       for(int i = 0; i < size; i++) sendcounts[i] = 0;
       for(int i = 0; i < totalneighbors; i++) sendcounts[ idToHomeRank(localResults[i].first, ppn, size) ]++;
       MPI_Barrier(comm);
       MPI_Alltoall(sendcounts, 1, MPI_INT, rcvcounts, 1, MPI_INT, comm);
       omp_par::scan(sendcounts, senddisp, size);
       omp_par::scan(rcvcounts, rcvdisp, size);
       rcvneighs = rcvdisp[size-1]+rcvcounts[size-1];
       assert(rcvneighs == k*homepoints);
       homeneighbors = new triple<long, double, long>[rcvneighs];
       MPI_Barrier(comm);
       MPI_Alltoallv(localResults, sendcounts, senddisp, tripledata, homeneighbors,
                     rcvcounts, rcvdisp, tripledata, comm); 

       if(subrank > 0) {
         nLocal = 0; //Get rid of duplicate points before next iteration
       }
    } else {
       rcvneighs = k*m;
       homeneighbors = localResults;   
    }

    //Arrange neighbors by ascending query ID
    #if TIMINGVERBOSE
    start1 = omp_get_wtime();
    #endif
    omp_par::merge_sort(homeneighbors, &(homeneighbors[rcvneighs]));
    #if TIMINGVERBOSE
    resultsort += omp_get_wtime() -start1;
    #endif


    //Merge the new and old nearest neighbors
    #if TIMINGVERBOSE
    double mergetime = omp_get_wtime(); 
    #endif
    vector<pair<double, long> > pNeighbors;
    pNeighbors.resize(k*homepoints);
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < k*homepoints; i++) {
      pNeighbors[i].first = homeneighbors[i].second;
      pNeighbors[i].second = homeneighbors[i].third;
    }

    if(size > 1)
       delete[] homeneighbors;
    vector< pair<double,long> > tempNeighbors;
    //bintree::knn_merge( kNN, pNeighbors, homepoints, k, tempNeighbors);
    knn_merge( kNN, pNeighbors, homepoints, k, tempNeighbors);


    #pragma omp parallel for schedule(static)
    for(int i = 0; i < k*homepoints; i++) {
      kNN[i].first = tempNeighbors[i].first;
      kNN[i].second = tempNeighbors[i].second;
    }
    #if TIMINGVERBOSE
    resultmerge += omp_get_wtime() - mergetime;
    #endif
    
    #if TIMINGVERBOSE
    MPI_Barrier(comm);
    resultcollection += omp_get_wtime() - start;
    #endif

    //Check sample of current result against exact kth-neighbor distances.
    if(convAndTiming) {
       verify( sampleIDs, globalKdist, globalKid, outQueryIDs, kNN, comm, missingNeighbors, hit_rate, relative_error );
       if(rank == 0) cout << "iter= " << curriter << ", mean relative error= " << 
                             relative_error << ", hit rate= " << hit_rate << endl;
    }


    #if TIMING
    MPI_Barrier(comm);
    querytime += omp_get_wtime() - querystart;
    #endif

    delete[] dist;
    delete[] sendcounts;
    delete[] refBucketIndex;
    delete[] rPP;
    delete[] a;
    delete[] b;
    delete[] localRef;
    delete[] localQuery;
    delete[] localRefIDs;
    delete[] localQueryIDs;

    curriter++;
  }

  #if TIMING
  double overalltime = omp_get_wtime() - overallstart;
  #endif

  delete[] newNeighbors;
  delete[] sqnormr;
  delete[] sqnormq;
  free(localResults);

  delete[] senddisp;
  delete[] rcvdisp;
  delete[] rcvcounts;
  delete[] globalKeyCount;
  delete[] localKeyCount;
  delete[] localQueryKeyCount;
  delete[] globalQueryKeyCount;
  delete[] bucketWorkload;
  delete[] bucketMapping;
  delete[] contiguousQueryIDs;

  MPI_Type_free(&tripledata);

  #if TIMING
     double maxdirecteval, mindirecteval;
     MPI_Reduce(&directeval, &maxdirecteval, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
     MPI_Reduce(&directeval, &mindirecteval, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
  #endif


  #if TIMING
     if( rank == 0 && convAndTiming) {
       #if TIMINGVERBOSE
          cout << "Ref repartitioning: " << refrepartition << endl;
          cout << "Ref alltoall: " << refalltoall << endl;
          cout << "Load balance: " << loadbaltime << endl;
          cout << "Ref hashing: " << refrehash << endl;
          cout << "Query repartitioning: " << queryrepartition << endl;
          cout << "Query alltoall: " << queryalltoall << endl;
          cout << "Query hashing: " << queryrehash << endl;
          cout << "Bucket grouping: " << groupingtime << endl;
       #endif

       cout << "Total construction: " << overalltime - querytime << endl;
       cout << "Max direct eval: " << maxdirecteval << endl;
       cout << "Min direct eval: " << mindirecteval << endl;

       #if TIMINGVERBOSE
          cout << "Result collection and merging: " << resultcollection << endl;
          cout << "Result sort: " << resultsort << endl;
          cout << "Result merging: " << resultmerge << endl;
       #endif
       cout << "Total LSH query: " << querytime << endl;
     }
  #endif

}





// All-to-all nearest neighbor version
void knn::lsh::distPartitionedKQuery
              ( double* ref, long *refIDs, long n, int nLocal, 
                int dim, double range, int k, long K, long Lmax, int bucketFactor, double targetError,
                std::vector< pair<double,long> > &kNN, vector<long>& outQueryIDs, MPI_Comm comm, int pointLimit, 
                bool convAndTiming ) {

  int size, rank;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  int numBuckets = size*bucketFactor;
  int maxthreads = omp_get_max_threads();

  if(!pointLimit) pointLimit = (n/(long)size)*4;

  int iters = Lmax;
  Lmax = 1; //We will generate functions for only one table at a time.

  //performance timers and counters
  double refrepartition = 0.0;
  double refalltoall = 0.0;
  double refrehash = 0.0;
  double queryrepartition = 0.0;
  double queryalltoall = 0.0;
  double queryrehash = 0.0;
  double directeval = 0.0;
  double resultcollection = 0.0;
  double resultmerge = 0.0;
  double resultsort = 0.0;
  double loadbaltime = 0.0;
  double maxsingleeval = 0.0;
  double minsingleeval = 0.0;
  double querytime = 0.0;
  double groupingtime = 0.0;
  long flops;
  int maxbucket = 0; 
  int maxquery = 0;
  int minbucket = INT_MAX; 
  int minquery = INT_MAX;

  int mLocal = nLocal;
  double *query = ref;
  long *queryIDs = refIDs;
  long m = n;


  int nLocalOrig = nLocal; 
  int mLocalOrig = mLocal; 


  //Generate new query point IDs contiguous on the interval [0,m)
  long *contiguousQueryIDs = new long[mLocal];  
  long lmlocal = mLocal;
  long firstID;
  MPI_Scan(&lmlocal, &firstID, 1, MPI_LONG, MPI_SUM, comm);
  firstID -= lmlocal; //need exclusive scan
  #pragma omp parallel for schedule(static)
  for(long i = 0; i < lmlocal; i++) contiguousQueryIDs[i] = firstID + i;


  int ppn = m/size; //Number of query points per process
  int homepoints = (rank==size-1) ? ppn+m%ppn : ppn; //Number of query points "owned" by this process


  //We assume query IDs are contiguous on [0,m), so we don't need to transmit them.
  outQueryIDs.resize(homepoints);
  long firstHomeID = ppn*rank;
  #pragma omp parallel for schedule(static)
  for(int i = 0; i < homepoints; i++) outQueryIDs[i] = firstHomeID + i;





  //Fill result vector with padding for proper sorting/merging.
  kNN.resize(k*homepoints);
  #pragma omp parallel for schedule(static)
  for(int i = 0; i < k*homepoints; i++) {
    kNN[i].first = DBL_MAX;
    kNN[i].second = -1;
  }

  int* localKeyCount = new int[numBuckets];
  int* globalKeyCount = new int[numBuckets];
  int* localQueryKeyCount = new int[numBuckets];
  int* globalQueryKeyCount = new int[numBuckets];
  long *bucketWorkload = new long[numBuckets];
  int* bucketMapping = new int[numBuckets]; //Mapping of buckets to MPI ranks
  int *rcvcounts = new int[size];
  int *senddisp = new int[size];
  int *rcvdisp = new int[size];

  int qsize = mLocal/8; //Current size of query-size depdendent buffers
  int rsize = nLocal/8; //Current size of reference-size depdendent buffers
  int mLsize = (double)mLocal;  //Current size of mLocal-dependent buffers

  std::pair<double, long> *newNeighbors;
  double* sqnormr;
  double* sqnormq;

  newNeighbors = new std::pair<double,long>[qsize*k];
  sqnormr = new double[rsize];
  sqnormq = new double[KNN_MAX_BLOCK_SIZE];

  //Pad results lines to be multiples of cache line size (assume 64-bytes)
  int resultStride = ((int)(std::ceil( sizeof(triple<long, double, long>)*(double)k/64.0 ))*64) 
                          / sizeof(triple<long, double, long>) ;
  triple<long, double, long> *localResults = NULL; // = 
  assert( !posix_memalign((void**)&localResults, 4096, mLsize*resultStride*sizeof(triple<long, double, long>)) );
  assert(localResults);

  MPI_Datatype tripledata;
  MPI_Type_contiguous(sizeof(triple<long, double, long>), MPI_BYTE, &tripledata);
  MPI_Type_commit(&tripledata);


 
  //Sample O(log m) query points and compute exact kth-neighbor distance.
    //Compare with this value at each iteration until error converges to desired value.
  vector<long> sampleIDs;
  vector<double> globalKdist;
  vector<long> globalKid;
  double hit_rate, relative_error=DBL_MAX;
  int missingNeighbors = INT_MAX;

  if(convAndTiming)
    get_sample_info(ref, query, refIDs, contiguousQueryIDs, nLocal, mLocal, dim, k,
                   sampleIDs, globalKdist, globalKid);


  //Iterate over the "tables."  Although, we do not insert points into a hash table in the traditional sense.
  #if TIMING
  double overallstart = omp_get_wtime();
  #endif
  int curriter = 0;
  int badproj = 0;
  int tbl;
  while( curriter < iters && (relative_error != relative_error /*NaN check*/ || relative_error > targetError ||
                     missingNeighbors > 0) ) {

    //Generate and broadcast random vectors for projections
    double *rPP, *a, *b; 
    setup( dim, range, K, 1, rPP, a, b );
    MPI_Bcast(rPP, K, MPI_DOUBLE, 0, comm);
    MPI_Bcast(a, K*1*dim, MPI_DOUBLE, 0, comm);
    MPI_Bcast(b, K*1, MPI_DOUBLE, 0, comm);


    nLocal = nLocalOrig;
    mLocal = nLocalOrig;
    double *localRef = new double[nLocal*dim];
    double *localQuery; 
    long *localRefIDs = new long[nLocal];  
    long *localQueryIDs; 
 
    parcopy(ref, localRef, nLocal*dim); 
    parcopy(refIDs, localRefIDs, nLocal); 

    unsigned int *refKeys = new unsigned int[nLocal];
    unsigned int *queryKeys = refKeys;
    MPI_Comm subcomm;
    int subrank, subsize;
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < numBuckets; i++)  localKeyCount[i] = 0;
   
    
    if(!rank) cout << "hashing..." << endl;   
    double start; 
    #if TIMINGVERBOSE
    start = omp_get_wtime();
    #endif
    //Hash reference points, and compress keys to number of "outer table buckets" 
    compute_hash_values( localRef, nLocal, a, b, K, 1, dim,
                         rPP, refKeys );
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < nLocal; i++) {
       refKeys[i] = refKeys[i] % numBuckets;
    }
    #if TIMINGVERBOSE
    refrehash += omp_get_wtime() - start;
    #endif

    int *sendcounts;
    if( size > 1 ) {   // Can skip all of this for single-process execution
   
       //Calculate number of points with each compressed key
       for(int i = 0; i < nLocal; i++) localKeyCount[ refKeys[i] ]++; 
       MPI_Allreduce(localKeyCount, globalKeyCount, numBuckets, MPI_INT, MPI_SUM, comm);
 

       if(!rank) cout << "Calculating workload..." << endl;   
       //Approximate the amount of work required to evaluate each bucket.
       long totalWork = 0;
       long workPerProc;
       int refLimit = pointLimit;
       int excessSize = 0;
       #pragma omp parallel for schedule(static) reduction(+:totalWork,excessSize)
       for(int i = 0; i < numBuckets; i++) {
         long rc = globalKeyCount[i];
         bucketWorkload[i] = rc*rc;
         totalWork += bucketWorkload[i];
         excessSize += (globalKeyCount[i] > refLimit); //Can't safely handle buckets larger than this.
       } 
       workPerProc = totalWork / (long) size;


       //Check for excessive work in  large buckets.  If found, discard this set of projections.
       long largeBucketWork = 0L; 
       #pragma omp parallel for schedule(static) reduction(+:largeBucketWork)
       for(int i = 0; i < numBuckets; i++) {
         if(bucketWorkload[i] > workPerProc ) largeBucketWork += bucketWorkload[i];
       } 
       if( ((largeBucketWork/workPerProc) >= (long) (size/2)) || excessSize  ) { //Make sure we have enough projections left.
         if(rank == 0) { cerr << "Notice: Bad projection #" << ++badproj << " detected (Too many large buckets).  Skipping." << endl; cerr.flush(); }
         delete[] refKeys;
         delete[] rPP;
         delete[] a;
         delete[] b; 
         delete[] localRef;
         delete[] localRefIDs;
         continue;
       }

       if(!rank) cout << "Balancing..." << endl;   
       //Assign one or more bucket to each process.
       int* pointMapping = new int[nLocal];  //Mapping of reference points to MPI ranks
       #if TIMINGVERBOSE
       double start2 = omp_get_wtime();
       #endif
       balance( bucketWorkload, globalKeyCount, globalKeyCount,  workPerProc, numBuckets, 
                refKeys, nLocal, n, n, pointMapping, 
                bucketMapping, &sendcounts, comm, &subcomm);



       #if TIMINGVERBOSE
       loadbaltime += omp_get_wtime() - start2;
       MPI_Barrier(comm);
       start = omp_get_wtime();
       #endif
       local_rearrange( &localRefIDs, &pointMapping, &refKeys, &localRef, nLocal, dim );
   
       
       //Transfer the reference points to their respective bucket's owner.
       double *newRef;
       long *newRefIDs;
       unsigned int *newRefKeys;
       long newnLocal;
       double commtime = omp_get_wtime();


       MPI_Barrier(comm);
       if(!rank) cout << "repartitioning..." << endl;   
       refLimit = 0;
       knn::repartition::repartition( localRefIDs, localRef, nLocal, sendcounts, dim, 
                                      &newRefIDs, &newRef, &newnLocal, comm, refLimit);


       refalltoall += omp_get_wtime() - commtime;
       delete[] localRef;
       delete[] localRefIDs;
       delete[] refKeys;
       delete[] pointMapping;
       localRef = newRef;
       localRefIDs = newRefIDs;
       nLocal = newnLocal;

       if(!rank) cout << "rehashing..." << endl;   
       refKeys = new unsigned int[nLocal];   
       double start; 
       #if TIMINGVERBOSE
       start = omp_get_wtime();
       #endif
       //Hash reference points, and compress keys to number of "outer table buckets" 
       compute_hash_values( localRef, nLocal, a, b, K, 1, dim,
                         rPP, refKeys );
       #pragma omp parallel for schedule(static)
       for(int i = 0; i < nLocal; i++) {
         refKeys[i] = refKeys[i] % numBuckets;
       }
       #if TIMINGVERBOSE
       refrehash += omp_get_wtime() - start;
       refrepartition += omp_get_wtime() - start;
       #endif
   


      MPI_Comm_rank(subcomm, &subrank);
      MPI_Comm_size(subcomm, &subsize);

       if(!rank) cout << "local_rearrange on small-bucket processes..." << endl;   
      //rearrange point data/IDs to group equal keys contiguously.
      if(subsize == 1)
        local_rearrange( &localRefIDs, &refKeys, &localRef, nLocal, dim );


       if(!rank) cout << "Broadcasting points..." << endl;   
      //Broadcast reference points to other processes in sub-communicator.
      MPI_Barrier(subcomm);
      MPI_Bcast(&nLocal, 1, MPI_INT, 0, subcomm);
      if(subrank > 0) {
        delete[] localRef;
        delete[] localRefIDs;
        delete[] refKeys;
        localRef = new double[dim*nLocal];
        localRefIDs = new long[nLocal];
        refKeys = new unsigned int[nLocal];
      }
      MPI_Barrier(subcomm);
      MPI_Bcast(localRef, dim*nLocal, MPI_DOUBLE, 0, subcomm);
      MPI_Bcast(localRefIDs, nLocal, MPI_LONG, 0, subcomm);
      MPI_Bcast(refKeys, nLocal, MPI_UNSIGNED, 0, subcomm);
    
      int pointsPerProc = nLocal / subsize;
      localQuery = &(localRef[pointsPerProc*subrank*dim]);
      localQueryIDs = &(localRefIDs[pointsPerProc*subrank]);
      queryKeys = &(refKeys[pointsPerProc*subrank]);
      mLocal = (subrank < subsize-1) ? pointsPerProc : nLocal - (pointsPerProc*(subsize-1));

    } else {  // 1 process
      sendcounts = new int[size];
      local_rearrange( &localRefIDs, &refKeys, &localRef, nLocal, dim );
      localQuery = localRef;
      localQueryIDs = localRefIDs;
      queryKeys = refKeys;
      mLocal = nLocal;
    }


    if(!rank) cout << "Finding bucket locations..." << endl;   
    //Find location of each bucket in ref point array (so that we can perform lookups).
    vector<pair<unsigned int,int> > bucketOffsets;
    bucketOffsets.reserve(bucketFactor*10);
    bucketOffsets.push_back(make_pair<unsigned int, int>(refKeys[0], 0));
    unsigned int currKey = refKeys[0];
    for(int i = 1; i < nLocal; i++) {
      if(refKeys[i] != currKey) {
        bucketOffsets.push_back(make_pair<unsigned int, int>(refKeys[i], i));
        currKey = refKeys[i];
      }
    }
    bucketOffsets.push_back(make_pair<unsigned int, int>(0xffffffff, nLocal)); //To find end of last bucket


    vector<pair<unsigned int,int> > queryOffsets;
    queryOffsets.reserve(bucketFactor*10);
    queryOffsets.push_back(make_pair<unsigned int, int>(queryKeys[0], 0));
    currKey = queryKeys[0];
    for(int i = 1; i < mLocal; i++) {
      if(queryKeys[i] != currKey) {
        queryOffsets.push_back(make_pair<unsigned int, int>(queryKeys[i], i));
        currKey = queryKeys[i];
      }
    }
    queryOffsets.push_back(make_pair<unsigned int, int>(0xffffffff, mLocal)); //To find end of last bucket
    delete[] refKeys;
    

    #if TIMING
    double querystart = omp_get_wtime();
    #endif


    maxquery = maxbucket = 0;
    int qoffsizem1 = queryOffsets.size() - 1;
    for(int i = 0; i < qoffsizem1; i++) {
       int numQ = queryOffsets[i+1].second - queryOffsets[i].second;
       if( numQ > maxquery ) maxquery = numQ;
    }

    int maxDistArraySize = 0;
    for(int i = 0; i < bucketOffsets.size() - 1; i++) {
      int numR = bucketOffsets[i+1].second - bucketOffsets[i].second;
      if( numR > maxbucket ) maxbucket = numR;
      int dasize = getBlockSize(numR, maxquery)*numR;
      if( dasize > maxDistArraySize ) maxDistArraySize = dasize;
    }

    int *refBucketIndex = new int[queryOffsets.size() - 1];
    #pragma omp parallel if(qoffsizem1 > 1000)
    {
      #pragma omp for schedule(static)
      for(int i = 0; i < qoffsizem1; i++) {
        unsigned int key = queryOffsets[i].first;
        //Find location of ref bucket with matching key, if it exists
        refBucketIndex[i] = -1;
        vector<pair<unsigned int,int> >::iterator it;
        pair<unsigned int, int> searchkey = std::make_pair(key, 0);
        it = std::lower_bound(bucketOffsets.begin(), bucketOffsets.end(), searchkey);
        if(it->first ==key) {
           refBucketIndex[i] = it - bucketOffsets.begin();
         }  
      }
    }

    if( maxbucket > rsize ) {
       rsize = (double)maxbucket;
       delete[] sqnormr;
       sqnormr = new double[rsize];
    }

    if(maxquery > qsize){
       qsize = (double)maxquery;
       delete[] newNeighbors;
       newNeighbors = new std::pair<double,long>[qsize*k];
    }

    if( mLocal > mLsize ) {
       mLsize = (double)mLocal;
       free(localResults);
       assert( !posix_memalign((void**)&localResults, 4096, mLsize*resultStride*sizeof(triple<long, double, long>)) );
       assert(localResults);
    }

   
    //Make sure that any padding goes to the end of the array when sorted.
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < mLocal*resultStride; i++) localResults[i].first = LONG_MAX;

    double *dist = new double[ maxDistArraySize ];
    MPI_Barrier(comm);


    if(!rank) cout << "Beginning evaluation..." << endl;   
    #if TIMING
    start = omp_get_wtime();
    #endif
    for(int i = 0; i < queryOffsets.size() - 1; i++) {
      int numQ = queryOffsets[i+1].second - queryOffsets[i].second;
      int numR;
      int qoffset = queryOffsets[i].second;
      int resultLoc = resultStride*qoffset;
      if( refBucketIndex[i] >= 0 ) { //Found matching bucket
        int roffset = bucketOffsets[refBucketIndex[i]].second;
        numR = bucketOffsets[refBucketIndex[i]+1].second - bucketOffsets[refBucketIndex[i]].second;
        directKQueryLowMem ( &(localRef[roffset*dim]), 
                              &(localQuery[qoffset*dim]), numR, numQ, k, dim, newNeighbors,
                              dist, sqnormr, sqnormq);
        #pragma omp parallel if(numQ*k > 1000)
        {
          #pragma omp for 
          for(int j = 0; j < numQ; j++) {
            for(int l = 0; l < k; l++) {
              localResults[resultLoc+j*resultStride+l].first = localQueryIDs[qoffset+j];
              localResults[resultLoc+j*resultStride+l].second = newNeighbors[j*k+l].first;
              localResults[resultLoc+j*resultStride+l].third = 
                           (newNeighbors[j*k+l].second < 0) ? newNeighbors[j*k+l].second : 
                                       localRefIDs[roffset + newNeighbors[j*k+l].second];
            }
          }
        }
      } else {  //Didn't find a matching bucket.  No neighbors.
        #pragma omp parallel if(numQ*k > 1000)
        {
          #pragma omp parallel for 
          for(int j = 0; j < numQ; j++) { //Fill with "padding" values
            for(int l = 0; l < k; l++) {
              localResults[resultLoc+j*resultStride+l].first = localQueryIDs[qoffset+j];
              localResults[resultLoc+j*resultStride+l].second = DBL_MAX;
              localResults[resultLoc+j*resultStride+l].third = -1;
            }
          }
        }
      }
    }

    #if TIMING
    directeval += omp_get_wtime() - start;
    #endif
    delete[] dist;
    delete[] refBucketIndex;

    #if TIMINGVERBOSE
    MPI_Barrier(comm);
    start = omp_get_wtime();
    #endif
    double start1;
    //Finally, transfer results for each query point back to its "home" process,
     //and update existing results with newly found neighbors.
    int totalneighbors = k*mLocal;
    triple<long, double, long> *homeneighbors; 
    int rcvneighs; 

    if(size > 1) {
       if(!rank) cout << "Sorting results..." << endl;   
       omp_par::merge_sort(localResults, &(localResults[mLocal*resultStride]), triple<long, double, long>::firstLess);
   
       if(!rank) cout << "Transmitting results..." << endl;   
       #pragma omp parallel for schedule(static)
       for(int i = 0; i < size; i++) sendcounts[i] = 0;
       for(int i = 0; i < totalneighbors; i++) sendcounts[ idToHomeRank(localResults[i].first, ppn, size) ]++;
       MPI_Barrier(comm);
       MPI_Alltoall(sendcounts, 1, MPI_INT, rcvcounts, 1, MPI_INT, comm);
       omp_par::scan(sendcounts, senddisp, size);
       omp_par::scan(rcvcounts, rcvdisp, size);
       rcvneighs = rcvdisp[size-1]+rcvcounts[size-1];
       assert(rcvneighs == k*homepoints);
       homeneighbors = new triple<long, double, long>[rcvneighs];
       MPI_Barrier(comm);
       MPI_Alltoallv(localResults, sendcounts, senddisp, tripledata, homeneighbors,
                     rcvcounts, rcvdisp, tripledata, comm); 

       if(subrank > 0) {
         nLocal = 0; //Get rid of duplicate points before next iteration
       }
    } else {
       rcvneighs = k*m;
       homeneighbors = localResults;   
    }

    if(!rank) cout << "Sorting home results..." << endl;   
    //Arrange neighbors by ascending query ID
    omp_par::merge_sort(homeneighbors, &(homeneighbors[rcvneighs]));


    //Merge the new and old nearest neighbors
    vector<pair<double, long> > pNeighbors;
    pNeighbors.resize(k*homepoints);
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < k*homepoints; i++) {
      pNeighbors[i].first = homeneighbors[i].second;
      pNeighbors[i].second = homeneighbors[i].third;
    }

    if(size > 1)
       delete[] homeneighbors;
    vector< pair<double,long> > tempNeighbors;
    //bintree::knn_merge( kNN, pNeighbors, homepoints, k, tempNeighbors);
    knn_merge( kNN, pNeighbors, homepoints, k, tempNeighbors);


    #pragma omp parallel for schedule(static)
    for(int i = 0; i < k*homepoints; i++) {
      kNN[i].first = tempNeighbors[i].first;
      kNN[i].second = tempNeighbors[i].second;
    }
    
    #if TIMINGVERBOSE
    resultcollection += omp_get_wtime() - start;
    #endif

    if(!rank) cout << "Checking error..." << endl;   
    //Check sample of current result against exact kth-neighbor distances.
    if(convAndTiming) {
      verify( sampleIDs, globalKdist, globalKid, outQueryIDs, kNN, comm,  missingNeighbors, hit_rate, relative_error );
      if(rank == 0) cout << "iter= " << curriter << ", mean relative error= " << 
                            relative_error << ", hit rate= " << hit_rate << endl;
    }


    #if TIMING
    MPI_Barrier(comm);
    querytime += omp_get_wtime() - querystart;
    #endif

    delete[] sendcounts;
    delete[] rPP;
    delete[] a;
    delete[] b;
    delete[] localRef;
    delete[] localRefIDs;

    curriter++;
  }

  #if TIMING
  double overalltime = omp_get_wtime() - overallstart;
  #endif

  delete[] newNeighbors;
  delete[] sqnormr;
  delete[] sqnormq;
  free(localResults);

  delete[] senddisp;
  delete[] rcvdisp;
  delete[] rcvcounts;
  delete[] globalKeyCount;
  delete[] localKeyCount;
  delete[] localQueryKeyCount;
  delete[] globalQueryKeyCount;
  delete[] bucketWorkload;
  delete[] bucketMapping;
  delete[] contiguousQueryIDs;

  MPI_Type_free(&tripledata);

  #if TIMING
     double maxdirecteval, mindirecteval;
     MPI_Reduce(&directeval, &maxdirecteval, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
     MPI_Reduce(&directeval, &mindirecteval, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
  #endif


  #if TIMING
     if( rank == 0 && convAndTiming) {
       #if TIMINGVERBOSE
          cout << "Ref repartitioning: " << refrepartition << endl;
          cout << "Ref alltoall: " << refalltoall << endl;
          cout << "Load balance: " << loadbaltime << endl;
          cout << "Ref hashing: " << refrehash << endl;
       #endif

       cout << "Total construction: " << overalltime - querytime << endl;
       cout << "Max direct eval: " << maxdirecteval << endl;
       cout << "Min direct eval: " << mindirecteval << endl;

       #if TIMINGVERBOSE
          cout << "Result collection and merging: " << resultcollection << endl;
       #endif
       cout << "Total LSH query: " << querytime << endl;
     }
  #endif

}






void knn::lsh::distLSHreplicatedQ
              ( double* ref, long *refIDs, double *query, long* queryIDs, long n, long m, int nLocal, 
                int dim, int k, long Lmax, int bucketFactor, double rMultiplier, 
                std::vector< pair<double,long> > &kNN, vector<long>& outQueryIDs, MPI_Comm comm  ) {


	double r;
	long K;
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	//Select parameters
	double rtime = omp_get_wtime();
        knn::lsh::rKSelect( ref, refIDs, query, queryIDs, n, m, nLocal, m,
                        dim, k, rMultiplier, bucketFactor, r, K, MPI_COMM_SELF );
	rtime = omp_get_wtime() - rtime;



	//Run LSH query locally
	double querytime = omp_get_wtime();
        knn::lsh::distPartitionedKQuery( ref, refIDs, query, queryIDs, n, m, nLocal,
                  m, dim, r, k, K, Lmax, bucketFactor, 0.0, kNN, outQueryIDs, MPI_COMM_SELF, false);
	MPI_Barrier(comm);
	querytime = MPI_Wtime() - querytime;
	if(rank == 0) cout << "r selection time: " << rtime << endl;
        if (rank == 0) cout << "Query time: " << querytime << endl;

	//Perform reduction to collect results
	double mergetime = omp_get_wtime();
	pair<double, long> *merged;
	knn::query_k(comm, k, 0, &(kNN[0]), m, k, merged);
	MPI_Barrier(comm);
	#pragma omp parallel if(m > 1000)
	{
		int nResults = kNN.size();
		#pragma omp for
		for(int i = 0; i < nResults; i++) kNN[i] = merged[i];
	}

	delete[] merged;
	mergetime = omp_get_wtime() - mergetime; 
 


        if (rank == 0) cout << "Merge time: " << mergetime << endl;
      	if (rank == 0) cout << "Overall time: " << querytime + rtime + mergetime << endl;
}



