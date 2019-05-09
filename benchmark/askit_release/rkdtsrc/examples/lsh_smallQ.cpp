#include<knn.h>

#include<vector>
#include<cassert>
#include<cmath>
#include<string>
#include<utility>
#include <ctime>
#include<omp.h>
#include<mpi.h>

using namespace std;


/*
 * This example program demonstrates a nearest neighbor query with a small 
 * query point set in which both the reference and query point sets loaded from data files.
 * Here, the same data file is used for both point sets (to reduce the
 * size of sample data).  Although, only a portion of the file is loaded for the
 * query set.  In general, the query points needn't be a subset of the reference
 * point set.  The distLSHreplicatedQ function is a wrapper which assumes that
 * the query point set is exactly the same on all processes.  The wrapper
 * will automatically choose r and K using the auto-tuner and run the query
 * locally on each process, merging the results at the end.
 */

int main(int argc, char **argv) {
   MPI_Init(&argc, &argv);
   int rank, size;
   double *ref=NULL, *query=NULL; 
   long *refIds, *queryIds;


   int dim = 10;     // dimension
   long nRef, nQuery;  // number of reference points owned by this process
   int k = 10; //Find 10 nearest neighbors (Note that the first will always be the point itself).
   long nglobal=0L; //Number of reference points owned by all processes
   long mglobal=0L; //Number of query points owned by all processes
   long id_offset;  //The number of points owned by lower ranks.  Used to generate contiguous point IDs.
   double mult; //Multiplier for determining search radius
   int numBuckets; //Number of hash buckets on each MPI process (can differ from process to process).

   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   //Must be called prior to using LSH.
   srand(time(NULL));

   //Use sample file in sample_data directory
   string strInputFile = "./sample_data/sample-normal-100K10d.bin";
   nRef = 100000; //Total number of points in sample file, replaced by number read by this process at tend of parallelIO
   nQuery = 1000; //Only load a small portion of the file for query points. Replaced by number read by this process at tend of parallelIO
   dim = 10;
   if(rank == 0) cout << "User Input File: " << strInputFile << endl;

   //Load reference points from file
   knn::parallelIO(strInputFile, nRef, dim, ref, MPI_COMM_WORLD);
   assert(nRef >0);
   //Load query points from file
   knn::parallelIO(strInputFile, nQuery, dim, query, MPI_COMM_SELF); //Use COMM_SELF to have each process read same portion of data
   assert(nQuery >0);
   mglobal = nQuery; //Entire query point set stored locally by each process

   //Get total number of reference points
   MPI_Allreduce( &nRef, &nglobal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
   MPI_Scan(&nRef, &id_offset, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
   id_offset -= nRef;


   //Generate reference point IDs
   refIds = new long[nRef];
   for(int i = 0; i < nRef; i++) refIds[i] = id_offset + (long)i;
   //Generate query point IDs
   queryIds = new long[nQuery];
   for(int i = 0; i < nQuery; i++) queryIds[i] = (long)i; //No offset. Same IDs on each process


   int maxIters = 8; //Exact number of LSH iterations to perform.
   

   //Now, perform LSH search.  
   std::vector< pair<double,long> > lsh_results;  //Distance to neighbors, and neighbor point IDs
   std::vector<long> outIDs; //The IDs of the query points whose neighbors are owned by this process
   MPI_Barrier(MPI_COMM_WORLD);
   double start = MPI_Wtime();
   numBuckets = std::max(10L, nRef/100); //Average of 100 points per bucket, but a minimum of 10 buckets.
   mult = 1.0;  //1.0 is generally a good choice.  
 
   knn::lsh::distLSHreplicatedQ
              ( ref, refIds, query, queryIds, nglobal, mglobal, nRef,
                dim, k, maxIters, numBuckets, mult, 
                lsh_results, outIDs, MPI_COMM_WORLD );

 
   MPI_Barrier(MPI_COMM_WORLD);
   if (rank == 0) cout << endl << "Total LSH running time: " << MPI_Wtime() - start << endl;




   MPI_Finalize();
   delete[] ref;
   delete[] refIds;
   delete[] query;
   delete[] queryIds;

   return 0;
}

