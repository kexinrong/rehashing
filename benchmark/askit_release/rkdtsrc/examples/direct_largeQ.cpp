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
 * This example program demonstrates a direct nearest neighbor query 
 * with both the reference and query point sets loaded from data files.
 * Here, the query is performed using the cyclic distributed algorithm, 
 * which is generally appropriate when the reference point set and query
 * point set are roughly the same size or when repartitioning for a rectangular
 * query would be too costly.
 * Here, the same data file is used for both point sets (to reduce the
 * size of sample data).  However, the query point set could be different
 * in general.
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

   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   //Must be called prior to using LSH.
   srand(time(NULL));

   //Use sample file in sample_data directory
   string strInputFile = "./sample_data/sample-normal-100K10d.bin";
   nRef = 100000; //Total number of points in sample file, replaced by number read by this process at tend of parallelIO
   nQuery = 100000; //Total number of points in sample file, replaced by number read by this process at tend of parallelIO
   dim = 10;
   if(rank == 0) cout << "User Input File: " << strInputFile << endl;

   //Load reference points from file
   knn::parallelIO(strInputFile, nRef, dim, ref, MPI_COMM_WORLD);
   assert(nRef >0);
   //Load query points from file
   knn::parallelIO(strInputFile, nQuery, dim, query, MPI_COMM_WORLD);
   assert(nQuery >0);

   //Get total number of reference/query points
   MPI_Allreduce( &nRef, &nglobal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
   MPI_Scan(&nRef, &id_offset, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce( &nQuery, &mglobal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
   assert(nRef == nQuery && nglobal == mglobal); //Not necessarily true in general, but should hold for this example
   id_offset -= nRef;

   //Allocate new reference and query point arrays with malloc (needed because of realloc in dist_directKQuery).
   double *refCopy = (double*) malloc( nRef * dim * sizeof(double) );
   double *queryCopy = (double*) malloc( nQuery * dim * sizeof(double) );
   memcpy( refCopy, ref, nRef * dim * sizeof(double) );
   memcpy( queryCopy, query, nQuery * dim * sizeof(double) );



   //Generate reference point IDs
   refIds = (long*)malloc( nRef * sizeof(long) );
   for(int i = 0; i < nRef; i++) refIds[i] = id_offset + (long)i;
   //Generate query point IDs
   queryIds = (long*)malloc( nQuery * sizeof(long) );
   for(int i = 0; i < nQuery; i++) queryIds[i] = id_offset + (long)i;


   //Now, perform direct search using "cyclic" algorithm
   MPI_Barrier(MPI_COMM_WORLD);
   double start = MPI_Wtime();
   
   std::pair<double, long> *result = knn::dist_directKQuery( refCopy, queryCopy, refIds, nRef, nQuery, k, dim, MPI_COMM_WORLD );


   MPI_Barrier(MPI_COMM_WORLD);
   if (rank == 0) cout << endl << "Total query running time: " << MPI_Wtime() - start << endl;




   MPI_Finalize();
   delete[] ref;
   delete[] query;
   free(refIds);
   free(queryIds);
   free(refCopy);
   free(queryCopy);

   return 0;
}

