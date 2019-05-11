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
 * This example program demonstrates an all-nearest neighbor query 
 * (reference point set is also the query point set) on a data set 
 * loaded from a binary file.  
 */

int main(int argc, char **argv) {
   MPI_Init(&argc, &argv);
   int rank, size;
   double *ref=NULL; 
   long *ids;




   int dim = 10;     // dimension
   long nPts;  // number of reference points owned by this process
   long K;	// # of hash functions, determined by rKSelect
   int k = 10; //Find 10 nearest neighbors (Note that the first will always be the point itself).
   long nglobal=0L; //Number of points owned by all processes
   long id_offset;  //The number of points owned by lower ranks.  Used to generate contiguous point IDs.
   int bf;   //Bucket factor
   double mult; //Multiplier for determining search radius (used by rKSelect)

   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   //Must be called prior to using LSH.
   srand(time(NULL));

   //Use sample file in sample_data directory
   string strInputFile = "./sample_data/sample-normal-100K10d.bin";
   nPts = 100000; //Total number of points in sample file, replaced by number read by this process at tend of parallelIO
   dim = 10;


   if(rank == 0) cout << "User Input File: " << strInputFile << endl;
   knn::parallelIO(strInputFile, nPts, dim, ref, MPI_COMM_WORLD);
   assert(nPts >0);

   MPI_Allreduce( &nPts, &nglobal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
   MPI_Scan(&nPts, &id_offset, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
   id_offset -= nPts;

   //Generate point IDs
   ids = new long[nPts];
   for(int i = 0; i < nPts; i++) ids[i] = id_offset + (long)i;


   //Choose bucket factor such that each bucket will contain 1000 points on average.
   bf = nglobal / 1000 / size;
   bf = std::max(bf, 5); //Require at least 5 buckets per process.

   int maxIters = 20; //Maximum LSH iterations allowed.
   double errorTarget = 1.0; //Allow maximum of 1.0% mean relative distance error (if possible within maxIters).
   

   //Choose range for search, and auto-tune parameter K.
   double rtime = omp_get_wtime();
   double r;
   mult = 1.0; //This value should be good in almost all cases.
   knn::lsh::rKSelect(ref, ids, ref, ids, nglobal, nglobal, nPts, nPts,
                         dim, k, mult, bf, r, K, MPI_COMM_WORLD);
   if(rank == 0 ) cout << "K = " << K <<endl;
   if(rank == 0) cout << "Parameter selection time: " << omp_get_wtime() - rtime << endl;
   if(rank == 0) cout << "r: " << r << endl;

    
   //Now, perform distributed LSH All-nearest search.  
   std::vector< pair<double,long> > lsh_results;  //Distance to neighbors, and neighbor point IDs
   std::vector<long> outIDs; //The IDs of the query points whose neighbors are owned by this process
   MPI_Barrier(MPI_COMM_WORLD);
   double start = MPI_Wtime();
   
   knn::lsh::distPartitionedKQuery( ref, ids, nglobal, nPts,
               dim, r, k, K, maxIters, bf, errorTarget, lsh_results, outIDs, MPI_COMM_WORLD  );
  
   MPI_Barrier(MPI_COMM_WORLD);
   if (rank == 0) cout << endl << "Total LSH running time: " << MPI_Wtime() - start << endl;




   MPI_Finalize();
   delete[] ref;
   delete[] ids;

   return 0;
}

