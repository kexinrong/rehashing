#include<direct_knn.h>
#include<knnreduce.h>
#include<lsh.h>
#include<generator.h>
#include<vector>
#include<cassert>
#include<cmath>
#include<string>
#include<utility>
#include<omp.h>
#include<mpi.h>
#include<CmdLine.h>
#include <ompUtils.h>
#include <ctime>
#include<parallelIO.h>
#include <papi_perf.h>
#include <eval.h>

using namespace Torch;
using namespace std;

void getArgs(int argc, char **argv);                    // get command-line arguments

// Command line arguments
int dim;                                // dimension
long nPts;                              // number of reference points
long K=0L;					// # of hash functions
long L=0L;					// # of hash tables
int k;
int bf;
int check;
double errorTarget;
double mult;
long num_threads;
int trials;
int autotune;
char *ptrInputFile = NULL;
string strInputFile;

int main(int argc, char **argv) {
   MPI_Init(&argc, &argv);
   int rank, size;
   double *ref=NULL;
   long *ids;
   long nglobal=0L;
   long id_offset;

   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   getArgs(argc, argv);

   init_papi();
   srand(time(NULL));

	if(rank == 0) cout << "User Input File: " << strInputFile << endl;
    int dummy_nref;
	knn::mpi_binread( strInputFile.c_str(), nPts, dim,
                        dummy_nref, ref, MPI_COMM_WORLD);
    nPts = dummy_nref;

    MPI_Allreduce( &nPts, &nglobal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Scan(&nPts, &id_offset, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    id_offset -= nPts;

    ids = new long[nPts];
    for(int i = 0; i < nPts; i++) ids[i] = id_offset + (long)i;

    if(rank == 0) cout<<"nglobal = "<<nglobal<<endl;

   //Copy reference points to query array for all-to-all query
   double checktime = omp_get_wtime();
   vector<long> sampleIDs;
   vector<double> globalKdist;
   vector<long> globalKid;
   double hit_rate, relative_error=DBL_MAX;
   int missingNeighbors = INT_MAX;
   int logm = std::ceil( 100.0*log10((double)nglobal)/log10(2.0) );
   if(check) {
     get_sample_info(ref, ref, ids, ids, nPts, nPts, dim, k,
                   sampleIDs, globalKdist, globalKid);
   }
   checktime = omp_get_wtime() - checktime;

   for(int t = 0; t < trials; t++) {
      std::vector< pair<double,long> > lsh_results;
      std::vector<long> outIDs;


   //Choose range for search.
   double rtime = omp_get_wtime();
   double r;

   papi_mpi_flop_start();
   if(autotune) {
         knn::lsh::rKSelect(ref, ids, ref, ids, nglobal, nglobal, nPts, nPts,
                         dim, k, mult, bf, r, K, MPI_COMM_WORLD);
   } else {
     if( K == 0 ) {
         knn::lsh::rKSelect(ref, ids, ref, ids, nglobal, nglobal, nPts, nPts,
                         dim, k, mult, bf, r, K, MPI_COMM_WORLD, false);
     } else {
          long temp = K;
         knn::lsh::rKSelect(ref, ids, ref, ids, nglobal, nglobal, nPts, nPts,
                         dim, k, mult, bf, r, K, MPI_COMM_WORLD, false);
          K = temp;
     }
   }

   if(rank == 0 ) cout << "K = " << K << ", Lmax = " << L <<endl;
   if(rank == 0) cout << "r selection time: " << omp_get_wtime() - rtime << endl;
   if(rank == 0) cout << "r: " << r << endl;


   //Now, perform distributed LSH search.
      MPI_Barrier(MPI_COMM_WORLD);
      double start = MPI_Wtime();
      knn::lsh::distPartitionedKQuery( ref, ids, nglobal, nPts,
                   dim, r, k, K, L, bf, errorTarget, lsh_results, outIDs, MPI_COMM_WORLD  );

      float query_mflops =  papi_mpi_flop_stop(); 
      MPI_Barrier(MPI_COMM_WORLD);
      if (rank == 0) cout << "Query time: " << MPI_Wtime() - start << endl;
      if (rank == 0 && query_mflops > 0.0) cout << "Query TFLOPS: " << query_mflops /1.0e6 << endl;

     if(check) {
        //Check error for random sample of query points
        verify( sampleIDs, globalKdist, globalKid, outIDs, lsh_results, missingNeighbors, hit_rate, relative_error );
        if( rank == 0) cout << "Mean error (sampled): " << relative_error << "%" << endl;
        if( rank == 0) cout << "Hit rate (sampled): " << hit_rate << "%" << endl;
        if( rank == 0) cout << "Error check time: " << checktime << endl;
        if( rank == 0) cout << "Estimated exact search time: " << checktime*((double)nglobal/(double)logm) << endl;
        if( rank == 0) cout << "Number of missing neighbors: " << missingNeighbors << " out of " << logm*k  << endl;
      }
   }

   MPI_Finalize();

   delete[] ref;
   delete[] ids;

   return 0;
}


void getArgs(int argc, char **argv){
        // Read in the options
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        CmdLine cmd;
        const char *help = "Help";
        int dc;
        int qc;
	    int iL, iK, inum_threads;

        cmd.addInfo(help);
        cmd.addICmdOption("-d", &dim, 2, "Dimension of the space (default = 2).");
        cmd.addICmdOption("-n", &dc, 1, "Total number of input data points to read.");
        cmd.addRCmdOption("-err", &errorTarget, 0.05, "Maximum allowable mean relative distance error (default = 0.05).");
        cmd.addRCmdOption("-mult", &mult, 1.0, "Multiplier.  Used to determine search radius (1.0).");
        cmd.addICmdOption("-K", &iK, 0, "Number of hash functions per table (optional).");
        cmd.addICmdOption("-L", &iL, 10, "Maximum number of iterations (10).");
        cmd.addICmdOption("-k", &k, 20, "Number of earest neighbors to find.");
        cmd.addICmdOption("-bf", &bf, 2, "Bucket factor: Number of buckets = bf*size");
        cmd.addICmdOption("-auto", &autotune, 1, "Enable/disable autotuning (1).");
        cmd.addSCmdOption("-file", &ptrInputFile, "data.bin", "input binary file storing points");
        cmd.addICmdOption("-trials", &trials, 1, "Number of queries to run (5)");
        cmd.addICmdOption("-verify", &check, 0, "Check results?");
        cmd.read(argc, argv);

        strInputFile = ptrInputFile;
	    L = iL;
	    K=iK;

        nPts = dc;
}

