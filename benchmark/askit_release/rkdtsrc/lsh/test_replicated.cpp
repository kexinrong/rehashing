#include<direct_knn.h>
#include<knnreduce.h>
#include<lsh.h>
#include<generator.h>
#include<vector>
#include<cassert>
#include<cmath>
#include<utility>
#include<omp.h>
#include<mpi.h>
#include<CmdLine.h>
#include <ompUtils.h>
#include <ctime>
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
int gen;
double errorTarget;
double mult;
long num_threads;			
int trials;
int autotune;


int main(int argc, char **argv) {
   MPI_Init(&argc, &argv);
   int rank, size;
   double *ref, *query;
   long *ids;
   long *queryids;
   long nglobal;
   long id_offset;

   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   getArgs(argc, argv);

   init_papi();
   srand(time(NULL)+rank);

   MPI_Allreduce( &nPts, &nglobal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
   MPI_Scan(&nPts, &id_offset, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
   id_offset -= nPts;

   int *iids = new int[nPts];
   ids = new long[nPts];
   queryids = new long[nPts];
   query = new double[nPts*dim];
   ref = new double[nPts*dim];
   for(int i = 0; i < nPts; i++) ids[i] = id_offset + (long)i;
   for(int i = 0; i < nPts; i++) queryids[i] = id_offset + (long)i;


   switch(gen) {
 
     case 0:
       if(rank == 0) cout << "Distribution: Uniform" << endl;
       generateUniform(nPts, dim, ref, MPI_COMM_WORLD);
       sleep(1);
       generateUniform(nPts, dim, query, MPI_COMM_WORLD);
       break;

     case 1:
       if(rank == 0) cout << "Distribution: Hypersphere shell" << endl;
       generateUnitHypersphere(nPts, dim, ref, MPI_COMM_WORLD);
       sleep(1);
       generateUnitHypersphere(nPts, dim, query, MPI_COMM_WORLD);
       break;

     case 2:
       if(rank == 0) cout << "Distribution: Unit gaussian" << endl;
       generateNormal(nPts, dim, ref, MPI_COMM_WORLD);
       sleep(1);
       generateNormal(nPts, dim, query, MPI_COMM_WORLD);
       break;

     case 3:
     {
       if(rank == 0) cout << "Distribution: Mixture of random gaussians" << endl;
       assert(nPts%2 == 0);
       generateNormal(nPts, dim, ref, MPI_COMM_WORLD);
       sleep(1);
       generateNormal(nPts, dim, query, MPI_COMM_WORLD);
       #pragma omp parallel for
       for(int i = 0; i < (nPts/2)*dim; i++) ref[i] *= 2.0;  //Scale half of the dataset
       #pragma omp parallel for
       for(int i = nPts/2; i < nPts; i++) ref[i*dim] += 1.0;  //Shift half of the dataset
       break;
     }

     case 4:
       delete [] ref;
       delete [] query;
       ref = new double[nPts*2*dim];
       if(rank == 0) cout << "Distribution: 10-d unit gaussian embedded in " << dim << "dimensions"  << endl;
       generateNormalEmbedding(nPts*2, dim, 10, ref, MPI_COMM_WORLD);

       query = new double[nPts*dim];
       #pragma omp parallel for
       for(int i = 0; i < nPts*dim; i++) query[i] = ref[nPts*dim+i];
       double *newRef = new double[nPts*dim];
       #pragma omp parallel for
       for(int i = 0; i < nPts*dim; i++) newRef[i] = ref[i];
       delete [] ref;
       ref = newRef;

       break;
  
     default:
       cerr << "Invalid generator selection" << endl;
       exit(1);
   }
   delete[] iids;


   MPI_Bcast( query, nPts*dim, MPI_DOUBLE, 0, MPI_COMM_WORLD );
   MPI_Bcast( queryids, nPts, MPI_LONG, 0, MPI_COMM_WORLD );


   float direct_mflops = 0.0; 
   double checktime = omp_get_wtime();
   vector<long> sampleIDs;
   vector<double> globalKdist;
   vector<long> globalKid;
   double hit_rate, relative_error=DBL_MAX;
   int missingNeighbors = INT_MAX;
   int logm = std::ceil( 100.0*log10((double)nglobal)/log10(2.0) );

   if(check) {
     int mLocal;
     if( rank == 0 ) 
       mLocal = nPts;
     else
       mLocal = 0;
     get_sample_info(ref, query, ids, queryids, nPts, mLocal, dim, k,
                   sampleIDs, globalKdist, globalKid);
   }
   checktime = omp_get_wtime() - checktime;



   //Now, perform distributed LSH search.  
   for(int t = 0; t < trials; t++) {
      std::vector< pair<double,long> > lsh_results;
      std::vector<long> outIDs;
      MPI_Barrier(MPI_COMM_WORLD);

      //Choose range for search.
      double rtime = omp_get_wtime();
      double r;
      papi_mpi_flop_start();


      double start = MPI_Wtime();

      knn::lsh::distLSHreplicatedQ( ref, ids, query, queryids, nglobal, nPts, nPts,
                dim, k, L, bf, mult, lsh_results, outIDs, MPI_COMM_WORLD );

      MPI_Barrier(MPI_COMM_WORLD);
      float query_mflops =  papi_mpi_flop_stop(); 
      double querytime = MPI_Wtime() - start;
      if (rank == 0) cout << "LSH time: " << querytime << endl;




      if (rank == 0 && query_mflops > 0.0) cout << "Query TFLOPS: " << query_mflops /1.0e6 << endl;
   
     if(check) {
        //Check error for random sample of query points
	if(rank == 0)  {
          verify( sampleIDs, globalKdist, globalKid, outIDs, lsh_results, MPI_COMM_SELF, missingNeighbors, hit_rate, relative_error );
          if( rank == 0) cout << "Mean error (sampled): " << relative_error << "%" << endl; 
          if( rank == 0) cout << "Hit rate (sampled): " << hit_rate << "%" << endl; 
          if( rank == 0) cout << "Error check time: " << checktime << endl; 
          if (rank == 0 && direct_mflops > 0.0) cout << "Direct MFLOPS: " << direct_mflops << endl;
          if( rank == 0) cout << "Estimated exact search time: " << checktime*((double)nglobal/(double)logm) << endl; 
          if( rank == 0) cout << "Number of missing neighbors: " << missingNeighbors << " out of " << logm*k  << endl; 
        }
      }
   }


   MPI_Finalize();

   delete[] ref;
   delete[] query;
   delete[] ids;
   delete[] queryids;

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
        cmd.addICmdOption("-dc", &dc, 20, "Number of data points per process (default = 20).");
        cmd.addRCmdOption("-err", &errorTarget, 0.05, "Maximum allowable mean relative distance error (default = 0.05).");
        cmd.addRCmdOption("-mult", &mult, 1.0, "Multiplier.  Used to determine search radius (1.0).");
        cmd.addICmdOption("-K", &iK, 0, "Number of hash functions per table (2).");
        cmd.addICmdOption("-L", &iL, 10, "Maximum number of iterations (10).");
        cmd.addICmdOption("-k", &k, 20, "Number of earest neighbors to find.");
        cmd.addICmdOption("-bf", &bf, 2, "Bucket factor: Number of buckets = bf*size");
        cmd.addICmdOption("-auto", &autotune, 1, "Enable/disable autotuning (1).");
        cmd.addICmdOption("-gen", &gen, 1, "Data generator (1). 0=uniform, 1=hypersphere, 2=unit guassian, 3=mix of gaussians.");
        cmd.addICmdOption("-trials", &trials, 1, "Number of queries to run (5)");
        cmd.addICmdOption("-verify", &check, 0, "Check results?");
        cmd.read(argc, argv);

	L = iL;
	K=iK;

        nPts = dc;
}

