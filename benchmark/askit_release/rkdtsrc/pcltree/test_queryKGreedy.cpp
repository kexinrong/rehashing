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
#include <mpitree.h>


using namespace Torch;
using namespace std;
using namespace knn;

void getArgs(int argc, char **argv);                    // get command-line arguments

// Command line arguments
int dim;                                // dimension
long nPts;                              // number of reference points
long K;					// # of hash functions
long L;					// # of hash tables
int k;
int bf;
int check;
int gen;
int iters;
double percentile;
double c;
long num_threads;			
int max_tree_level;
int max_points_per_node;
int min_comm_size_per_tree_node;
int seedType;


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

   MPI_Allreduce( &nPts, &nglobal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
   MPI_Scan(&nPts, &id_offset, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
   id_offset -= nPts;

   ref = new double[nPts*dim];
   query = new double[nPts*dim];
   ids = new long[nPts];
   queryids = new long[nPts];
   for(int i = 0; i < nPts; i++) ids[i] = id_offset + (long)i;
   for(int i = 0; i < nPts; i++) queryids[i] = id_offset + (long)i;


   switch(gen) {
 
     case 0:
       if(rank == 0) cout << "Distribution: Uniform" << endl;
       generateUniform(nPts, dim, ref, MPI_COMM_WORLD);
       break;

     case 1:
       if(rank == 0) cout << "Distribution: Hypersphere shell" << endl;
       generateUnitHypersphere(nPts, dim, ref, MPI_COMM_WORLD);
       break;

     case 2:
       if(rank == 0) cout << "Distribution: Unit gaussian" << endl;
       generateNormal(nPts, dim, ref, MPI_COMM_WORLD);
       break;

     case 3:
     {
       if(rank == 0) cout << "Distribution: Mixture of random gaussians" << endl;
       int *iids = new int[nPts];
       double var[4] = { 1.5, 1.0, 1.0, 1.0 };
       generateMixOfUserGaussian( nPts, dim, 2, 0.5, var, ref, iids, MPI_COMM_WORLD );
       delete[] iids;
       break;
     }
   
     default:
       cerr << "Invalid generator selection" << endl;
       exit(1);
   }

   //Copy reference points to query array for all-to-all query
   for( int i = 0; i < nPts*dim; i++ ) query[i] = ref[i];



   std::pair<double, long> *dist_results;
   int logm = nPts; //std::ceil( 10.0*log10((double)nglobal)/log10(2.0) );
   double *globalKth = new double[logm];
   long *sampleIDs = new long[logm];
   double checktime = omp_get_wtime();
   if(check) {
     double *directquery;

     // Sample log_2(m) query points, and compute exact results for error checking.
     directquery = new double[logm*dim];
     double r;
   
     //Sample log_2(mLocal) query points, and broadcast from rank 0.
     srand(42);
     for(int i = 0; i < logm; i++) sampleIDs[i] = (long)((((double)rand())/(double)RAND_MAX) * (double)nglobal);
     MPI_Bcast(sampleIDs, logm, MPI_LONG, 0, MPI_COMM_WORLD);
   
     //Search for selected query points in local query set.  Use linear search because of small number of samples.
     vector< pair<long, int> > found;
     #pragma omp parallel for
     for(int i = 0; i < logm; i++) {
       for(int j = 0; j < nPts; j++) {
         if( queryids[j] == sampleIDs[i] ){
           #pragma omp critical
           { 
             found.push_back( make_pair<long, int>(queryids[j], j) );
           }
         }
       }
     }
   
     int numFound = found.size();
     double *localSamples = new double[logm*dim];
     for(int i = 0; i < numFound; i++) {
       for(int j = 0; j < dim; j++) {
         localSamples[i*dim+j] = query[found[i].second*dim+j];
       }
     }
   
     //Distribute sample query points to all processes.
     int sendcount = numFound*dim;
     int *rcvcounts = new int[size];
     int *rcvdisp = new int[size];
     MPI_Allgather(&sendcount, 1, MPI_INT, rcvcounts, 1, MPI_INT, MPI_COMM_WORLD);
     omp_par::scan(rcvcounts, rcvdisp, size);
     assert( (rcvdisp[size-1]+rcvcounts[size-1])/dim == logm);
     MPI_Allgatherv(localSamples, sendcount, MPI_DOUBLE, directquery, rcvcounts, rcvdisp, MPI_DOUBLE, MPI_COMM_WORLD);
   
     //Perform a direct query on local reference points.
     pair<double, long> *localResult = knn::directKQueryLowMem(ref, directquery, nPts, logm, k, dim);
     pair<double, long> *mergedResult;
     knn::query_k(MPI_COMM_WORLD, k, 0, localResult, logm, k, mergedResult);

     if(rank == 0) {
       for(int i = 0; i < logm; i++) {
         globalKth[i] = mergedResult[i*k+k-1].first;
       }     
     }
     MPI_Bcast(globalKth, logm, MPI_DOUBLE, 0, MPI_COMM_WORLD);

     delete[] mergedResult;
     delete[] localResult;
     delete[] rcvcounts;
     delete[] rcvdisp;
     delete[] directquery;
     delete[] localSamples;
   }
   checktime = omp_get_wtime() - checktime;


   vector< pair<double, long> > kNN;
   vector<long> curr_query_IDs;
   vector< pair<double, long> > curr_nearest;

   double querytime = MPI_Wtime();
   for(int l = 0; l < iters; l++ ) {

   
      MTData data;
      data.X.resize(dim*nPts);  // points
      data.gids.resize(nPts);       // global ids
      data.dim = dim;
   
      MTData queryData;
      queryData.X.resize(dim*nPts);
      queryData.gids.resize(nPts);
      queryData.dim = dim;
   
      for(int i = 0; i < nPts*dim; i++) data.X[i] = ref[i];
      for(int i = 0; i < nPts*dim; i++) queryData.X[i] = query[i];
      for(int i = 0; i < nPts; i++) data.gids[i] = rank*nPts+i;
      for(int i = 0; i < nPts; i++) queryData.gids[i] = rank*nPts+i;
   
   
      MTNode root;
      root.options.pruning_verbose=true;
      double time = MPI_Wtime();
      root.Insert( NULL,
         max_points_per_node,
         max_tree_level,
         min_comm_size_per_tree_node,
         MPI_COMM_WORLD,
         &data,
	 seedType);
      MPI_Barrier(MPI_COMM_WORLD);
      if( rank == 0 ) cout << "construction: " << MPI_Wtime() - time << endl;
   
      vector<long> outQueryIDs; 
      vector< pair<double, long> > new_nearest;
      queryK_GreedyApprox( &queryData, nPts*size, &root, k, &outQueryIDs,  &new_nearest);
   
      if(curr_query_IDs.size() == 0) {
         curr_query_IDs = outQueryIDs;
      }  

      if(curr_nearest.size() == 0) {
         curr_nearest = new_nearest;
      } else {
         assert(curr_nearest.size() == new_nearest.size());
         pair<double, long> *merged = knn::lsh::merge_results( &new_nearest[0], &curr_nearest[0], outQueryIDs.size(), k );
         for(int i = 0; i < new_nearest.size(); i++) curr_nearest[i] = merged[i];
         delete[] merged;
      }
   }

   kNN = curr_nearest;


   MPI_Barrier(MPI_COMM_WORLD);
   if(rank == 0) cout << "query: " << MPI_Wtime() - querytime << endl;


   if(check) {
     double start = omp_get_wtime();

     //Check error for random sample of query points
     double localErrorSum = 0.0;
     double globalErrorSum;
     for(int i = 0; i < curr_query_IDs.size(); i++) {
       for(int j = 0; j < logm; j++) {
         if( curr_query_IDs[i] == sampleIDs[j] ) {
           int kth = k-1;
           double error = std::abs( std::sqrt(kNN[i*k+kth].first) - std::sqrt(globalKth[j]) );
           localErrorSum += error / std::sqrt(globalKth[j]);
         }
       }
     }
     MPI_Reduce(&localErrorSum, &globalErrorSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
     checktime += omp_get_wtime() - start;
     if( rank == 0) cout << "Mean error (sampled): " << globalErrorSum/(double)logm * 100.0 << "%" << endl;
     if( rank == 0) cout << "Error check time: " << checktime << endl;
     if( rank == 0) cout << "Estimated exact search time: " << checktime*size << endl;

   }

   MPI_Finalize();

   delete[] ref;
   delete[] query;
   delete[] ids;
   delete[] queryids;
   delete[] globalKth;
   delete[] sampleIDs;


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
		
        cmd.addInfo(help);
        cmd.addICmdOption("-d", &dim, 2, "Dimension of the space (default = 2).");
        cmd.addICmdOption("-s", &seedType, 1, "0:random seeds, 1:ostrovsky seeds (default = 1).");
        cmd.addICmdOption("-dc", &dc, 20, "Number of data points per process (default = 20).");
        cmd.addICmdOption("-k", &k, 20, "Number of earest neighbors to find.");
        cmd.addICmdOption("-mtl", &max_tree_level, 4, "maximum tree depth");
        cmd.addICmdOption("-mppn", &max_points_per_node, 10, "maximum number of points per tree node");
        cmd.addICmdOption("-mcsptn", &min_comm_size_per_tree_node, 1, "min comm size per tree node");
        cmd.addICmdOption("-gen", &gen, 1, "Data generator (1). 0=uniform, 1=hypersphere, 2=unit guassian, 3=mix of gaussians.");
        cmd.addICmdOption("-i", &iters, 1, "Number of iterations (1).");
        cmd.addICmdOption("-verify", &check, 0, "Check results?");
        cmd.read(argc, argv);

        nPts = dc;
}

