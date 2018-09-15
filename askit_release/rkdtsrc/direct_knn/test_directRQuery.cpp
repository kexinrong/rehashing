#include<direct_knn.h>
#include<generator.h>
#include<vector>
#include<cassert>
#include<cmath>
#include<utility>
#include<omp.h>
#include<mpi.h>
#include<CmdLine.h>
#include <ompUtils.h>

using namespace Torch;
using namespace std;

void getArgs(int argc, char **argv);                    // get command-line arguments
vector< pair<double, long> >* naiveQuery(double* ref, long n, double *query,long  m, double r, long dim);

// Command line arguments
int dim;                                // dimension
long nPts;                              // number of reference points
double r;                               // Search radius





int main(int argc, char **argv) {
   MPI_Init(&argc, &argv);
   int rank, size;
   double *ref;
   long *ids;

   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   assert( size == 1 );

   getArgs(argc, argv);

   ref = (double*) malloc(nPts*dim*sizeof(double));
   ids = (long*) malloc(nPts*sizeof(long));
   for(int i = 0; i < nPts; i++) ids[i] = (long)rank*(long)nPts + (long)i;
  
/* 
   int *iids = new int[nPts];
   genPointInRandomLine(nPts, dim, ref, iids, MPI_COMM_WORLD, false, rank * nPts);
   delete[] iids;
*/

   generateNormal(nPts, dim, ref, MPI_COMM_WORLD);

   std::pair<double, long> *result;
   int *neighbor_count, *correct_neighbor_counts;

   double start = omp_get_wtime();
   knn::directRQuery( ref, ref, nPts, nPts, r*r, dim, ids, &neighbor_count, &result );
   cout << "Query time: " << omp_get_wtime() - start << endl;

   vector< pair<double, long> >* correctResults;
   correctResults = naiveQuery ( ref, nPts, ref, nPts,  r, dim);

   int id_mismatches = 0;
   int dist_mismatches = 0;
   int curr = 0;  
   int *neighbor_offset = new int[nPts+1];
   omp_par::scan(neighbor_count, neighbor_offset, nPts);
   neighbor_offset[nPts] = neighbor_offset[nPts-1] + neighbor_count[nPts-1];

   #pragma omp parallel for
   for( int i = 0; i < nPts; i++ ) {
      for( int j = 0; j < correctResults[i].size(); j++ ) {
         int k;
         for( k = neighbor_offset[i]; k < neighbor_offset[i+1]; k++) {
            if(correctResults[i][j].second == result[k].second) {
               if( abs(correctResults[i][j].first - result[k].first) > 1.0e-6) {
                  #pragma omp critical
                  {
                     dist_mismatches++;
                  }
               }
               break;
            }
         }
         if( k == neighbor_offset[i+1] ) {
            #pragma omp critical
            {
               dist_mismatches++;
               id_mismatches++;
            }
         }
      }
   }


   cout << "ID mismatches: " << id_mismatches << " Distance mismatches: " << dist_mismatches << endl;

   MPI_Finalize();

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
        cmd.addICmdOption("-dc", &dc, 20, "Number of data points per process (default = 20).");
        cmd.addRCmdOption("-r", &r, 0, "Search radius for queryR (default = 0).");
        cmd.read(argc, argv);

        r *= r;
        nPts = dc;
}


vector< pair<double, long> >* naiveQuery(double* ref, long n, double *query,long  m, double r, long dim){
 
   r = r*r;
   omp_set_num_threads( omp_get_num_procs() );

   vector< pair<double, long> > *near_neighbors = new vector< pair<double, long> >[m];
   #pragma omp parallel for
   for (long queryPtIndex = 0; queryPtIndex < m; queryPtIndex++){
      //for every query point...
      double dist;
      for (long dataPtIndex=0; dataPtIndex < n; dataPtIndex++) {
         //calculate the distance to every data point
         dist=0.0;
         for (unsigned long i=0; i<dim; i++) {
            //dist+= pow(((*iter)[i] - (*dataIter)[i]), 2);
            dist+= (ref[dim*dataPtIndex + i] - query[dim*queryPtIndex + i])
                      * (ref[dim*dataPtIndex + i] - query[dim*queryPtIndex + i]);

         }

         //add to appropriate vector if/when the distance is smaller than specified.
         if (dist <= r) {
            int s = near_neighbors[queryPtIndex].size();
            near_neighbors[queryPtIndex].resize(s + 1);
            near_neighbors[queryPtIndex][s].first = dist;
            near_neighbors[queryPtIndex][s].second = dataPtIndex;
          
         }
      }
   }
   return near_neighbors;
}


