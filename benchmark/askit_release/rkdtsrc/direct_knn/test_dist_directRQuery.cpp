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

// Command line arguments
int dim;                                // dimension
long nPts;                              // number of reference points
double r;                               // Search radius





int main(int argc, char **argv) {
   MPI_Init(&argc, &argv);
   int rank, size;
   double *ref, *query;
   long *ids;
   long nglobal;
   long id_offset;

   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   getArgs(argc, argv);

   //Give each process a slightly different number of points.
   srand(time(NULL)*rank);
   int size_adjustment = rand() % (nPts/10);
   nPts = nPts-size_adjustment;

   MPI_Allreduce( &nPts, &nglobal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
   MPI_Scan(&nPts, &id_offset, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
   id_offset -= nPts;

   ref = (double*) malloc(nPts*dim*sizeof(double));
   query = (double*) malloc(nPts*dim*sizeof(double));
   ids = (long*) malloc(nPts*sizeof(long));
   for(int i = 0; i < nPts; i++) ids[i] = id_offset + (long)i;
  
/* 
   int *iids = new int[nPts];
   genPointInRandomLine(nPts, dim, ref, iids, MPI_COMM_WORLD, false, rank * nPts);
   delete[] iids;
*/
   
   generateNormal(nPts, dim, ref, MPI_COMM_WORLD);


   //Copy reference points to query array for all-to-all query
   for( int i = 0; i < nPts*dim; i++ ) query[i] = ref[i];

   vector< pair<double, long> >* dist_results = new vector< pair<double, long> >[nPts];

   knn::dist_directRQuery( ref, query, nglobal, nglobal, nPts, nPts, r,
                dim, ids, dist_results, MPI_COMM_WORLD); 

   double *all_ref;
   long *all_ids; 

   if( rank == 0) {
      all_ref = new double[nglobal*dim];
      all_ids = new long[nglobal];
   }


   int *pointcounts = new int[size];
   int *id_displacement = new int[size+1];
   int *valuecounts = new int[size];
   int *value_displacement = new int[size];
   MPI_Gather(&nPts, 1, MPI_INT, pointcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
   omp_par::scan(pointcounts, id_displacement, size);
   id_displacement[size] = nglobal;
   for( int i = 0; i < size; i++ ) {
      value_displacement[i] = id_displacement[i]*dim;
      valuecounts[i] = pointcounts[i] * dim;
   }

   MPI_Gatherv(ids, nPts, MPI_LONG, all_ids, pointcounts, id_displacement, MPI_LONG, 0, MPI_COMM_WORLD);
   MPI_Gatherv(ref, nPts*dim, MPI_DOUBLE, all_ref, valuecounts, value_displacement, MPI_DOUBLE, 0, MPI_COMM_WORLD);

/*
   delete [] ref;
   delete [] query;
   delete [] ids;
*/
   std::pair<double, long> *seq_result;
   int *seq_neighbor_count, *correct_neighbor_counts;

   if( rank == 0 ) {
      // Now, gather all points at rank 0, and perform a sequential query.
      knn::directRQuery( all_ref, all_ref, nglobal, nglobal, r, dim, all_ids, &seq_neighbor_count, &seq_result );

      delete[] all_ids;
      delete[] all_ref;

   }


   //Scatter the results to appropriate proccesses for comparison.
   int *send_counts = new int[size];
   int *disp = new int[size];
   int recv_count;

   if( rank == 0 ) {
      #pragma omp parallel for
      for(int i = 0; i < size; i++) send_counts[i] = 0;
      for(int i = 0; i < size; i++) {
         for( int j = id_displacement[i]; j < id_displacement[i+1]; j++ ) 
            send_counts[i] += seq_neighbor_count[j];
      }
      omp_par::scan(send_counts, disp, size);
   }

   MPI_Scatter( send_counts, 1, MPI_INT, &recv_count, 1, MPI_INT, 0, MPI_COMM_WORLD );
    
   correct_neighbor_counts = new int[nPts];
   MPI_Scatterv( seq_neighbor_count, pointcounts, id_displacement,  MPI_INT, 
                 correct_neighbor_counts, nPts, MPI_INT, 0, MPI_COMM_WORLD );
    
   int *neighbor_offset = new int[nPts+1];
   omp_par::scan(correct_neighbor_counts, neighbor_offset, nPts);

   pair<double, long> *result_buffer = new pair<double, long>[recv_count];

   MPI_Datatype pairtype;
   MPI_Type_contiguous( sizeof(pair<double, long>), MPI_BYTE, &pairtype);
   MPI_Type_commit(&pairtype);


   MPI_Scatterv( seq_result, send_counts, disp, pairtype, result_buffer, 
                 recv_count, pairtype, 0, MPI_COMM_WORLD );

   neighbor_offset[nPts] = recv_count;

   //Now, compare the neighbors
   int id_mismatches = 0;
   int dist_mismatches = 0;
   int curr = 0;  

   for( int i = 0; i < nPts; i++ ) {
      for( int j = neighbor_offset[i]; j < neighbor_offset[i+1]; j++ ) {
         int k;
         for( k = 0; k < dist_results[i].size(); k++) {
            if(dist_results[i][k].second == result_buffer[j].second) {
               if( abs(dist_results[i][k].first - result_buffer[j].first) > 1.0e-6) 
                  dist_mismatches++;
               break;
            }
         }
         if( k == dist_results[i].size() ) {
            dist_mismatches++;
            id_mismatches++;
         }
      }
   }


   int total_id_mismatches, total_dist_mismatches;
   MPI_Reduce(&id_mismatches, &total_id_mismatches, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
   MPI_Reduce(&dist_mismatches, &total_dist_mismatches, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

   if( rank == 0) cout << "ID mismatches: " << total_id_mismatches << " Distance mismatches: " << total_dist_mismatches << endl;


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

