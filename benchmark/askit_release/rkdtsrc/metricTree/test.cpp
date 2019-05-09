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

#include "mpitree.h"
#include "clustering.h"
#include "repartition.h"
#include "metrictree.h"
#include "parallelIO.h"

using namespace Torch;
using namespace std;

__inline static
double find_max(double *arr, int first, int last)
{
	double maxv = 0.0;
	maxv = *max_element(arr+first, arr+last);
	return maxv;
}



void getArgs(int argc, char **argv);                    // get command-line arguments
void printMTree(pMetricNode in_node);
void saveMTree(char * filename, pMetricNode in_node);


// Command line arguments
int dim;                                // dimension
long numof_ref_points;                  // number of reference points
long numof_query_points;		// number of query points
long K;					// # of hash functions
long L;					// # of hash tables
int k;
int bf;
int check;
int gen;
double percentile;
double c;
long num_threads;
int tree_dbg_out;
int tree_time_out;
int max_tree_level;
int max_points_per_node;
int min_comm_size_per_tree_node;
int seedType;
string strInputFile;

int main(int argc, char **argv) {
   MPI_Init(&argc, &argv);

   int rank, size;
   double *ref, *query;
   long *refids;
   long *queryids;
   long nglobal;
   long mglobal;
   long refid_offset;
   long queryid_offset;

   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   getArgs(argc, argv);

   ref = new double[numof_ref_points*dim];
   query = new double[numof_query_points*dim];
  	
      
   switch(gen) {
     case 0:
       if(rank == 0) cout << "Distribution: Uniform" << endl;
       generateUniform(numof_ref_points, dim, ref, MPI_COMM_WORLD);
       generateUniform(numof_query_points, dim, query, MPI_COMM_WORLD);
       break;

     case 1:
       if(rank == 0) cout << "Distribution: Hypersphere shell" << endl;
       generateUnitHypersphere(numof_ref_points, dim, ref, MPI_COMM_WORLD);
       generateUnitHypersphere(numof_query_points, dim, query, MPI_COMM_WORLD);
       break;

     case 2:
       if(rank == 0) cout << "Distribution: Unit gaussian" << endl;
       generateNormal(numof_ref_points, dim, ref, MPI_COMM_WORLD);
       generateNormal(numof_query_points, dim, query, MPI_COMM_WORLD);
       break;
     
     case 3:
     {
       if(rank == 0) cout << "Distribution: Mixture of random gaussians" << endl;
       int *dummy_rlabels = new int[numof_ref_points];
       int *dummy_qlabels = new int[numof_query_points];
       double var[4] = { 1.5, 1.0, 1.0, 1.0 };

       generateMixOfUserGaussian( numof_ref_points, dim, 2, 0.5, 
		                   var, ref, dummy_rlabels, MPI_COMM_WORLD );
       generateMixOfUserGaussian( numof_query_points, dim, 2, 0.5, 
		        	   var, query, dummy_qlabels, MPI_COMM_WORLD );
       delete[] dummy_rlabels;
       delete[] dummy_qlabels;
       break;
     }
	 
	 case 4:
	 {
       if(rank == 0) cout << "Distribution: Line" << endl;
		int * dummy_refids = new int[numof_ref_points];
		int * dummy_queryids = new int[numof_query_points];
	   genPointInRandomLine(numof_ref_points, dim, ref, dummy_refids, MPI_COMM_WORLD, 1, rank*numof_ref_points);
       genPointInRandomLine(numof_query_points, dim, query, dummy_queryids, MPI_COMM_WORLD, 1, rank*numof_query_points);
	   delete [] dummy_refids;
	   delete [] dummy_queryids;
       break;
	 }
	 
	 case 5:
	 {
       if(rank == 0) cout << "User Input File: " << strInputFile << endl;
	   double *dummy_points;
	   long total_numof_input_points = numof_ref_points*size;
	   knn::parallelIO(strInputFile, total_numof_input_points, dim, dummy_points, MPI_COMM_WORLD);
       for(int i = 0; i < total_numof_input_points*dim; i++) query[i] = dummy_points[i];
       for(int i = 0; i < total_numof_input_points*dim; i++) ref[i] = dummy_points[i];
	   delete [] dummy_points;
	   break;
	 }
  
     default:
       cerr << "Invalid generator selection" << endl;
       exit(1);
   }

   MPI_Allreduce( &numof_ref_points, &nglobal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce( &numof_query_points, &mglobal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
	
   MPI_Scan(&numof_ref_points, &refid_offset, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
   MPI_Scan(&numof_query_points, &queryid_offset, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
 
   refid_offset -= numof_ref_points;
   queryid_offset -= numof_query_points;
   refids = new long[numof_ref_points];
   queryids = new long[numof_query_points];
   for(int i = 0; i < numof_ref_points; i++) refids[i] = refid_offset + (long)i;
   for(int i = 0; i < numof_query_points; i++) queryids[i] = queryid_offset + (long)i;

	for(int i = 0; i < numof_query_points*dim; i++) query[i] = ref[i];
	for(int i = 0; i < numof_query_points; i++) queryids[i] = refids[i];
  
   // ----- random sample logm points for evaluation purpose -----
   if(rank == 0) cout<<"nproc = "<<size<<endl;
   if(rank == 0) cout<<"\n------------- Sample log_2(m) points ------------"<<endl;
   int logm = std::min( (double)mglobal, 100.0*std::ceil(log10((double)mglobal)/log10(2.0)) );
   double *globalKdist = new double[logm*k];
   long *globalKid = new long [logm*k];
   long *sampleIDs = new long[logm];
   double *globalFurthestDist = new double [logm];

   double start_t, partial_direct_nn_t;
   double checktime = omp_get_wtime();
   if(check) {
     double *directquery = new double[logm*dim];
     double r;
     srand(time(NULL));
     for(int i = 0; i < logm; i++) 
	     sampleIDs[i] = (long)((((double)rand())/(double)RAND_MAX) * (double)mglobal);
     std::sort(sampleIDs, sampleIDs+logm);
     MPI_Bcast(sampleIDs, logm, MPI_LONG, 0, MPI_COMM_WORLD);
   
     vector< pair<long, int> > found;
     for(int i = 0; i < logm; i++) {
       for(int j = 0; j < numof_query_points; j++) {
         if( queryids[j] == sampleIDs[i] ){
             found.push_back( make_pair<long, int>(queryids[j], j) );
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
     MPI_Allgatherv(localSamples, sendcount, MPI_DOUBLE, directquery, 
		     rcvcounts, rcvdisp, MPI_DOUBLE, MPI_COMM_WORLD);
   
     //Perform a direct query on local reference points.
     // + find the furthest point from each sampled point
     double * tmpDist = new double [logm*numof_ref_points];
     double * localFurthestDist = new double [logm];
     knn::compute_distances(ref, directquery, numof_ref_points, logm, dim, tmpDist);
     for(int i = 0; i < logm; i++) {
     	localFurthestDist[i] = find_max(tmpDist, i*numof_ref_points, (i+1)*numof_ref_points);
     }
     MPI_Allreduce(localFurthestDist, globalFurthestDist, logm, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
     delete [] tmpDist;

     // + find the exact k nearest neighbors of each sampled point
     start_t = omp_get_wtime();
     pair<double, long> *localResult = new pair<double, long> [logm*k];
	 knn::directKQueryLowMem(ref, directquery, numof_ref_points, logm, k, dim, localResult);
     for(int i = 0; i < logm; i++) {
     	for(int j = 0; j < k; j++)
		localResult[i*k+j].second = refids[localResult[i*k+j].second];
     }
     pair<double, long> *mergedResult;
     knn::query_k(MPI_COMM_WORLD, k, 0, localResult, logm, k, mergedResult);

     partial_direct_nn_t = omp_get_wtime() - start_t;
     double max_nn_t = 0.0;
     MPI_Reduce(&partial_direct_nn_t, &max_nn_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
     if(rank == 0) cout<<"Estimated direct time: "<<((double)mglobal/(double)logm)*max_nn_t<<endl;

     if(rank == 0) {
       for(int i = 0; i < logm; i++) {
		   for(int j = 0; j < k; j++) {
			   globalKdist[i*k+j] = mergedResult[i*k+j].first;
				globalKid[i*k+j] = mergedResult[i*k+j].second;
			}
       }     
     }
     MPI_Bcast(globalKdist, logm*k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
     MPI_Bcast(globalKid, logm*k, MPI_LONG, 0, MPI_COMM_WORLD);

     delete[] mergedResult;
     delete[] localResult;
     delete[] rcvcounts;
     delete[] rcvdisp;
     delete[] directquery;
     delete[] localSamples;
     delete[] localFurthestDist;
   }
   checktime = omp_get_wtime() - checktime;
   MPI_Barrier(MPI_COMM_WORLD);

   // ----------------- MT Exact Test -----------------------------
   if (rank == 0) cout<<"\n\n----------- MT Exact Test ------------"<<endl;
   MetricData refData;
   refData.X.resize(dim*numof_ref_points);
   refData.gids.resize(numof_ref_points);
   refData.dim = dim;
   //cout<<"rank: "<<rank<<" ref: ";
   for(int i = 0; i < numof_ref_points*dim; i++) {
       refData.X[i] = ref[i];
	   //cout<<ref[i]<<" ";
   }
   //cout<<endl;

   for(int i = 0; i < numof_ref_points; i++) {
       refData.gids[i] = refids[i];
   }

   char ptrOutputName[256] = {0};
   if(tree_time_out) {   
	sprintf(ptrOutputName, "proc%05d_dim%03d_rank%05d_r%02d.info", 
			size, dim, rank, tree_time_out);	     
    remove(ptrOutputName);
   }

   MetricData queryData;
   queryData.X.resize(dim*numof_query_points);
   queryData.gids.resize(numof_query_points);
   queryData.dim = dim;
   for(int i = 0; i < numof_query_points*dim; i++)
       queryData.X[i] = query[i];
   for(int i = 0; i < numof_query_points; i++)
       queryData.gids[i] = queryids[i];
  	
   if(tree_dbg_out) {
   for(int i = 0; i < numof_query_points; i++) {
	//cout<<"gid: "<<queryData.gids[i]<<" coord: ";
		cout<<refData.gids[i]<<" ";
		for(int j = 0; j < dim; j++)
			cout<<refData.X[i*dim+j]<<" ";
		cout<<endl;
	}
	//cout<<endl;
	MPI_Barrier(MPI_COMM_WORLD);

	if(rank==0) cout<<endl;

	for(int i = 0; i < numof_query_points; i++) {
	//	//cout<<"gid: "<<queryData.gids[i]<<" coord: ";
		cout<<queryData.gids[i]<<" ";
		for(int j = 0; j < dim; j++)
			cout<<queryData.X[i*dim+j]<<" ";
		cout<<endl;
	}
	//cout<<endl;
	}




   // tree construction
   MetricNode root;
   root.options.debug_verbose=tree_dbg_out;
   root.options.timing_verbose=tree_time_out;
   root.options.pruning_verbose=true;
   int numof_kids=2;
   start_t = MPI_Wtime();
   root.Insert(	NULL, max_points_per_node, 
				max_tree_level, min_comm_size_per_tree_node,
				MPI_COMM_WORLD, &refData, seedType);
   MPI_Barrier(MPI_COMM_WORLD);
   if(rank == 0) cout<<"construction time: "<<MPI_Wtime() - start_t <<endl;
	
   if(tree_time_out) {
	MPI_Barrier(MPI_COMM_WORLD);
	saveMTree(ptrOutputName, &root);
	MPI_Barrier(MPI_COMM_WORLD);
   }

   // query
   vector< pair<double, long> > MTkNN;
   vector<long> *queryOutIDs = new vector<long>();
   start_t = MPI_Wtime();
   MTqueryK( &queryData, numof_ref_points*size, 16.0, &root, k, queryOutIDs, &MTkNN);
   
   double query_t = MPI_Wtime() - start_t;

   // evaluation
   if(check) {
     double start = omp_get_wtime();
     //Check error for random sample of query points
     double localErrorSum = 0.0;
     double globalErrorSum;
     double localnn2fur = 0.0;
     double globalnn2fur = 0.0;
     double localHitRate = 0.0;
     double globalHitRate = 0.0;
     for(int i = 0; i < (*queryOutIDs).size(); i++) {
       for(int j = 0; j < logm; j++) {
         if( (*queryOutIDs)[i] == sampleIDs[j] ) {
	    for(int t = 0; t < k; t++) {
              vector<long> tmpVector;
	      vector<long>::iterator it;
	      tmpVector.resize(k);
	      copy(globalKid+j*k, globalKid+j*k+k, tmpVector.begin());
	      it = find(tmpVector.begin(), tmpVector.end(), MTkNN[i*k+t].second);
	      if(it != tmpVector.end()) localHitRate += 1.0;
              localnn2fur += std::sqrt(globalKdist[j*k+t]) / std::sqrt(globalFurthestDist[j]);
	      if(MTkNN[i*k+t].second != -1L && std::sqrt(globalKdist[j*k+t]) > 0.000000000001 ) {
           	double error = std::abs( std::sqrt(MTkNN[i*k+t].first) - std::sqrt(globalKdist[j*k+t]) );
	        localErrorSum += error / std::sqrt(globalKdist[j*k+t]);
	      }
	   }
	 } // if (outIDs == sampleIDs)
       }
     }
     
     localErrorSum /= (double)k; 
     localnn2fur /= (double)k;
     localHitRate /= (double)k;
     MPI_Reduce(&localErrorSum, &globalErrorSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
     MPI_Reduce(&localnn2fur, &globalnn2fur, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
     MPI_Reduce(&localHitRate, &globalHitRate, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
     checktime += omp_get_wtime() - start;
     if( rank == 0) cout << "+++++++++++++++++++++++++++++++++ "<<endl;
     if(rank == 0) cout<<"query time: "<< query_t <<endl;
     if( rank == 0) cout << "Hit Rate (sampled): " << globalHitRate/(double)logm * 100.0 << "%" << endl; 
     if( rank == 0) cout << "Mean error (sampled): " << globalErrorSum/(double)logm * 100.0 << "%" << endl; 
     if( rank == 0) cout << "Nearest/Furthest (sampled): " << globalnn2fur/(double)logm << endl;
     if( rank == 0) cout << "Error check time: " << checktime << endl; 

   }
   MPI_Barrier(MPI_COMM_WORLD);

   MPI_Finalize();

   delete[] ref;
   delete[] query;
   delete[] refids;
   delete[] queryids;
   delete[] globalKdist;
   delete[] globalKid;
   delete[] globalFurthestDist;
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
	int iL, iK, inum_threads;
	int rn, qn;
	char *ptrInputFile = NULL;
	int fn;

    cmd.addInfo(help);
    cmd.addSCmdOption("-file", &ptrInputFile, "data.bin", "input binary file storing points");
    //cmd.addICmdOption("-fn", &fn, 36864, "number of points in the input file, used with -file (default = 36864).");
    cmd.addICmdOption("-d", &dim, 4, "Dimension of the space (default = 4).");
    cmd.addICmdOption("-rn", &rn, 20, "Number of referrence data points per process (default = 20).");
    cmd.addICmdOption("-qn", &qn, 20, "Number of query data points per process (default = 20).");
    cmd.addRCmdOption("-percentile", &percentile, 0.5, "Percentile for r-selection (0.5).");
    cmd.addRCmdOption("-c", &c, 3.0, "Approximation factor (used to determin L; default = 3.0).");
    cmd.addICmdOption("-K", &iK, 2, "Number of hash functions per table (2).");
    cmd.addICmdOption("-L", &iL, 2, "Number of hash tables (2).");
    cmd.addICmdOption("-k", &k, 2, "Number of earest neighbors to find (2).");
    cmd.addICmdOption("-bf", &bf, 2, "Bucket factor: Number of buckets = bf*size (2).");
    cmd.addICmdOption("-gen", &gen, 0, "Data generator (0). 0=uniform, 1=hypersphere, 2=unit guassian, 3=mix of gaussians, 4=line, 5=user specified.");
    cmd.addICmdOption("-t", &inum_threads, 8, "Number of OpenMP threads (8).");
    cmd.addICmdOption("-verify", &check, 0, "Check results? (0)");
    cmd.addICmdOption("-dbg", &tree_dbg_out, 0, "Enable tree debugging output (0)");
    cmd.addICmdOption("-time", &tree_time_out, 0, "Enable tree timing output (0)");
    cmd.addICmdOption("-seed", &seedType, 1, "0:random, 1: ostrovskey (1)");
    cmd.addICmdOption("-mtl", &max_tree_level, 10, "maximum tree depth (10)");
    cmd.addICmdOption("-mppn", &max_points_per_node, 1000, "maximum number of points per tree node (1000)");
    cmd.addICmdOption("-mcsptn", &min_comm_size_per_tree_node, 1, "min comm size per tree node (1)");
    cmd.read(argc, argv);

	L = iL;
	K=iK;
	num_threads=inum_threads;

    numof_ref_points = rn;
	numof_query_points = qn;

	strInputFile = ptrInputFile;
	//total_numof_input_points = fn;
}



void printMTree(pMetricNode in_node)
{
	int nproc, rank;
	//MPI_Comm in_comm = in_node->comm;
	//MPI_Comm_size(in_comm, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if(NULL == in_node->kid) { 	// if leaf
		cout<<rank
		    <<" - "<<pow(2.0, (double)in_node->level)-1+in_node->chid
		    <<" - "<<in_node->level
		    <<" - "<<in_node->data->gids.size()
		    <<endl;
		fflush(stdout);
		return;
	}
	else {				// if not leaf
		printMTree(in_node->kid);
	}

}



void saveMTree(char * filename, pMetricNode in_node)
{
	int worldnproc, worldrank, commsize;
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
	MPI_Comm_size(in_node->comm, &commsize);

	ofstream fout(filename, ios::app|ios::out);
        
	if(NULL == in_node->kid) { 	// if leaf

		fout<<worldrank
		    <<" "<<in_node->level
		    <<" "<<pow(2.0, (double)in_node->level)-1+in_node->chid
		    <<" "<<in_node->data->gids.size()
		    <<" "<<commsize
		    <<endl;
		//for(int i = 0; i < in_node->data->gids.size(); i++)
		//	fout<<in_node->data->gids[i]<<" ";
		//fout<<endl;

		fout.flush();
		fout.close();
		return;
	}
	else {				// if not leaf
		saveMTree(filename, in_node->kid);
	}

}


