#include <mpi.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <omp.h>
#include "repartition.h"
#include "clustering.h"
#include "direct_knn.h"
#include "mpitree.h"
#include "lsh.h"
#include "verbose.h"

using namespace std;
using namespace knn;
using namespace knn::repartition;

bool my_compare_function(int i, int j) {
	return ( i < j );
}

void queryR( pMTData inData, long rootNpoints, double dupFactor,
	     pMTNode searchNode, 
	     double range, 
	     vector< pair<double, long> > *&neighbors)
{
  int size, rank;
  MPI_Comm comm = searchNode->comm;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
	
  int numof_kids = 2;
  int dim = inData->dim;
  int numof_query_points = inData->X.size() / dim;
  int global_numof_query_points = 0;
  int total_query_points;
  if(searchNode->options.pruning_verbose) {
    MPI_Allreduce( &numof_query_points, &total_query_points, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
  }


  MTData* leafData;
  pMTNode leaf;
  distributeToLeaves(inData, rootNpoints, dupFactor, searchNode, range,  &leafData, &leaf);
  inData->Copy(leafData);
  delete leafData;
  searchNode = leaf;
  int global_numof_ref_points = searchNode->Nglobal;
  comm = searchNode->comm;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  //If the "leaf" we got is not a leaf, find its leaf node.
  while( searchNode->kid )
    searchNode = searchNode->kid;

  numof_query_points = inData->X.size() / dim;

  MPI_Allreduce( &numof_query_points, &global_numof_query_points, 1, MPI_INT, MPI_SUM, comm );
  int numof_ref_points = searchNode->data->X.size() / dim;

	
  long *glb_ref_ids = (long*) malloc(numof_ref_points*sizeof(long));
  double *ref_points = (double*)malloc(numof_ref_points*dim*sizeof(double));
  #pragma omp parallel for
  for(int i = 0; i < numof_ref_points; i++) 
  	glb_ref_ids[i] = searchNode->data->gids[i];

  #pragma omp parallel for
  for(int i = 0; i < numof_ref_points*dim; i++)
  	ref_points[i] = searchNode->data->X[i];
		
  neighbors = new vector< pair<double, long> >[numof_query_points];



  //Optionally print verbose pruning info.
  if(searchNode->options.pruning_verbose) {
    //double mypruning = 100.0 - (((double)global_numof_query_points)/((double)rootNpoints))*100.0;
    double mypruning = ( (double)(total_query_points - global_numof_query_points) ) 
							/ ( (double)(total_query_points - global_numof_ref_points) ) * 100.0;
	  double maxpruning, minpruning, avgpruning;
    int worldsize, worldrank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);

    MPI_Reduce(&mypruning, &maxpruning, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mypruning, &minpruning, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mypruning, &avgpruning, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    avgpruning /= (double)worldsize;
    long maxppproc, minppproc, avgppproc;
    long myppproc = numof_query_points;
    MPI_Reduce(&myppproc, &maxppproc, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&myppproc, &minppproc, 1, MPI_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&myppproc, &avgppproc, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    avgppproc /= (long)worldsize;
	long maxrpproc, minrpproc, avgrpproc;
    long myrpproc = numof_ref_points;
    MPI_Reduce(&myrpproc, &maxrpproc, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&myrpproc, &minrpproc, 1, MPI_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&myrpproc, &avgrpproc, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    avgrpproc /= (long)worldsize;

    if(worldrank == 0) {
      cout << "Number of ref points handled per process:" << endl;
      cout << "Max: " << maxrpproc << "  Min: " << minrpproc << "  Avg: " << avgrpproc << endl;  
    }

    if(worldrank == 0) {
      cout << "Number of query points handled per process:" << endl;
      cout << "Max: " << maxppproc << "  Min: " << minppproc << "  Avg: " << avgppproc << endl;  
    }

    if(worldrank == 0) {
      cout << "Percentage of query points pruned:" << endl;
      cout << "Max: " << maxpruning << "   Min: " << minpruning << "   Avg: " << avgpruning << endl;
    }
      
    MPI_Barrier(MPI_COMM_WORLD);
  }



  knn::dist_directRQuery( ref_points, &(inData->X[0]), 
  	global_numof_ref_points, global_numof_query_points, 
    	numof_ref_points, numof_query_points, 
    	range*range, dim, glb_ref_ids, 
    	neighbors, 
    	comm);
    

  free(ref_points);
  free(glb_ref_ids);

  return;

}


/*
void queryR_Approx( pMTData inData, long rootNpoints, double dupFactor, 
	     pMTNode searchNode, 
	     double range, 
	     vector<long> *&neighbors)
{
  int size, rank;
  MPI_Comm comm = searchNode->comm;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
	
  int numof_kids = 2;
  int dim = inData->dim;
  int numof_query_points = inData->X.size() / dim;
  int global_numof_query_points = 0;
  int total_query_points;
  if(searchNode->options.pruning_verbose) {
     MPI_Allreduce( &numof_query_points, &total_query_points, 1, MPI_INT, MPI_SUM, comm );
  }


  MTData* leafData;
  pMTNode leaf;
  distributeToLeaves(inData, rootNpoints, dupFactor, searchNode, range,  &leafData, &leaf);
  inData->Copy(leafData);
  delete leafData;
  searchNode = leaf;
  int global_numof_ref_points = searchNode->Nglobal;
  comm = searchNode->comm;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  //If the "leaf" we got is not a leaf, find its leaf node.
  while( searchNode->kid )
    searchNode = searchNode->kid;


  numof_query_points = inData->X.size() / dim;


  MPI_Allreduce( &numof_query_points, &global_numof_query_points, 1, MPI_INT, MPI_SUM, comm );
  int numof_ref_points = searchNode->data->X.size() / dim;

	
  long *glb_ref_ids = (long*) malloc(numof_ref_points*sizeof(long));
  double *ref_points = (double*)malloc(numof_ref_points*dim*sizeof(double));
  #pragma omp parallel for
  for(int i = 0; i < numof_ref_points; i++) 
    glb_ref_ids[i] = searchNode->data->gids[i];

  #pragma omp parallel for
  for(int i = 0; i < numof_ref_points*dim; i++)
    ref_points[i] = searchNode->data->X[i];
	


  //Optionally print verbose pruning info.
  if(searchNode->options.pruning_verbose) {
    //double mypruning = 100.0 - (((double)global_numof_query_points)/((double)rootNpoints))*100.0;
    double mypruning = ( (double)(total_query_points - global_numof_query_points) ) 
							/ ( (double)(total_query_points - global_numof_ref_points) ) * 100.0;
    double maxpruning, minpruning, avgpruning;
    int worldsize, worldrank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);

    MPI_Reduce(&mypruning, &maxpruning, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mypruning, &minpruning, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mypruning, &avgpruning, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    avgpruning /= (double)worldsize;
    long maxppproc, minppproc, avgppproc;
    long myppproc = numof_query_points;
    MPI_Reduce(&myppproc, &maxppproc, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&myppproc, &minppproc, 1, MPI_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&myppproc, &avgppproc, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    avgppproc /= (long)worldsize;
	long maxrpproc, minrpproc, avgrpproc;
    long myrpproc = numof_ref_points;
    MPI_Reduce(&myrpproc, &maxrpproc, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&myrpproc, &minrpproc, 1, MPI_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&myrpproc, &avgrpproc, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    avgrpproc /= (long)worldsize;

    if(worldrank == 0) {
      cout << "Number of ref points handled per process:" << endl;
      cout << "Max: " << maxrpproc << "  Min: " << minrpproc << "  Avg: " << avgrpproc << endl;  
    }
    
    if(worldrank == 0) {
	  cout << "Number of query points handled per process:" << endl;
      cout << "Max: " << maxppproc << "  Min: " << minppproc << "  Avg: " << avgppproc << endl;  
    }
      
    if(worldrank == 0) {
      cout << "Percentage of query points pruned:" << endl;
      cout << "Max: " << maxpruning << "   Min: " << minpruning << "   Avg: " << avgpruning << endl;
    }
      
    MPI_Barrier(MPI_COMM_WORLD);
  }


  neighbors = new vector<long>[numof_query_points];

  knn::lsh::dist_lshRQuery
            ( ref_points, &(inData->X[0]), global_numof_ref_points, global_numof_query_points,
              numof_ref_points, numof_query_points, range,
              dim, glb_ref_ids, 4, 8, 1,
              neighbors, comm  );

    
  free(ref_points);
  free(glb_ref_ids);

  return;

}
*/


/*
void queryR_ApproxK( pMTData inData, long rootNpoints, double dupFactor,
	     pMTNode searchNode, 
	     double range, int k, int c, int max_iters,
	     vector<pair<double,long> > *&neighbors, vector<long> &queryIDs)
{
	int size, rank;
	MPI_Comm comm = searchNode->comm;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);
	int worldsize, worldrank;
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
	
	int numof_kids = 2;
	int dim = inData->dim;
	int numof_query_points = inData->X.size() / dim;
	int global_numof_query_points = 0;
	int total_query_points;
    	if(searchNode->options.pruning_verbose) {
		MPI_Allreduce( &numof_query_points, &total_query_points, 1, MPI_INT, MPI_SUM, comm );
	}



	MTData* leafData;
	pMTNode leaf;
	distributeToLeaves(inData, rootNpoints, dupFactor, searchNode, range, &leafData, &leaf);
	inData->Copy(leafData);
	delete leafData;
	searchNode = leaf;
	int global_numof_ref_points = searchNode->Nglobal;
	comm = searchNode->comm;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);
	//If the "leaf" we got is not a leaf, find its leaf node.
	while( searchNode->kid )
    		searchNode = searchNode->kid;

	numof_query_points = inData->X.size() / dim;

	MPI_Allreduce( &numof_query_points, &global_numof_query_points, 1, MPI_INT, MPI_SUM, comm );
	int numof_ref_points = searchNode->data->X.size() / dim;

	//Optionally print verbose pruning info.
	if(searchNode->options.pruning_verbose) {
		//double mypruning = 100.0 - (((double)global_numof_query_points)/((double)rootNpoints))*100.0;
		double mypruning = ( (double)(total_query_points - global_numof_query_points) ) 
								/ ( (double)(total_query_points - global_numof_ref_points) ) * 100.0;
		double maxpruning, minpruning, avgpruning;

		MPI_Reduce(&mypruning, &maxpruning, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&mypruning, &minpruning, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&mypruning, &avgpruning, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		avgpruning /= (double)worldsize;
		long maxppproc, minppproc, avgppproc;
		long myppproc = numof_query_points;
		MPI_Reduce(&myppproc, &maxppproc, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&myppproc, &minppproc, 1, MPI_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&myppproc, &avgppproc, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
		avgppproc /= (long)worldsize;
		long maxrpproc, minrpproc, avgrpproc;
		long myrpproc = numof_ref_points;
		MPI_Reduce(&myrpproc, &maxrpproc, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&myrpproc, &minrpproc, 1, MPI_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&myrpproc, &avgrpproc, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
		avgrpproc /= (long)worldsize;

	if(worldrank == 0) {
			cout << "Number of ref points handled per process:" << endl;
			cout << "Max: " << maxrpproc << "  Min: " << minrpproc << "  Avg: " << avgrpproc << endl;  
	}
      
	if(worldrank == 0) {
			cout << "Number of query points handled per process:" << endl;
			cout << "Max: " << maxppproc << "  Min: " << minppproc << "  Avg: " << avgppproc << endl;  
	}
      
		if(worldrank == 0) {
			cout << "Percentage of query points pruned:" << endl;
			cout << "Max: " << maxpruning << "   Min: " << minpruning << "   Avg: " << avgpruning << endl;
	}
      
		MPI_Barrier(MPI_COMM_WORLD);
	}



	long K, L;
	long lmax_iters = max_iters;
	knn::lsh::selectKL(global_numof_ref_points, c, K, L);
	L = std::min(lmax_iters, L);

	std::vector< std::pair<double, long> > kNN;
        int bucketFactor = (size > 100) ? 10 : 100;
	knn::lsh::distPartitionedKQuery(&(searchNode->data->X[0]), &(searchNode->data->gids[0]), &(inData->X[0]), &(inData->gids[0]), 
                global_numof_ref_points, global_numof_query_points, numof_ref_points, numof_query_points, dim, 
                range, k, K, L, bucketFactor, kNN, queryIDs, comm);

	int ppn = global_numof_query_points/size;
	int homepoints = (rank==size-1) ? ppn+global_numof_query_points%ppn : ppn; //Number of query points "owned" by each process
	assert(homepoints == kNN.size()/k);
	
	//Finally, copy valid neighbors into return vectors
	neighbors = new vector<pair<double, long> >[homepoints];
	for(int i = 0; i < homepoints; i++) {
		for(int j = 0; j < k; j++) {
			if(kNN[i*k+j].second != -1L) neighbors[i].push_back(kNN[i*k+j]);
			else break; //No more valid neighbors
		}
	}

	return;
	

}
*/



void queryR( pMTData inData, long rootNpoints, double dupFactor,
            pMTNode searchNode, 
            vector< pair<double, long> > *&neighbors, int *nvectors)
{
  int size, rank;
	
  MTData* leafData;
  pMTNode leaf;
 
  int total_query_points; 
  int dim = inData->dim;
  int numof_query_points = inData->X.size() / dim;
  assert( inData->radii.size() == numof_query_points );

  if(searchNode->options.pruning_verbose) {
     MPI_Allreduce( &numof_query_points, &total_query_points, 1, MPI_INT, MPI_SUM, 
                    searchNode->comm );
   }


  distributeToLeaves(inData, rootNpoints, dupFactor, searchNode, &leafData, &leaf);
  inData->Copy(leafData);
  delete leafData;
  searchNode = leaf;
  int global_numof_ref_points = searchNode->Nglobal;
  MPI_Comm comm = searchNode->comm;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  //If the "leaf" we got is not a leaf, find its leaf node.
  while( searchNode->kid )
    searchNode = searchNode->kid;



  numof_query_points = inData->X.size() / dim;
  int global_numof_query_points = 0;
  MPI_Allreduce( &numof_query_points, &global_numof_query_points, 1, MPI_INT, MPI_SUM, comm );
  int numof_ref_points = searchNode->data->X.size() / dim;
  
  
  long *glb_ref_ids = (long*)malloc(numof_ref_points*sizeof(long));
  double *ref_points = (double*)malloc(numof_ref_points*dim*sizeof(double));
  #pragma omp parallel for
  for(int i = 0; i < numof_ref_points; i++) 
  	glb_ref_ids[i] = searchNode->data->gids[i];
  
  #pragma omp parallel for
  for(int i = 0; i < numof_ref_points*dim; i++)
  	ref_points[i] = searchNode->data->X[i];
  
  neighbors = new vector< pair<double, long> >[numof_query_points];
  *nvectors = numof_query_points;
  assert( inData->radii.size() == numof_query_points );


  //Optionally print verbose pruning info.
  if(searchNode->options.pruning_verbose) {
    //double mypruning = 100.0 - (((double)global_numof_query_points)/((double)rootNpoints))*100.0;
    double mypruning = ( (double)(total_query_points - global_numof_query_points) ) 
							/ ( (double)(total_query_points - global_numof_ref_points) ) * 100.0;
	double maxpruning, minpruning, avgpruning;
    int worldsize, worldrank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);

    MPI_Reduce(&mypruning, &maxpruning, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mypruning, &minpruning, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mypruning, &avgpruning, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    avgpruning /= (double)worldsize;
    long maxppproc, minppproc, avgppproc;
    long myppproc = numof_query_points;
    MPI_Reduce(&myppproc, &maxppproc, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&myppproc, &minppproc, 1, MPI_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&myppproc, &avgppproc, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    avgppproc /= (long)worldsize;
	long maxrpproc, minrpproc, avgrpproc;
	long myrpproc = numof_ref_points;
	MPI_Reduce(&myrpproc, &maxrpproc, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&myrpproc, &minrpproc, 1, MPI_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&myrpproc, &avgrpproc, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	avgrpproc /= (long)worldsize;

	if(worldrank == 0) {
      cout << "Number of ref points handled per process:" << endl;
      cout << "Max: " << maxrpproc << "  Min: " << minrpproc << "  Avg: " << avgrpproc << endl;  
    }

	if(worldrank == 0) {
	  cout << "Number of query points handled per process:" << endl;
      cout << "Max: " << maxppproc << "  Min: " << minppproc << "  Avg: " << avgppproc << endl;  
    }

    if(worldrank == 0) {
      cout << "Percentage of query points pruned:" << endl;
      cout << "Max: " << maxpruning << "  Min: " << minpruning << "  Avg: " << avgpruning << endl;  
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }



  knn::dist_directRQueryIndividual( ref_points, &(inData->X[0]), 
                         global_numof_ref_points, global_numof_query_points, 
                         numof_ref_points, numof_query_points, 
                         &(inData->radii[0]), dim, glb_ref_ids, 
                         neighbors, 
                         comm);




  free(ref_points);
  free(glb_ref_ids);
  
  
  return;
  
}



void queryRK( pMTData inData, long rootNpoints, double dupFactor, int k,
            pMTNode searchNode, 
            vector< pair<double, long> > *&neighbors, int *nvectors)
{

  double start_t = 0.0;

  int size, rank;
  int total_query_points; 
  int dim = inData->dim;
  int numof_query_points = inData->X.size() / dim;
  assert( inData->radii.size() == numof_query_points );

  if(searchNode->options.pruning_verbose) {
     MPI_Allreduce( &numof_query_points, &total_query_points, 1, MPI_INT, MPI_SUM, 
                    searchNode->comm );
   }


  MTData* leafData;
  pMTNode leaf;
  distributeToLeaves(inData, rootNpoints, dupFactor, searchNode, &leafData, &leaf);
  inData->Copy(leafData);
  delete leafData;
  searchNode = leaf;
  int global_numof_ref_points = searchNode->Nglobal;
  MPI_Comm comm = searchNode->comm;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  //If the "leaf" we got is not a leaf, find its leaf node.
  while( searchNode->kid )
    searchNode = searchNode->kid;
  
  numof_query_points = inData->X.size() / dim;
  int global_numof_query_points = 0;
  MPI_Allreduce( &numof_query_points, &global_numof_query_points, 1, MPI_INT, MPI_SUM, comm );
  int numof_ref_points = searchNode->data->X.size() / dim;
  
  
  long *glb_ref_ids = (long*)malloc(numof_ref_points*sizeof(long));
  double *ref_points = (double*)malloc(numof_ref_points*dim*sizeof(double));
  #pragma omp parallel for
  for(int i = 0; i < numof_ref_points; i++) 
  	glb_ref_ids[i] = searchNode->data->gids[i];
  
  #pragma omp parallel for
  for(int i = 0; i < numof_ref_points*dim; i++)
  	ref_points[i] = searchNode->data->X[i];
  
  neighbors = new vector< pair<double, long> >[numof_query_points];
  *nvectors = numof_query_points;
  assert( inData->radii.size() == numof_query_points );

  
  //Optionally print verbose pruning info.
  if(searchNode->options.pruning_verbose) {
    int worldsize, worldrank;
    MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);

	//double mypruning = 100.0 - (((double)global_numof_query_points)/((double)rootNpoints))*100.0;
    int avg_nq = total_query_points / worldsize;
	double mypruning = ( (double)(total_query_points - global_numof_query_points) ) 
							/ ( (double)(total_query_points - avg_nq) ) * 100.0;
	
	double maxpruning, minpruning, avgpruning;
    MPI_Reduce(&mypruning, &maxpruning, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mypruning, &minpruning, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mypruning, &avgpruning, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    avgpruning /= (double)worldsize;
    long maxppproc, minppproc, avgppproc;
    long myppproc = numof_query_points;
    MPI_Reduce(&myppproc, &maxppproc, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&myppproc, &minppproc, 1, MPI_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&myppproc, &avgppproc, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    avgppproc /= (long)worldsize;
	long maxrpproc, minrpproc, avgrpproc;
	long myrpproc = numof_ref_points;
	MPI_Reduce(&myrpproc, &maxrpproc, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&myrpproc, &minrpproc, 1, MPI_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&myrpproc, &avgrpproc, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	avgrpproc /= (long)worldsize;
 
	long *nps = new long [worldsize];
	long *nrs = new long [worldsize];
	MPI_Gather(&myrpproc, 1, MPI_LONG, nrs, 1, MPI_LONG, 0, MPI_COMM_WORLD);
	MPI_Gather(&myppproc, 1, MPI_LONG, nps, 1, MPI_LONG, 0, MPI_COMM_WORLD);

	if(worldrank == 0) {
      
	  cout<<"Number of ref points per process: ";
	  for(int i = 0; i < worldsize; i++) 
		  cout<<nrs[i]<<" ";
	  cout<<endl;
	 
	  cout<<"Number of query points per process: ";
	  for(int i = 0; i < worldsize; i++) 
		  cout<<nps[i]<<" ";
	  cout<<endl;
		
	  cout << "Number of ref points handled per process:" << endl;
      cout << "Max: " << maxrpproc << "  Min: " << minrpproc << "  Avg: " << avgrpproc << endl;  
      
	  cout << "Number of query points handled per process:" << endl;
      cout << "Max: " << maxppproc << "  Min: " << minppproc << "  Avg: " << avgppproc << endl;  
      
	  cout << "Percentage of query points pruned:" << endl;
      cout << "Max: " << maxpruning << "  Min: " << minpruning << "  Avg: " << avgpruning << endl;  
		

	}

	delete [] nps;
	delete [] nrs;

    MPI_Barrier(MPI_COMM_WORLD);

  }

  start_t = omp_get_wtime();
  knn::dist_directRQueryIndividualK( ref_points, &(inData->X[0]), 
                         global_numof_ref_points, global_numof_query_points, 
                         numof_ref_points, numof_query_points, 
                         &(inData->radii[0]), k, dim, glb_ref_ids, 
                         neighbors, 
                         comm);

  Direct_Kernel_T_ += omp_get_wtime() - start_t;

  free(ref_points);
  free(glb_ref_ids);
  
  return;
}

