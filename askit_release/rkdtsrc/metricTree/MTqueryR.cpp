#include <stdio.h>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <omp.h>
#include "repartition.h"
#include "clustering.h"
#include "direct_knn.h"
#include "mpitree.h"
#include "lsh.h"

#include "metrictree.h"

using namespace std;
using namespace knn;
using namespace knn::repartition;


void MTqueryR( pMetricData inData, long rootNpoints, double dupFactor,
	     pMetricNode searchNode, 
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


  MetricData* leafData;
  pMetricNode leaf;
  MTdistributeToLeaves(inData, rootNpoints, dupFactor, searchNode, range,  &leafData, &leaf);
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

  MPI_Allreduce( &numof_query_points, &global_numof_query_points, 
				 1, MPI_INT, MPI_SUM, comm );
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
							/ ( (double)(total_query_points - global_numof_ref_points) ) * 100;
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
						  global_numof_ref_points, 
						  global_numof_query_points, 
						  numof_ref_points, 
						  numof_query_points, 
						  range*range, dim, glb_ref_ids, 
						  neighbors, comm);
    

  free(ref_points);
  free(glb_ref_ids);

  return;

}



void MTqueryR(	pMetricData inData, long rootNpoints, 
				double dupFactor,
				pMetricNode searchNode, 
				vector< pair<double, long> > *&neighbors, 
				int *nvectors)
{
  int size, rank;
	
  MetricData* leafData;
  pMetricNode leaf;
 
  int total_query_points; 
  int dim = inData->dim;
  int numof_query_points = inData->X.size() / dim;
  assert( inData->radii.size() == numof_query_points );

  if(searchNode->options.pruning_verbose) {
     MPI_Allreduce( &numof_query_points, &total_query_points, 1, MPI_INT, MPI_SUM, 
                    searchNode->comm );
   }


  MTdistributeToLeaves(inData, rootNpoints, dupFactor, searchNode, &leafData, &leaf);
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
							/ ( (double)(total_query_points - global_numof_ref_points) );
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



void MTqueryRK( pMetricData inData, long rootNpoints, 
				double dupFactor, int k,
				pMetricNode searchNode, 
				vector< pair<double, long> > *&neighbors, 
				int *nvectors)
{
  int size, rank;

  int total_query_points; 
  int dim = inData->dim;
  int numof_query_points = inData->X.size() / dim;
  assert( inData->radii.size() == numof_query_points );

  if(searchNode->options.pruning_verbose) {
     MPI_Allreduce( &numof_query_points, &total_query_points, 
					1, MPI_INT, MPI_SUM, searchNode->comm );
   }

  double start_t;
  double stage_go2leaf_t, stage_datacopy_t, stage_nn_t;
  int timing_verbose = searchNode->options.timing_verbose;

  if(timing_verbose) { start_t = MPI_Wtime();}
  MetricData* leafData;
  pMetricNode leaf;
  MTdistributeToLeaves(inData, rootNpoints, dupFactor, 
						searchNode, &leafData, &leaf);
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


  if(timing_verbose) {stage_go2leaf_t = MPI_Wtime() - start_t;}
  


  if(timing_verbose) start_t = MPI_Wtime();
  numof_query_points = inData->X.size() / dim;
  int global_numof_query_points = 0;
  MPI_Allreduce( &numof_query_points, &global_numof_query_points, 
					1, MPI_INT, MPI_SUM, comm );
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

  if(timing_verbose) { stage_datacopy_t = MPI_Wtime() - start_t; }
  
  //Optionally print verbose pruning info.
  if(searchNode->options.pruning_verbose) {
    //double mypruning = 100.0 - (((double)global_numof_query_points)/((double)rootNpoints))*100.0;
    double mypruning = ((double)(total_query_points - global_numof_query_points))
								/ ((double)(total_query_points - global_numof_ref_points)) * 100.0;
	  
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


  if(timing_verbose) {start_t = MPI_Wtime(); }
  knn::dist_directRQueryIndividualK( ref_points, &(inData->X[0]), 
                         global_numof_ref_points, 
						 global_numof_query_points, 
                         numof_ref_points, numof_query_points, 
                         &(inData->radii[0]), k, dim, glb_ref_ids, 
                         neighbors, 
                         comm);

  if(timing_verbose) {stage_nn_t = MPI_Wtime() - start_t; }

  /*
  if(timing_verbose) {
    	int worldsize, worldrank;
    	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
    	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);

 	MPI_Barrier(MPI_COMM_WORLD);
	if(worldrank == 0) {
		//cout<<"+++++++++++++++++++++++++++++++++++++++"<<endl
	        cout <<"rank - "<<" nn search time"<<endl;
		//cout.flush();
		fflush(stdout);
		fsync(STDOUT_FILENO);
	}
  	MPI_Barrier(MPI_COMM_WORLD);
	for(int t = 0; t < worldsize; t++) {
		if(worldrank == t) {
			cout<<worldrank<<" - "
		               <<stage_go2leaf_t+stage_datacopy_t+stage_nn_t
			       <<endl;
			//cout.flush();
			fflush(stdout);
			fsync(STDOUT_FILENO);
		}
  		MPI_Barrier(MPI_COMM_WORLD);
	}
  	MPI_Barrier(MPI_COMM_WORLD);
  }
   */

  if(timing_verbose) {
    	int worldsize, worldrank;
    	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
    	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
	
	char ptrOutputName[256] = {0};
	sprintf(ptrOutputName, "proc%05d_dim%03d_rank%05d_r%02d.info", 
			worldsize, dim, worldrank, timing_verbose);
	ofstream fout(ptrOutputName, ios::app|ios::out);
	fout<<"queryRK: worldrank  nn_t  go2leaf_t  datacopy_t"<<endl;
	fout<<worldrank<<" "
		<<stage_nn_t<<" "
		<<stage_go2leaf_t<<" "
		<<stage_datacopy_t
		<<endl;
	fout<<"queryRK_problem_size:(#ref,#query) "<<numof_ref_points<<" "<<numof_query_points<<endl;
  	fout.flush();
	fout.close();
	MPI_Barrier(MPI_COMM_WORLD);
  }


  free(ref_points);
  free(glb_ref_ids);
  
  return;
}




