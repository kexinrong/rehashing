#include <mpi.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <utility>
#include <map>
#include <cfloat>

#include "ompUtils.h"
#include "repartition.h"
#include "clustering.h"
#include "direct_knn.h"
#include "mpitree.h"
#include "lsh.h"
#include "verbose.h"

using namespace std;
using namespace knn;
using namespace knn::repartition;




/**
 * Performs first pass to select initial guesses for R for k-query.  For internal use only.
 * \param inData Query points.
 * \param searchNode Root of current search.
 * \param k Number of near neighbors to find.
 * \param R [out] Initial guesses for search radii (k nearest neighbors are /at most/ this far away).
 */
void queryKSelectRs(pMTData redistQuery, 
                    pMTData homeData,
                    pMTNode searchNode, 
                    int global_numof_query_points,
                    int k,
                    double **R) {
  
  int size, rank, worldsize, worldrank;
  MPI_Comm comm = searchNode->comm;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
  MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
  int ppn = global_numof_query_points/worldsize; //Number of points per process
	
  int dim = redistQuery->dim;
  int numof_query_points = redistQuery->X.size() / dim;
  int numof_home_points = homeData->X.size() / dim;
    
  int numof_ref_points = searchNode->data->X.size() / dim;
  long *glb_ref_ids = (long*)malloc(numof_ref_points*sizeof(long));
  double *ref_points = (double*)malloc(numof_ref_points*dim*sizeof(double));
  long *redistIDs = &(redistQuery->gids[0]); //For debugging
  long *homeIDs = &(homeData->gids[0]); //For debugging
 
  #pragma omp parallel for
  for(int i = 0; i < numof_ref_points; i++) 
    glb_ref_ids[i] = searchNode->data->gids[i];
  
  #pragma omp parallel for
  for(int i = 0; i < numof_ref_points*dim; i++)
    ref_points[i] = searchNode->data->X[i];

  //Find k nearest neighbors within this node.
  pair<double, long> *kneighbors = dist_directKQuery( ref_points, &(redistQuery->X[0]), 
                                              glb_ref_ids,
                                              numof_ref_points, numof_query_points,
                                              k, dim, comm );
  
  pair<long, double> *kthdist = new pair<long, double>[numof_query_points];
  
  #pragma omp parallel for
  for(int i = 0; i < numof_query_points; i++) {
    kthdist[i].first = redistQuery->gids[i];
    kthdist[i].second = kneighbors[(i+1)*k-1].first;
  }
  delete[] kneighbors;
  free(ref_points);
  free(glb_ref_ids);
  
  
  //Sort in order of ascending query point ID.
  omp_par::merge_sort(&(kthdist[0]), &(kthdist[numof_query_points]));
  
  //Collect search radii at "home" process of each query point (rank query_id/worldsize in MPI_COMM_WORLD).
  int *sendcounts = new int[worldsize];
  int *rcvcounts = new int[worldsize];
  int *senddisp = new int[worldsize];
  int *rcvdisp = new int[worldsize];
  #pragma omp parallel for
  for(int i = 0; i < worldsize; i++) sendcounts[i] = 0;
  for(int i = 0; i < numof_query_points; i++) sendcounts[ knn::lsh::idToHomeRank(kthdist[i].first, ppn, worldsize) ]++;
  MPI_Alltoall(sendcounts, 1, MPI_INT, rcvcounts, 1, MPI_INT, MPI_COMM_WORLD);
  omp_par::scan(sendcounts, senddisp, worldsize);
  omp_par::scan(rcvcounts, rcvdisp, worldsize);
  int rcvdists = rcvdisp[worldsize-1]+rcvcounts[worldsize-1];
  pair<long, double> *homedistances = new pair<long, double>[rcvdists];
  MPI_Datatype pairdata;
  MPI_Type_contiguous(sizeof(pair<long, double>), MPI_BYTE, &pairdata);
  MPI_Type_commit(&pairdata);
  MPI_Alltoallv(kthdist, sendcounts, senddisp, pairdata, homedistances,
                rcvcounts, rcvdisp, pairdata, MPI_COMM_WORLD);
  
  
  //Sort k-distances by query ID
  omp_par::merge_sort(homedistances, &(homedistances[rcvdists]));
  
  //Find minimum radius for each query ID (first with each ID)
  map<long, double> minDists;
  long currID = -1;
  for(int i = 0; i < rcvdists; i++) {
    if(homedistances[i].first != currID) {
      currID = homedistances[i].first;
      minDists[currID] = homedistances[i].second;
    }
  }
  
  *R = new double[numof_home_points];
  #pragma omp parallel for
  for(int i = 0; i < numof_home_points; i++)  {
    map<long,double>::iterator it = minDists.find(homeData->gids[i]);

    (*R)[i] = (*it).second + (*it).second*1.0e-9;
  }
  
  delete[] homedistances;
  delete[] kthdist;
  delete[] sendcounts;
  delete[] rcvcounts;
  delete[] senddisp;   
  delete[] rcvdisp;
    
  return;  
}




void queryK( pMTData inData, long rootNpoints, double dupFactor,
                pMTNode searchNode, 
                int k, vector<long> *queryIDs,
                vector< pair<double, long> > *kNN){
 
  double start_t;
 
  int worldsize, worldrank;
  MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
  MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
  int dim = inData->dim;
  int numof_query_points = inData->X.size() / dim;
  int global_numof_query_points;
  MPI_Allreduce( &numof_query_points, &global_numof_query_points, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
  int ppn = global_numof_query_points/worldsize; //Number of points per node

  //First, relocate query points to their home processes 
    //This is also where we will collect the initial search radii and the query results.  
  long *gid_copy = new long[numof_query_points];
  double *data_copy = new double[numof_query_points*dim];
  int *homeproc = new int[numof_query_points]; 
  int *sendcounts = new int[worldsize];
  #pragma omp parallel for
  for(int i = 0; i < worldsize; i++) sendcounts[i] = 0;
  double *new_data;
  long new_n, *new_ids;
  #pragma omp parallel for
  for(int i = 0; i < numof_query_points; i++)  {
    homeproc[i] = knn::lsh::idToHomeRank(inData->gids[i], ppn, worldsize);
    gid_copy[i] = inData->gids[i];
  }
  #pragma omp parallel for
  for(int i = 0; i < numof_query_points*dim; i++)  data_copy[i] = inData->X[i];
  for(int i = 0; i < numof_query_points; i++)
     sendcounts[homeproc[i]]++;

  local_rearrange(&gid_copy, &homeproc, &data_copy, numof_query_points, dim);
  
  MPI_Barrier(MPI_COMM_WORLD);
  start_t = omp_get_wtime();
  knn::repartition::repartition(gid_copy, data_copy, numof_query_points, sendcounts, dim,
              &new_ids, &new_data, &new_n, MPI_COMM_WORLD);
  Repartition_Query_T_ += omp_get_wtime() - start_t;

  delete[] gid_copy;
  delete[] data_copy;
  delete[] homeproc;
  
  inData->X.resize(new_n*dim);
  inData->gids.resize(new_n);
  #pragma omp parallel for
  for(int i = 0; i < new_n; i++)  {
    inData->gids[i] = new_ids[i];
  }
  #pragma omp parallel for
  for(int i = 0; i < new_n*dim; i++)  inData->X[i] = new_data[i];
  delete[] new_ids;
  delete[] new_data;
  numof_query_points = inData->X.size() / dim;
  
  //Create a copy of the query set.
  pMTData querycopy = new MTData();
  querycopy->Copy(inData);
  
  //Initial pass to determine search radii.
  double *R;
  pMTData redistQuery;
  pMTNode leaf;

  distributeToNearestLeaf(querycopy, searchNode, &redistQuery, &leaf);

  queryKSelectRs(redistQuery, inData, leaf, global_numof_query_points, k, &R);
  
  delete redistQuery;


  //Perform r-query with individual radii
  inData->radii.resize(numof_query_points);
  #pragma omp parallel for
  for(int i = 0; i < numof_query_points; i++)
    inData->radii[i] = R[i];

  delete[] R;
  vector< pair<double, long> > *rneighbors;
  querycopy->Copy(inData);
  int nvectors;
  
  queryRK(querycopy, rootNpoints, dupFactor, k, searchNode, rneighbors, &nvectors);

  //Pack neighbors into array of (queryID, distance, refID) triples. 
  int *neighcount = new int[nvectors];
  int *neighcountscan = new int[nvectors];
  #pragma omp parallel for
  for(int i = 0; i < nvectors; i++)
    neighcount[i] = rneighbors[i].size();
  omp_par::scan(neighcount, neighcountscan, nvectors);
  int totalneighbors = neighcountscan[nvectors-1] + neighcount[nvectors-1];
  triple<long, double, long> *tNeighbors = new triple<long, double, long>[totalneighbors];
  #pragma omp parallel for
  for(int i = 0; i < nvectors; i++) {
    for(int j = 0; j < neighcount[i]; j++) {
      triple<long, double, long> *currNeigh;
      currNeigh = &(tNeighbors[neighcountscan[i]+j]);
      currNeigh->first = querycopy->gids[i];
      currNeigh->second = rneighbors[i][j].first;
      currNeigh->third = rneighbors[i][j].second;
    }
  }
  delete[] rneighbors;
  delete[] neighcount;
  delete[] neighcountscan;
  delete querycopy;
  
  //Sort array of triples and transimit to appropriate home processes.
  omp_par::merge_sort(tNeighbors, &(tNeighbors[totalneighbors]));
  int *rcvcounts = new int[worldsize];
  int *senddisp = new int[worldsize];
  int *rcvdisp = new int[worldsize];
  #pragma omp parallel for
  for(int i = 0; i < worldsize; i++) sendcounts[i] = 0;
  for(int i = 0; i < totalneighbors; i++) sendcounts[ knn::lsh::idToHomeRank(tNeighbors[i].first, ppn, worldsize) ]++;
  MPI_Alltoall(sendcounts, 1, MPI_INT, rcvcounts, 1, MPI_INT, MPI_COMM_WORLD);
  omp_par::scan(sendcounts, senddisp, worldsize);
  omp_par::scan(rcvcounts, rcvdisp, worldsize);
  int rcvneighs = rcvdisp[worldsize-1]+rcvcounts[worldsize-1];
  assert(rcvneighs >= numof_query_points*k);

  MPI_Barrier(MPI_COMM_WORLD);
  start_t = omp_get_wtime();
  triple<long, double, long> *homeneighbors = new triple<long, double, long>[rcvneighs];
  MPI_Datatype tripledata;
  MPI_Type_contiguous(sizeof(triple<long, double, long>), MPI_BYTE, &tripledata);
  MPI_Type_commit(&tripledata);
  MPI_Alltoallv(tNeighbors, sendcounts, senddisp, tripledata, homeneighbors,
                rcvcounts, rcvdisp, tripledata, MPI_COMM_WORLD);
  Repartition_Query_T_ += omp_get_wtime() - start_t;

  delete[] tNeighbors;
  
  //Determine starting location of each query point's neighbors
  omp_par::merge_sort(homeneighbors, &(homeneighbors[rcvneighs]));
  int *offsets = new int[numof_query_points];
  long currID=homeneighbors[0].first;
  int currIDX=1;
  offsets[0]=0;
  for(int i = 0; i < rcvneighs; i++) {
    if(homeneighbors[i].first != currID){
      offsets[currIDX++]=i;
      currID = homeneighbors[i].first;
    }
  }


  //Store results in output vectors.
  queryIDs->resize(numof_query_points);

  kNN->resize(k*numof_query_points) ;
  #pragma omp parallel for
  for(int i = 0; i < numof_query_points; i++) {
    double min = DBL_MAX;
    long min_idx = -1;
    double last_min = -1.0;
    long last_min_idx = -1;

    (*queryIDs)[i]=homeneighbors[offsets[i]].first;
    int curroffset=offsets[i];

    for( int j = 0; j < k; j ++ ) {
      long currID = homeneighbors[offsets[i]].first;
      int l=0;

      while((curroffset+l < rcvneighs) && (homeneighbors[curroffset+l].first == currID))  { //Loop over all neighbors for the same query point

        if( homeneighbors[curroffset+l].second < min 
            && homeneighbors[curroffset+l].third != last_min_idx 
            &&  homeneighbors[curroffset+l].second >= last_min ) {
          min = homeneighbors[curroffset+l].second;
          min_idx = homeneighbors[curroffset+l].third;
        }
        l++;
      }
      (*kNN)[i*k+j].first = min;
      (*kNN)[i*k+j].second = min_idx;
      last_min = min;
      last_min_idx = min_idx;
      min = DBL_MAX;
    }
  }


  delete[] homeneighbors;  
  delete[] sendcounts;
  delete[] rcvcounts;
  delete[] senddisp;   
  delete[] rcvdisp;
  delete[] offsets;

}


/*
void queryK_Approx( pMTData inData, long rootNpoints, double dupFactor,
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
	MPI_Allreduce( &numof_query_points, &total_query_points, 1, MPI_INT, MPI_SUM, comm );


	vector< pair<double,long> > *rkNN;
	vector<long> rkIDs;
	queryR_ApproxK( inData, rootNpoints, dupFactor, searchNode, range, k, c, max_iters, rkNN, rkIDs);

	
	//Transfer all neighbors for each query point to its "home" process, then sort and keep the k nearest.
	int mLocal = rkIDs.size();
	MPI_Datatype tripledata;
	MPI_Type_contiguous(sizeof(triple<long, double, long>), MPI_BYTE, &tripledata);
	MPI_Type_commit(&tripledata);
	triple<long, double, long> *localResults = new triple<long, double, long>[mLocal*k];

	#pragma omp parallel for
	for(int i = 0; i < mLocal; i++) {
		for(int j = 0; j < k; j++) {
			if( j < rkNN[i].size() ) {
				localResults[i*k+j].first = rkIDs[i];
				localResults[i*k+j].second = rkNN[i][j].first;
				localResults[i*k+j].third = rkNN[i][j].second;
			}  else {
				localResults[i*k+j].first = rkIDs[i];
				localResults[i*k+j].second = DBL_MAX;
				localResults[i*k+j].third = -1L;
			}
		}
	}
	delete[] rkNN;
	omp_par::merge_sort(localResults, &(localResults[mLocal*k]));

	int ppn = total_query_points/worldsize;
 	int homepoints = (worldrank==worldsize-1) ? ppn+total_query_points%ppn : ppn; //Number of global query points "owned" by each process

	int *sendcounts = new int[worldsize];
	int *senddisp = new int[worldsize];
	int *recvcounts = new int[worldsize];
	int *recvdisp = new int[worldsize];
	#pragma omp parallel for
        for(int i = 0; i < worldsize; i++) sendcounts[i] = 0;
        for(int i = 0; i < mLocal; i++) {
		sendcounts[ knn::lsh::idToHomeRank(rkIDs[i], ppn, size) ] += k;
	}
	MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
        omp_par::scan(sendcounts, senddisp, worldsize);
        omp_par::scan(recvcounts, recvdisp, worldsize);
	int totalrecv = recvcounts[worldsize-1] + recvdisp[worldsize-1];
	triple<long, double, long> *homeResults = new triple<long, double, long>[totalrecv];

	MPI_Alltoallv(localResults, sendcounts, senddisp, tripledata, homeResults, recvcounts, recvdisp, tripledata, MPI_COMM_WORLD);
	std::sort(homeResults, &(homeResults[totalrecv]));
	
	//Find where each point's neighbors begin. 
	int *qpOffsets = new int[homepoints]; //Array index where each query point's neighbors begin.
	qpOffsets[0] = 0;
	long currID = -1L; 
	int currIDX = 0;	
	for(int i = 0; i < totalrecv; i+=k) {
		if(homeResults[i].first != currID) {
			qpOffsets[currIDX++] = i;
			currID = homeResults[i].first;
		}
	}
	assert(currIDX == homepoints);

 	queryIDs.resize(homepoints);	
	//Finally, copy valid neighbors into return vectors
	neighbors = new vector<pair<double, long> >[homepoints];
	for(int i = 0; i < homepoints; i++) {
		queryIDs[i] = homeResults[qpOffsets[i]].first;
		for(int j = 0; j < k; j++) {
			if(homeResults[qpOffsets[i]+j].third != -1L) {
				neighbors[i].push_back(
					make_pair<double, long>(homeResults[qpOffsets[i]+j].second, 
					homeResults[qpOffsets[i]+j].third));
			} else {
				break;
			}
		}
	}



	MPI_Type_free(&tripledata);
	delete[] qpOffsets;
	delete[] localResults;
	delete[] homeResults;
	delete[] sendcounts;
	delete[] senddisp;
	delete[] recvcounts;
	delete[] recvdisp;

	return;
	

}
*/





void queryK_GreedyApprox( pMTData inData, long rootNpoints, 
                pMTNode searchNode, 
                int k, vector<long> *queryIDs,
                vector< pair<double, long> > *kNN){
  
  int worldsize, worldrank;
  MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
  MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
  int dim = inData->dim;
  int numof_query_points = inData->X.size() / dim;
  int global_numof_query_points;
  MPI_Allreduce( &numof_query_points, &global_numof_query_points, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
  int ppn = global_numof_query_points/worldsize; //Number of points per node
  int homepoints = (worldrank==worldsize-1) ? ppn+global_numof_query_points%ppn : ppn; //Number of query points "owned" by each process

  
  //Create a copy of the query set.
  pMTData querycopy = new MTData();
  querycopy->Copy(inData);
  

  //Initial pass to determine search radii.
  pMTData redistQuery;
  pMTNode leaf;
  distributeToNearestLeaf(querycopy, searchNode, &redistQuery, &leaf);
  delete querycopy;


  numof_query_points = redistQuery->X.size() / dim;
    
  int numof_ref_points = leaf->data->X.size() / dim;
  long *glb_ref_ids = (long*)malloc(numof_ref_points*sizeof(long));
  double *ref_points = (double*)malloc(numof_ref_points*dim*sizeof(double));
 
  #pragma omp parallel for
  for(int i = 0; i < numof_ref_points; i++) 
    glb_ref_ids[i] = leaf->data->gids[i];
  
  #pragma omp parallel for
  for(int i = 0; i < numof_ref_points*dim; i++)
    ref_points[i] = leaf->data->X[i];

  //Find k nearest neighbors within this node.
  pair<double, long> *kneighbors = dist_directKQuery( ref_points, &(redistQuery->X[0]), 
                                              glb_ref_ids,
                                              numof_ref_points, numof_query_points,
                                              k, dim, leaf->comm );
  
  free(ref_points);
  free(glb_ref_ids);
  
  //Pack neighbors into array of (queryID, distance, refID) triples. 
  int totalneighbors = k*numof_query_points;
  triple<long, double, long> *tNeighbors = new triple<long, double, long>[totalneighbors];
  #pragma omp parallel for
  for(int i = 0; i < numof_query_points; i++) {
    for(int j = 0; j < k; j++) {
      triple<long, double, long> *currNeigh;
      currNeigh = &(tNeighbors[i*k+j]);
      currNeigh->first = redistQuery->gids[i];
      currNeigh->second = kneighbors[i*k+j].first;
      currNeigh->third = kneighbors[i*k+j].second;
    }
  }
  
  //Sort array of triples and transimit to appropriate home processes.
  omp_par::merge_sort(tNeighbors, &(tNeighbors[totalneighbors]));
  int *sendcounts = new int[worldsize];
  int *rcvcounts = new int[worldsize];
  int *senddisp = new int[worldsize];
  int *rcvdisp = new int[worldsize];
  #pragma omp parallel for
  for(int i = 0; i < worldsize; i++) sendcounts[i] = 0;
  for(int i = 0; i < totalneighbors; i++) sendcounts[ knn::lsh::idToHomeRank(tNeighbors[i].first, ppn, worldsize) ]++;
  MPI_Alltoall(sendcounts, 1, MPI_INT, rcvcounts, 1, MPI_INT, MPI_COMM_WORLD);
  omp_par::scan(sendcounts, senddisp, worldsize);
  omp_par::scan(rcvcounts, rcvdisp, worldsize);
  int rcvneighs = rcvdisp[worldsize-1]+rcvcounts[worldsize-1];
  assert(rcvneighs == homepoints*k);
  triple<long, double, long> *homeneighbors = new triple<long, double, long>[rcvneighs];
  MPI_Datatype tripledata;
  MPI_Type_contiguous(sizeof(triple<long, double, long>), MPI_BYTE, &tripledata);
  MPI_Type_commit(&tripledata);
  MPI_Alltoallv(tNeighbors, sendcounts, senddisp, tripledata, homeneighbors,
                rcvcounts, rcvdisp, tripledata, MPI_COMM_WORLD);
  delete[] tNeighbors;
  omp_par::merge_sort(homeneighbors, &(homeneighbors[rcvneighs]));
  
  //Store results in output vectors.
  queryIDs->resize(homepoints);

  kNN->resize(k*homepoints) ;
  #pragma omp parallel for
  for(int i = 0; i < homepoints; i++) {
    (*queryIDs)[i]=homeneighbors[i*k].first;
    for( int j = 0; j < k; j ++ ) {
      (*kNN)[i*k+j].first = homeneighbors[i*k+j].second;
      (*kNN)[i*k+j].second = homeneighbors[i*k+j].third;
    }
  }


  delete redistQuery;
  delete[] homeneighbors;  
  delete[] sendcounts;
  delete[] rcvcounts;
  delete[] senddisp;   
  delete[] rcvdisp;

}


