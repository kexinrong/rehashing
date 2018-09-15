#include <mpi.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <float.h>
#include <map>
#include "repartition.h"
#include "clustering.h"
#include "direct_knn.h"
#include "lsh.h"
#include "distributeToLeaf.h"
#include "binTree.h"
#include "binQuery.h"
#include "generator.h"
#include "stTree.h"
#include "stTreeSearch.h"
#include "verbose.h"
#include "eval.h"
#include "gatherTree.h"
#include "parallelIO.h"

using namespace std;
using namespace knn;
using namespace knn::repartition;

#define _RKDT_DEBUG_ false
#define _COLLECT_COORD_DEBUG_ false
#define _RKDT_ALL2ALL_DEBUG_ false
#define _KNN_MERGE_DEBUG_ false

// ============= the following functions are used for exact search =============

/**
 * Performs first pass to select initial guesses for R for k-query.  For internal use only.
 * \param inData Query points.
 * \param searchNode Root of current search.
 * \param k Number of near neighbors to find.
 * \param R [out] Initial guesses for search radii (k nearest neighbors are /at most/ this far away).
 */
void bintree::queryKSelectRs( pbinData redistQuery, pbinData homeData,
					   pbinNode searchNode, int global_numof_query_points,
					   int k, double **R)
{

  double start_t;

  int size, rank, worldsize, worldrank;
  MPI_Comm comm = searchNode->comm;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
  MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
  int ppn = global_numof_query_points/worldsize; //Number of points per process

  int dim = redistQuery->dim;
  int numof_query_points = redistQuery->numof_points;
  int numof_home_points = homeData->numof_points;
  int numof_ref_points = searchNode->data->numof_points;

  long *glb_ref_ids = (long*)malloc(numof_ref_points*sizeof(long));
  double *ref_points = (double*)malloc(numof_ref_points*dim*sizeof(double));
  long *redistIDs = &(redistQuery->gids[0]); //For debugging
  long *homeIDs = &(homeData->gids[0]); //For debugging

  #pragma omp parallel for
  for(int i = 0; i < numof_ref_points; i++) glb_ref_ids[i] = searchNode->data->gids[i];

  #pragma omp parallel for
  for(int i = 0; i < numof_ref_points*dim; i++) ref_points[i] = searchNode->data->X[i];

  //Find k nearest neighbors within this node.
  pair<double, long> *kneighbors;
  if (numof_query_points > 0) {
	  kneighbors = dist_directKQuery( ref_points, &(redistQuery->X[0]), 
                                      glb_ref_ids, numof_ref_points, numof_query_points,
                                      k, dim, comm );
  }

  //Sort in order of ascending query point ID.
  pair<long, double> *kthdist = new pair<long, double>[numof_query_points];
  if(numof_query_points > 0) {
	#pragma omp parallel for
	for(int i = 0; i < numof_query_points; i++) {
		kthdist[i].first = redistQuery->gids[i];
		kthdist[i].second = kneighbors[(i+1)*k-1].first;
	}
	delete[] kneighbors;
	free(ref_points);
	free(glb_ref_ids);
	omp_par::merge_sort(&(kthdist[0]), &(kthdist[numof_query_points]));
  }


  //vector<int> numof_query_points_per_rank(worldsize);
  //MPI_Allgather(&numof_query_points, 1, MPI_INT, &(numof_query_points_per_rank[0]), 1, MPI_INT, MPI_COMM_WORLD);
  //vector<long> scan_numof_query(worldsize);
  //scan_numof_query[0] = numof_query_points_per_rank[0];
  //for(int i = 1; i < scan_numof_query.size(); i++)
  //  scan_numof_query[i] = scan_numof_query[i-1] + numof_query_points_per_rank[i];

  //Collect search radii at "home" process of each query point (rank query_id/worldsize in MPI_COMM_WORLD).
  int *sendcounts = new int[worldsize];
  int *rcvcounts = new int[worldsize];
  int *senddisp = new int[worldsize];
  int *rcvdisp = new int[worldsize];
  for(int i = 0; i < worldsize; i++) sendcounts[i] = 0;
  for(int i = 0; i < numof_query_points; i++) {
      //sendcounts[ knn::lsh::idToHomeRank(kthdist[i].first, ppn, worldsize) ]++;
      sendcounts[ knn::home_rank(global_numof_query_points, worldsize, kthdist[i].first) ]++;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_CALL(MPI_Alltoall(sendcounts, 1, MPI_INT, rcvcounts, 1, MPI_INT, MPI_COMM_WORLD));
  omp_par::scan(sendcounts, senddisp, worldsize);
  omp_par::scan(rcvcounts, rcvdisp, worldsize);
  int rcvdists = rcvdisp[worldsize-1]+rcvcounts[worldsize-1];
  pair<long, double> *homedistances = new pair<long, double>[rcvdists];
  MPI_Datatype pairdata;
  MPI_Type_contiguous(sizeof(pair<long, double>), MPI_BYTE, &pairdata);
  MPI_Type_commit(&pairdata);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_CALL(MPI_Alltoallv(kthdist, sendcounts, senddisp, pairdata, homedistances,
			 rcvcounts, rcvdisp, pairdata, MPI_COMM_WORLD));
  
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


void bintree::queryK(	pbinData inData, long rootNpoints, double dupFactor,
						pbinNode searchNode,
						int k, vector<long> *queryIDs,
						vector< pair<double, long> > *kNN){

	int worldsize, worldrank;
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);

	double start_t = omp_get_wtime();

	int dim = inData->dim;
	int numof_query_points = inData->numof_points;
	int global_numof_query_points;
	MPI_CALL(MPI_Allreduce( &numof_query_points, &global_numof_query_points, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD ));
	int ppn = global_numof_query_points/worldsize; //Number of points per node

	#if STAGE_DTREE_OUTPUT_VERBOSE
		MPI_Barrier(MPI_COMM_WORLD);
        if(worldrank == 0) {
			cout<<"   + enter queryK "
				<<" inData->numof_points="<<inData->numof_points<<endl;
		}
		start_t = omp_get_wtime();
	#endif

    //vector<int> numof_query_points_per_rank(worldsize);
    //MPI_Allgather(&numof_query_points, 1, MPI_INT, &(numof_query_points_per_rank[0]), 1, MPI_INT, MPI_COMM_WORLD);
    //vector<long> scan_numof_query(worldsize);
    //scan_numof_query[0] = numof_query_points_per_rank[0];
    //for(int i = 1; i < scan_numof_query.size(); i++)
    //    scan_numof_query[i] = scan_numof_query[i-1] + numof_query_points_per_rank[i];

	//First, relocate query points to their home processes
    //This is also where we will collect the initial search radii and the query results.
	long *gid_copy = new long[numof_query_points];
	double *data_copy = new double[numof_query_points*dim];
	int *homeproc = new int[numof_query_points];
	int *sendcounts = new int[worldsize];
	#pragma omp parallel for schedule(static,256)
	for(int i = 0; i < worldsize; i++) sendcounts[i] = 0;
	double *new_data;
	long new_n, *new_ids;
	#pragma omp parallel for
	for(int i = 0; i < numof_query_points; i++)  {
		homeproc[i] = knn::home_rank(global_numof_query_points, worldsize, inData->gids[i]);
		//homeproc[i] = knn::lsh::idToHomeRank(inData->gids[i], ppn, worldsize);
		//homeproc[i] = knn::lsh::idToHomeRank(inData->gids[i], scan_numof_query);
		gid_copy[i] = inData->gids[i];
	}
	#pragma omp parallel for
	for(int i = 0; i < numof_query_points*dim; i++)  data_copy[i] = inData->X[i];
	for(int i = 0; i < numof_query_points; i++)
		sendcounts[homeproc[i]]++;

	local_rearrange(&gid_copy, &homeproc, &data_copy, numof_query_points, dim);
	knn::repartition::repartition(gid_copy, data_copy, numof_query_points, sendcounts, dim,
								&new_ids, &new_data, &new_n, MPI_COMM_WORLD);

	delete[] gid_copy;
	delete[] data_copy;
	delete[] homeproc;

	inData->numof_points = new_n;
	inData->X.resize(new_n*dim);
	inData->gids.resize(new_n);
	#pragma omp parallel for
	for(int i = 0; i < new_n; i++) inData->gids[i] = new_ids[i];
	#pragma omp parallel for
	for(int i = 0; i < new_n*dim; i++)  inData->X[i] = new_data[i];
	delete[] new_ids;
	delete[] new_data;
	numof_query_points = new_n;

	#if STAGE_DTREE_OUTPUT_VERBOSE
		MPI_Barrier(MPI_COMM_WORLD);
		if(worldrank == 0) {
			cout<<"   + repartition points to home process done -> "<<omp_get_wtime() - start_t
				<<" inData->numof_points="<<inData->numof_points<<endl;
		}
		start_t = omp_get_wtime();
	#endif

	//Create a copy of the query set.
	pbinData querycopy = new binData();
	querycopy->Copy(inData);

	//Initial pass to determine search radii.
	double *R;
	pbinData redistQuery;
	pbinNode leaf;
	bintree::distributeToNearestLeaf(querycopy, searchNode, &redistQuery, &leaf);

	#if STAGE_DTREE_OUTPUT_VERBOSE
		MPI_Barrier(MPI_COMM_WORLD);
		if(worldrank == 0) {
			cout<<"   + distribute points to nearest leaf done -> "<<omp_get_wtime() - start_t
				<<" redistQuery->numof_points="<<redistQuery->numof_points<<endl;
		}
		start_t = omp_get_wtime();
	#endif


	bintree::queryKSelectRs(redistQuery, inData, leaf, global_numof_query_points, k, &R);
	delete redistQuery;

	#if STAGE_DTREE_OUTPUT_VERBOSE
		MPI_Barrier(MPI_COMM_WORLD);
		if(worldrank == 0) {
			cout<<"   + find R for each point -> "<<omp_get_wtime() - start_t<<endl;
		}
		start_t = omp_get_wtime();
	#endif

	//Perform r-query with individual radii
	inData->radii.resize(numof_query_points);
	#pragma omp parallel for
	for(int i = 0; i < numof_query_points; i++) inData->radii[i] = R[i];
	delete[] R;

	vector< pair<double, long> > *rneighbors;
	querycopy->Copy(inData);
	int nvectors;
	bintree::queryRK(querycopy, rootNpoints, dupFactor, k, searchNode, rneighbors, &nvectors);

	#if STAGE_DTREE_OUTPUT_VERBOSE
		MPI_Barrier(MPI_COMM_WORLD);
		if(worldrank == 0) {
			cout<<"   + queryRK done -> "<<omp_get_wtime() - start_t
				<<" nvectors="<<nvectors<<endl;
		}
		start_t = omp_get_wtime();
	#endif

	//Pack neighbors into array of (queryID, distance, refID) triples. 
	int *neighcount = new int[nvectors];
	int *neighcountscan = new int[nvectors];
	#pragma omp parallel for
	for(int i = 0; i < nvectors; i++)
		neighcount[i] = rneighbors[i].size();
	omp_par::scan(neighcount, neighcountscan, nvectors);

	int totalneighbors = 0;
	if(nvectors > 0) totalneighbors = neighcountscan[nvectors-1] + neighcount[nvectors-1];

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


	#if PCL_DEBUG_VERBOSE
        for(int rr = 0; rr < worldsize; rr++) {
			if(rr == worldrank) {
				for(int i = 0; i < nvectors; i++) {
					cout<<"("<<worldrank<<") id: "<<querycopy->gids[i]<<"  ";
					for(int j = 0; j < neighcount[i]; j++)
						cout<<"("<<rneighbors[i][j].second<<" "<<rneighbors[i][j].first<<")  ";
					cout<<endl;
				}
			}
			cout.flush();
			fflush(stdout);
			MPI_Barrier(MPI_COMM_WORLD);
			if(worldrank == 0) cout<<"+"<<endl;
		}
	#endif


	delete[] rneighbors;
	delete[] neighcount;
	delete[] neighcountscan;
	delete querycopy;


	//Sort array of triples and transimit to appropriate home processes.
	omp_par::merge_sort(tNeighbors, &(tNeighbors[totalneighbors]));
	int *rcvcounts = new int[worldsize];
	int *senddisp = new int[worldsize];
	int *rcvdisp = new int[worldsize];
	for(int i = 0; i < worldsize; i++) sendcounts[i] = 0;
    for(int i = 0; i < totalneighbors; i++) {
        //sendcounts[ knn::lsh::idToHomeRank(tNeighbors[i].first, ppn, worldsize) ]++;
        sendcounts[ knn::home_rank(global_numof_query_points, worldsize, tNeighbors[i].first) ]++;
    }
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_CALL(MPI_Alltoall(sendcounts, 1, MPI_INT, rcvcounts, 1, MPI_INT, MPI_COMM_WORLD));
	omp_par::scan(sendcounts, senddisp, worldsize);
	omp_par::scan(rcvcounts, rcvdisp, worldsize);
	int rcvneighs = rcvdisp[worldsize-1]+rcvcounts[worldsize-1];

	#if STAGE_DTREE_OUTPUT_VERBOSE
		MPI_Barrier(MPI_COMM_WORLD);
        if(worldrank == 0) {
			cout<<"   + repartition results prepare done -> "<<omp_get_wtime() - start_t
				<<" nvectors="<<nvectors<<endl;
		}
		start_t = omp_get_wtime();
	#endif

	#if PCL_DEBUG_VERBOSE
		cout<<"("<<worldrank<<") rcvneighs = "<<rcvneighs<<" numof_query_points = "<<numof_query_points<<endl;
	#endif

	assert(rcvneighs >= numof_query_points*k);
	triple<long, double, long> *homeneighbors = new triple<long, double, long>[rcvneighs];
	MPI_Datatype tripledata;
	MPI_Type_contiguous(sizeof(triple<long, double, long>), MPI_BYTE, &tripledata);
	MPI_Type_commit(&tripledata);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_CALL(MPI_Alltoallv(tNeighbors, sendcounts, senddisp, tripledata, homeneighbors,
							rcvcounts, rcvdisp, tripledata, MPI_COMM_WORLD));

	delete[] tNeighbors;
	MPI_Type_free(&tripledata);

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


	#if STAGE_DTREE_OUTPUT_VERBOSE
		MPI_Barrier(MPI_COMM_WORLD);
		if(worldrank == 0) cout<<"   + repartition results done -> "<<omp_get_wtime() - start_t<<endl;
		start_t = omp_get_wtime();
	#endif

	#if PCL_DEBUG_VERBOSE
		if(worldrank == 0) cout<<endl<<"homeneighbors: "<<endl;
		for(int i = 0; i < rcvneighs; i++)
			cout<<"("<<homeneighbors[i].first<<" "<<homeneighbors[i].third<<" "<<homeneighbors[i].second<<")  ";
		cout<<endl;
	#endif


	#if PCL_DEBUG_VERBOSE
		for(int t = 0; t < worldsize; t++) {
			if(worldrank == t) {
				cout<<"("<<worldrank<<") nq = "<<numof_query_points
					<<" rcvneighbors = "<<rcvneighs<<endl;
				cout<<"  offset: ";
				for(int i = 0; i < numof_query_points; i++)
					cout<<offsets[i]<<" ";
				cout<<endl;
			}
			cout.flush();
			MPI_Barrier(MPI_COMM_WORLD);
		}
	#endif

	//Store results in output vectors.
	queryIDs->resize(numof_query_points);
	kNN->resize(k*numof_query_points) ;
	#pragma omp parallel for
	for(int i = 0; i < numof_query_points; i++) {
		double min = DBL_MAX, last_min = -1.0;
		long min_idx = -1, last_min_idx = -1;

		(*queryIDs)[i]=homeneighbors[offsets[i]].first;
		int curroffset=offsets[i];

		for( int j = 0; j < k; j ++ ) {
			long currID = homeneighbors[offsets[i]].first;
			int l=0;

			//Loop over all neighbors for the same query point
			while((curroffset+l < rcvneighs) && (homeneighbors[curroffset+l].first == currID))  { 
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

	#if STAGE_DTREE_OUTPUT_VERBOSE
		MPI_Barrier(MPI_COMM_WORLD);
		if(worldrank == 0) cout<<"   + store results output done -> "<<omp_get_wtime() - start_t<<endl;
		start_t = omp_get_wtime();
	#endif

	delete[] homeneighbors;  
	delete[] sendcounts;
	delete[] rcvcounts;
	delete[] senddisp;   
	delete[] rcvdisp;
	delete[] offsets;


}



void bintree::queryRK( pbinData inData, long rootNpoints,
				double dupFactor, int k,
				pbinNode searchNode,
				vector< pair<double, long> > *&neighbors,
				int *nvectors)
{

  int size, rank;
  int worldrank, worldsize;
  MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldsize);

  double start_t = 0.0;

  int total_query_points; 
  int dim = inData->dim;
  //int numof_query_points = inData->X.size() / dim;
  int numof_query_points = inData->numof_points;
  assert( inData->radii.size() == numof_query_points );

  if(searchNode->options.pruning_verbose) {
    MPI_CALL(MPI_Allreduce( &numof_query_points, &total_query_points, 
			    1, MPI_INT, MPI_SUM, searchNode->comm ));
   }

  binData* leafData;
  pbinNode leaf;
  bintree::distributeToLeaves(inData, rootNpoints, dupFactor, 
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
  

  //numof_query_points = inData->X.size() / dim;
  numof_query_points = inData->numof_points;
  int global_numof_query_points = 0;
  MPI_CALL(MPI_Allreduce( &numof_query_points, &global_numof_query_points, 
			  1, MPI_INT, MPI_SUM, comm ));
  //int numof_ref_points = searchNode->data->X.size() / dim;
  int numof_ref_points = searchNode->data->numof_points;
  
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
	int avg_nq = total_query_points / worldsize;
	double mypruning = ((double)(total_query_points - global_numof_query_points))
								/ ((double)(total_query_points - avg_nq)) * 100.0;

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

	  //cout<<"Number of ref points per process: ";
	  //for(int i = 0; i < worldsize; i++) 
	  //	  cout<<nrs[i]<<" ";
	  //cout<<endl;
	 
	  //cout<<"Number of query points per process: ";
	  //for(int i = 0; i < worldsize; i++) 
		//  cout<<nps[i]<<" ";
	  //cout<<endl;
	
      cout<< "\tnumber of ref points handled per process:" 
		  << "\tmax: " << maxrpproc << "  min: " << minrpproc << "  avg: " << avgrpproc << endl;  
      
	  cout<< "\tnumber of query points handled per process:" 
		  << "\tmax: " << maxppproc << "  min: " << minppproc << "  avg: " << avgppproc << endl;  
      
	  cout<< "\tpercentage of query points pruned:" 
		  << "\tmax: " << maxpruning << "  min: " << minpruning << "  avg: " << avgpruning << endl;  
    }

	delete [] nps;
	delete [] nrs;

    MPI_Barrier(MPI_COMM_WORLD);
  }


  start_t = omp_get_wtime();
  if(numof_query_points > 0) {
	knn::dist_directRQueryIndividualK( ref_points, &(inData->X[0]), 
                         global_numof_ref_points, 
						 global_numof_query_points, 
                         numof_ref_points, numof_query_points, 
                         &(inData->radii[0]), k, dim, glb_ref_ids, 
                         neighbors, 
                         comm);
  }
  Direct_Kernel_T_ += omp_get_wtime() - start_t;


  free(ref_points);
  free(glb_ref_ids);
  
  return;
}



// -------------- for "rkdt" -------------------
/*
// ==================== this is the original function ==============
void bintree::queryK_Greedy_a2a( long rootNpoints,
							 pbinNode searchNode, int k,
							 int max_points, int max_tree_level,
							 vector<long> *queryIDs,
							 vector< pair<double, long> > *kNN)
{
    double start_t, end_t;
	double stage_t;

	int worldrank, worldsize;
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);

	if(STAGE_OUTPUT_VERBOSE) stage_t = omp_get_wtime();

	pbinNode leaf = searchNode;
	while(leaf->kid != NULL)
		leaf = leaf->kid;

	int dim = leaf->data->dim;
	long global_numof_query_points = rootNpoints;
	int ppn = global_numof_query_points/worldsize; //Number of points per node
	int homepoints = (worldrank==worldsize-1) ? ppn+global_numof_query_points%ppn : ppn; //Number of query points "owned" by each process

    if(_RKDT_ALL2ALL_DEBUG_) {
        for(int r = 0; r < worldsize; r++) {
            if(worldrank == r) {
                cout<<"rank "<<worldrank<<": rootNpoints = "<<rootNpoints
                    <<", global_numof_query_points = "<<global_numof_query_points
                    <<", ppn = "<<ppn
                    <<", glb_query / worldsize = "<<global_numof_query_points/worldsize
                    <<", homepoints = "<<homepoints
                    <<endl;
                cout.flush();
            }
            MPI_Barrier( MPI_COMM_WORLD );
        }
    }

	pbinData redistQuery = leaf->data;

	int numof_ref_points = leaf->data->numof_points;
	int numof_query_points = numof_ref_points;

	if(STAGE_OUTPUT_VERBOSE) {
		MPI_Barrier(MPI_COMM_WORLD);
        if(worldrank == 0) {
			cout<<"  = Query: get leaf points done! -> "<<omp_get_wtime() - stage_t<<endl;
		}
		stage_t = omp_get_wtime();
    }

	int totalneighbors = k*numof_query_points;
	triple<long, double, long> *tNeighbors = new triple<long, double, long>[totalneighbors];

	long *glb_ref_ids = new long [numof_ref_points];
	#pragma omp parallel for
	for(int i = 0; i < numof_ref_points; i++)
		glb_ref_ids[i] = leaf->data->gids[i];

	//Find k nearest neighbors within this node.
	int leafCommSize;
	MPI_Comm_size(leaf->comm, &leafCommSize);

    if(_RKDT_ALL2ALL_DEBUG_) {
        MPI_Barrier( MPI_COMM_WORLD );
        if(worldrank == 0) cout<<"\t\t\t queryK_Greedy_a2a: go to leaf, before search"<<endl;
    }

    if(leafCommSize > 1) {
		double *ref_points = new double [numof_ref_points*dim];
		#pragma omp parallel for
		for(int i = 0; i < numof_ref_points*dim; i++)
			ref_points[i] = leaf->data->X[i];

		pair<double, long> *kneighbors;
        if(numof_query_points > 0) {
			kneighbors = dist_directKQuery( ref_points, &(redistQuery->X[0]), glb_ref_ids,
											numof_ref_points, numof_query_points,
														k, dim, leaf->comm );
		}
		delete [] ref_points;

		//Pack neighbors into array of (queryID, distance, refID) triples.
		#pragma omp parallel for
		for(int i = 0; i < numof_query_points; i++) {
			for(int j = 0; j < k; j++) {
				triple<long, double, long> *currNeigh;
				currNeigh = &(tNeighbors[i*k+j]);
				currNeigh->first = glb_ref_ids[i];		//redistQuery->gids[i];
				currNeigh->second = kneighbors[i*k+j].first;
				currNeigh->third = kneighbors[i*k+j].second;
			}
		}

		delete [] kneighbors;
	}
    else {	// leafCommSize == 1
		vector< pair<double, long> > *kneighbors = new vector< pair<double, long> >;
	    if(numof_query_points > 0) {
			int flag_stree_r = 0;
			if(leaf->options.flag_r == 2) flag_stree_r = 2;
			stTreeSearch_rkdt_a2a_me(redistQuery, k, flag_stree_r,
									 max_points, max_tree_level - leaf->level,
									 kneighbors);

			// redistQuery has been deleted inside the above function
			// since redistQuery point to leaf->data, that means leaf->data has been released. set it to NULL then
			leaf->data = NULL;
		}

        if(_RKDT_ALL2ALL_DEBUG_) {
            MPI_Barrier( MPI_COMM_WORLD );
            if(worldrank == 0) cout<<"\t\t\t queryK_Greedy_a2a: shared memory search done"<<endl;
        }

		//Pack neighbors into array of (queryID, distance, refID) triples.
		#pragma omp parallel for
		for(int i = 0; i < numof_ref_points; i++) {
			for(int j = 0; j < k; j++) {
				triple<long, double, long> *currNeigh;
				currNeigh = &(tNeighbors[i*k+j]);
				currNeigh->first = glb_ref_ids[i];	//redistQuery->gids[i];
				currNeigh->second = (*kneighbors)[i*k+j].first;
				currNeigh->third = (*kneighbors)[i*k+j].second;
			}
		}

		delete kneighbors;
	}	// end else
	delete [] glb_ref_ids;


	if(STAGE_OUTPUT_VERBOSE) {
		MPI_Barrier(MPI_COMM_WORLD);
		if(worldrank == 0) {
			cout<<"  = Query: find nn done! -> "<<omp_get_wtime() - stage_t
				<<" leafCommSize: "<<leafCommSize
				<<endl;
		}
		stage_t = omp_get_wtime();
    }


    //Sort array of triples and transimit to appropriate home processes.
    if(totalneighbors > 0) omp_par::merge_sort(tNeighbors, &(tNeighbors[totalneighbors]));


#if COMM_TIMING_VERBOSE
  MPI_Barrier(MPI_COMM_WORLD);
  start_t = omp_get_wtime();
#endif

    if(_RKDT_ALL2ALL_DEBUG_) {
        MPI_Barrier( MPI_COMM_WORLD );
        if(worldrank == 0) cout<<"\t\t\t queryK_Greedy_a2a: redistribute knn to origin begin"<<endl;
    }

    //vector<int> numof_query_points_per_rank(worldsize);
    //MPI_Allgather(&numof_query_points, 1, MPI_INT, &(numof_query_points_per_rank[0]), 1, MPI_INT, MPI_COMM_WORLD);
    //vector<long> scan_numof_query(worldsize);
    //scan_numof_query[0] = numof_query_points_per_rank[0];
    //for(int i = 1; i < scan_numof_query.size(); i++)
    //    scan_numof_query[i] = scan_numof_query[i-1] + numof_query_points_per_rank[i];

    int *sendcounts = new int[worldsize];
    int *rcvcounts = new int[worldsize];
    int *senddisp = new int[worldsize];
    int *rcvdisp = new int[worldsize];
    #pragma omp parallel for
    for(int i = 0; i < worldsize; i++) sendcounts[i] = 0;
    for(int i = 0; i < totalneighbors; i++) sendcounts[ knn::lsh::idToHomeRank(tNeighbors[i].first, ppn, worldsize) ]++;
    //for(int i = 0; i < totalneighbors; i++) sendcounts[ knn::lsh::idToHomeRank(tNeighbors[i].first, scan_numof_query) ]++;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_CALL(MPI_Alltoall(sendcounts, 1, MPI_INT, rcvcounts, 1, MPI_INT, MPI_COMM_WORLD));
    omp_par::scan(sendcounts, senddisp, worldsize);
    omp_par::scan(rcvcounts, rcvdisp, worldsize);
    int rcvneighs = rcvdisp[worldsize-1]+rcvcounts[worldsize-1];

    if(_RKDT_ALL2ALL_DEBUG_) {
        for(int r = 0; r < worldsize; r++) {
            if(worldrank == r) {
                cout<<"rank "<<worldrank<<": rcvneighs = "<<rcvneighs<<", homepoints = "<<homepoints
                    <<", k = "<<k<<", homepoints*k = "<<homepoints*k
                    <<", ppn = "<<ppn<<endl;
                cout.flush();
            }
            MPI_Barrier( MPI_COMM_WORLD );
        }
    }

    assert(rcvneighs == homepoints*k);
    triple<long, double, long> *homeneighbors = new triple<long, double, long>[rcvneighs];
    MPI_Datatype tripledata;
    MPI_Type_contiguous(sizeof(triple<long, double, long>), MPI_BYTE, &tripledata);
    MPI_Type_commit(&tripledata);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_CALL(MPI_Alltoallv(tNeighbors, sendcounts, senddisp, tripledata, homeneighbors,
			 rcvcounts, rcvdisp, tripledata, MPI_COMM_WORLD));
    delete[] tNeighbors;

    if(_RKDT_ALL2ALL_DEBUG_) {
        MPI_Barrier( MPI_COMM_WORLD );
        if(worldrank == 0) cout<<"\t\t\t queryK_Greedy_a2a: redistribute knn to origin done"<<endl;
    }


#if ALL_TO_ALL_VERBOSE
  int tmpsendsum = 0, tmprecvsum = 0;
  for(int i = 0; i < worldsize; i++)
	  tmpsendsum += sendcounts[i];
  for(int i = 0; i < worldsize; i++)
	  tmprecvsum += rcvcounts[i];
  cout<<"worldrank: "<<worldrank<<" recvcount: "<<tmprecvsum<<" sendcount: "<<tmpsendsum<<endl;
#endif

#if COMM_TIMING_VERBOSE
  Repartition_Query_T_ += omp_get_wtime() - start_t;
#endif


    omp_par::merge_sort(homeneighbors, &(homeneighbors[rcvneighs]));

    if(_RKDT_ALL2ALL_DEBUG_) {
        MPI_Barrier( MPI_COMM_WORLD );
        if(worldrank == 0) cout<<"\t\t\t queryK_Greedy_a2a: sort home neighbors"<<endl;
    }


    if(STAGE_OUTPUT_VERBOSE) {
        if(worldrank == 0)
            cout<<"  = Query: repartition results to home process done! -> "<<omp_get_wtime() - stage_t<<endl;
        stage_t = omp_get_wtime();
    }

    //Store results in output vectors.

    if(_RKDT_ALL2ALL_DEBUG_) {
        MPI_Barrier( MPI_COMM_WORLD );
        if(worldrank == 0) {
            int tmp_knn_size = homepoints*k;
            cout<<"\t\t\t queryK_Greedy_a2a: homepoints = "<<homepoints
                <<", k = "<<k<<", knn_resize = "<<tmp_knn_size<<endl;
        }
    }

    queryIDs->resize(homepoints);
    kNN->resize(k*homepoints) ;

    if(_RKDT_ALL2ALL_DEBUG_) {
        MPI_Barrier( MPI_COMM_WORLD );
        if(worldrank == 0)
            cout<<"\t\t\t queryK_Greedy_a2a: allocate memory for final result done"<<endl;
    }

    #pragma omp parallel for
    for(int i = 0; i < homepoints; i++) {
        (*queryIDs)[i]=homeneighbors[i*k].first;
        for( int j = 0; j < k; j ++ ) {
            (*kNN)[i*k+j].first = homeneighbors[i*k+j].second;
            (*kNN)[i*k+j].second = homeneighbors[i*k+j].third;
        }
    }

    if(_RKDT_ALL2ALL_DEBUG_) {
        MPI_Barrier( MPI_COMM_WORLD );
        if(worldrank == 0) cout<<"\t\t\t queryK_Greedy_a2a: pack results done"<<endl;
    }

    MPI_Type_free(&tripledata);
    //delete redistQuery;
    delete[] homeneighbors;
    delete[] sendcounts;
    delete[] rcvcounts;
    delete[] senddisp;
    delete[] rcvdisp;

    if(_RKDT_ALL2ALL_DEBUG_) {
        MPI_Barrier( MPI_COMM_WORLD );
        if(worldrank == 0) cout<<"\t\t\t queryK_Greedy_a2a: release memory"<<endl;
    }

	if(STAGE_OUTPUT_VERBOSE) {
		if(worldrank == 0)
            cout<<"  = Query: prepare output done! -> "<<omp_get_wtime() - stage_t<<endl;
    }

}
*/



// ==================== this is the cleaned and debug version ==============
void bintree::queryK_Greedy_a2a(long rootNpoints, pbinNode searchNode, int k,
							    int max_points, int max_tree_level,
							    vector<long> *queryIDs,
							    vector< pair<double, long> > *kNN)
{
    int worldrank, worldsize;
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);

    double start_t = omp_get_wtime(), end_t, stage_t = omp_get_wtime();
    double srkdt_t;

	pbinNode leaf = searchNode;
	while(leaf->kid != NULL) leaf = leaf->kid;

	int dim = leaf->data->dim;
	long global_numof_query_points = rootNpoints;
	int ppn = global_numof_query_points/worldsize; //Number of points per node
	//int homepoints = (worldrank==worldsize-1) ? ppn+global_numof_query_points%ppn : ppn; //Number of query points "owned" by each process
    long divd = global_numof_query_points / worldsize;
    long rem = global_numof_query_points % worldsize;
    int homepoints = worldrank < rem ? divd+1 : divd;

    if(_RKDT_ALL2ALL_DEBUG_) {
        for(int r = 0; r < worldsize; r++) {
            if(worldrank == r) {
                cout<<"rank "<<worldrank<<": rootNpoints = "<<rootNpoints
                    <<", global_numof_query_points = "<<global_numof_query_points
                    <<", ppn = "<<ppn
                    <<", glb_query / worldsize = "<<global_numof_query_points/worldsize
                    <<", homepoints = "<<homepoints
                    <<endl;
                cout.flush();
            }
            MPI_Barrier( MPI_COMM_WORLD );
        }
    }

	pbinData redistQuery = leaf->data;

	int numof_ref_points = leaf->data->numof_points;
	int numof_query_points = numof_ref_points;
	long totalneighbors = k*numof_query_points;
	triple<long, double, long> *tNeighbors = new triple<long, double, long>[totalneighbors];

	long *glb_ref_ids = new long [numof_ref_points];
	#pragma omp parallel for
	for(int i = 0; i < numof_ref_points; i++)
		glb_ref_ids[i] = leaf->data->gids[i];

	//Find k nearest neighbors within this node.
	int leafCommSize;
	MPI_Comm_size(leaf->comm, &leafCommSize);

    if(_RKDT_ALL2ALL_DEBUG_) {
        MPI_Barrier( MPI_COMM_WORLD );
        if(worldrank == 0) cout<<"\t\t\t queryK_Greedy_a2a: go to leaf, before search"<<endl;
    }

    if(leafCommSize > 1) {
		double *ref_points = new double [numof_ref_points*dim];
		#pragma omp parallel for
		for(int i = 0; i < numof_ref_points*dim; i++)
			ref_points[i] = leaf->data->X[i];

		pair<double, long> *kneighbors;
        if(numof_query_points > 0) {
			kneighbors = dist_directKQuery( ref_points, &(redistQuery->X[0]), glb_ref_ids,
											numof_ref_points, numof_query_points, k, dim, leaf->comm );
		}
		delete [] ref_points;

		//Pack neighbors into array of (queryID, distance, refID) triples.
		#pragma omp parallel for
        for(int i = 0; i < numof_query_points; i++) {
            for(int j = 0; j < k; j++) {
				triple<long, double, long> *currNeigh;
				currNeigh = &(tNeighbors[i*k+j]);
				currNeigh->first = glb_ref_ids[i];		//redistQuery->gids[i];
				currNeigh->second = kneighbors[i*k+j].first;
				currNeigh->third = kneighbors[i*k+j].second;
			}
		}

		delete [] kneighbors;
	}
    else {	// leafCommSize == 1
		vector< pair<double, long> > *kneighbors = new vector< pair<double, long> >;
			//int flag_stree_r = 0;
			//if(leaf->options.flag_r == 2) flag_stree_r = 2;
			//stTreeSearch_rkdt_a2a_me(redistQuery, k, flag_stree_r,
			//	max_points, max_tree_level - leaf->level, kneighbors);

			// redistQuery has been deleted inside the above function
			// since redistQuery point to leaf->data, that means leaf->data has been released. set it to NULL then
			//leaf->data = NULL;

        srkdt_t = omp_get_wtime();
        find_knn_srkdt_a2a(redistQuery, k, max_points, max_tree_level-leaf->level, 1, kneighbors);
        STree_Search_T_ += omp_get_wtime() - srkdt_t;
        delete leaf->data;
        leaf->data = NULL;

        if(false) {
            for(int r = 0; r < worldsize; r++) {
                if(r == worldrank) {
                    for(int ii = 0; ii < numof_ref_points; ii++) {
                        cout<<"rank "<<worldrank<<" ["<<glb_ref_ids[ii]<<"]: ";
                        for(int jj = 0; jj < k; jj++) {
                            cout<<"("<<(*kneighbors)[ii*k+jj].second<<", "<<(*kneighbors)[ii*k+jj].first<<") ";
                           }
                    }
                    cout.flush();
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
        }

        if(_RKDT_ALL2ALL_DEBUG_) {
            MPI_Barrier( MPI_COMM_WORLD );
            if(worldrank == 0) cout<<"\t\t\t queryK_Greedy_a2a: shared memory search done"<<endl;
        }

		//Pack neighbors into array of (queryID, distance, refID) triples.
		#pragma omp parallel for
        for(int i = 0; i < numof_ref_points; i++) {
            for(int j = 0; j < k; j++) {
			    triple<long, double, long> *currNeigh;
			    currNeigh = &(tNeighbors[i*k+j]);
			    currNeigh->first = glb_ref_ids[i];	//redistQuery->gids[i];
				currNeigh->second = (*kneighbors)[i*k+j].first;
			    currNeigh->third = (*kneighbors)[i*k+j].second;
		    }
	    }

		delete kneighbors;
	}	// end else
	delete [] glb_ref_ids;

    //Sort array of triples and transimit to appropriate home processes.
    if(totalneighbors > 0) omp_par::merge_sort(tNeighbors, &(tNeighbors[totalneighbors]));

    #if COMM_TIMING_VERBOSE
        MPI_Barrier(MPI_COMM_WORLD);
        start_t = omp_get_wtime();
    #endif

    if(_RKDT_ALL2ALL_DEBUG_) {
        MPI_Barrier( MPI_COMM_WORLD );
        if(worldrank == 0) cout<<"\t\t\t queryK_Greedy_a2a: redistribute knn to origin begin"<<endl;
    }

    int *sendcounts = new int[worldsize];
    int *rcvcounts = new int[worldsize];
    int *senddisp = new int[worldsize];
    int *rcvdisp = new int[worldsize];
    #pragma omp parallel for
    for(int i = 0; i < worldsize; i++) sendcounts[i] = 0;
    for(int i = 0; i < totalneighbors; i++) {
        //sendcounts[ knn::lsh::idToHomeRank(tNeighbors[i].first, ppn, worldsize) ]++;
        sendcounts[ knn::home_rank(global_numof_query_points, worldsize, tNeighbors[i].first) ]++;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_CALL(MPI_Alltoall(sendcounts, 1, MPI_INT, rcvcounts, 1, MPI_INT, MPI_COMM_WORLD));
    omp_par::scan(sendcounts, senddisp, worldsize);
    omp_par::scan(rcvcounts, rcvdisp, worldsize);
    int rcvneighs = rcvdisp[worldsize-1]+rcvcounts[worldsize-1];

    if(_RKDT_ALL2ALL_DEBUG_) {
        for(int r = 0; r < worldsize; r++) {
            if(worldrank == r) {
                cout<<"rank "<<worldrank<<": rcvneighs = "<<rcvneighs<<", homepoints = "<<homepoints
                    <<", k = "<<k<<", homepoints*k = "<<homepoints*k
                    <<", ppn = "<<ppn<<endl;
                cout.flush();
            }
            MPI_Barrier( MPI_COMM_WORLD );
        }
    }

    if(_RKDT_ALL2ALL_DEBUG_) {
        for(int r = 0; r < worldsize; r++) {
            if(worldrank == r) {
                cout<<"rank "<<worldrank<<" tNeighbors "<<totalneighbors<<", homeneighbors = "<<rcvneighs<<endl;
                //cout<<"rank "<<worldrank<<": send_count ";
                //for(int i = 0; i < worldsize; i++) cout<<sendcounts[i]<<" ";
                //cout<<endl;
                //cout<<"rank "<<worldrank<<": send_disp ";
                //for(int i = 0; i < worldsize; i++) cout<<senddisp[i]<<" ";
                //cout<<endl;
                //cout<<"rank "<<worldrank<<": recv_count ";
                //for(int i = 0; i < worldsize; i++) cout<<rcvcounts[i]<<" ";
                //cout<<endl;
                //cout<<"rank "<<worldrank<<": recv_disp ";
                //for(int i = 0; i < worldsize; i++) cout<<rcvdisp[i]<<" ";
                //cout<<endl;
                cout.flush();
            }
            MPI_Barrier( MPI_COMM_WORLD );
        }
    }

    assert(rcvneighs == homepoints*k);
    triple<long, double, long> *homeneighbors = new triple<long, double, long>[rcvneighs];
    MPI_Datatype tripledata;
    MPI_Type_contiguous(sizeof(triple<long, double, long>), MPI_BYTE, &tripledata);
    MPI_Type_commit(&tripledata);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_CALL(MPI_Alltoallv(tNeighbors, sendcounts, senddisp, tripledata, homeneighbors,
			 rcvcounts, rcvdisp, tripledata, MPI_COMM_WORLD));
    delete [] tNeighbors;

    if(_RKDT_ALL2ALL_DEBUG_) {
        MPI_Barrier( MPI_COMM_WORLD );
        if(worldrank == 0) cout<<"\t\t\t queryK_Greedy_a2a: redistribute knn to origin done"<<endl;
    }

    #if COMM_TIMING_VERBOSE
        Repartition_Query_T_ += omp_get_wtime() - start_t;
    #endif

    omp_par::merge_sort(homeneighbors, &(homeneighbors[rcvneighs]));

    if(_RKDT_ALL2ALL_DEBUG_) {
        MPI_Barrier( MPI_COMM_WORLD );
        if(worldrank == 0) cout<<"\t\t\t queryK_Greedy_a2a: sort home neighbors"<<endl;
    }

    //Store results in output vectors.

    if(_RKDT_ALL2ALL_DEBUG_) {
        for(int r = 0; r < worldsize; r++) {
            if(worldrank == r) {
                int tmp_knn_size = homepoints*k;
                cout<<"\t\t\trank "<<worldrank<<": queryK_Greedy_a2a: homepoints = "<<homepoints
                    <<", k = "<<k<<", knn_resize = "<<tmp_knn_size<<endl;
                cout.flush();
            }
            MPI_Barrier( MPI_COMM_WORLD );
        }
    }

    queryIDs->resize(homepoints);

    if(_RKDT_ALL2ALL_DEBUG_) {
        MPI_Barrier( MPI_COMM_WORLD );
        if(worldrank == 0)
            cout<<"\t\t\t queryK_Greedy_a2a: allocate memory for queryIDs done"<<endl;
    }

    kNN->resize(k*homepoints) ;

    if(_RKDT_ALL2ALL_DEBUG_) {
        MPI_Barrier( MPI_COMM_WORLD );
        if(worldrank == 0)
            cout<<"\t\t\t queryK_Greedy_a2a: allocate memory for knn done"<<endl;
    }

    #pragma omp parallel for
    for(int i = 0; i < homepoints; i++) {
        (*queryIDs)[i]=homeneighbors[i*k].first;
        for( int j = 0; j < k; j ++ ) {
            (*kNN)[i*k+j].first = homeneighbors[i*k+j].second;
            (*kNN)[i*k+j].second = homeneighbors[i*k+j].third;
        }
    }

    if(_RKDT_ALL2ALL_DEBUG_) {
        MPI_Barrier( MPI_COMM_WORLD );
        if(worldrank == 0) cout<<"\t\t\t queryK_Greedy_a2a: pack results done"<<endl;
    }

    MPI_Type_free(&tripledata);
    //delete redistQuery;
    delete[] homeneighbors;
    delete[] sendcounts;
    delete[] rcvcounts;
    delete[] senddisp;
    delete[] rcvdisp;

    if(_RKDT_ALL2ALL_DEBUG_) {
        MPI_Barrier( MPI_COMM_WORLD );
        if(worldrank == 0) cout<<"\t\t\t queryK_Greedy_a2a: release memory"<<endl;
    }
}



/*
// currently, I just add one function to quickly fix the bugs for fmm
void bintree::queryK_Greedy_a2a( long rootNpoints,
							 pbinNode searchNode, int k,
							 int max_points, int max_tree_level,
                             vector<long> &inscan_numof_query_points_per_rank,
							 vector<long> *queryIDs,
							 vector< pair<double, long> > *kNN)
{
	double start_t, end_t;
	double stage_t;

	int worldrank, worldsize;
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);

	#if STAGE_OUTPUT_VERBOSE
		stage_t = omp_get_wtime();
	#endif

	pbinNode leaf = searchNode;
	while(leaf->kid != NULL)
		leaf = leaf->kid;

	int dim = leaf->data->dim;
	int global_numof_query_points = rootNpoints;
	int ppn = global_numof_query_points/worldsize; //Number of points per node
	int homepoints = (worldrank==worldsize-1) ? ppn+global_numof_query_points%ppn : ppn; //Number of query points "owned" by each process

	pbinData redistQuery = leaf->data;

	//int numof_query_points = redistQuery->X.size() / dim;
	//int numof_ref_points = leaf->data->X.size() / dim;

	int numof_ref_points = leaf->data->numof_points;
	int numof_query_points = numof_ref_points;

	#if STAGE_OUTPUT_VERBOSE
		MPI_Barrier(MPI_COMM_WORLD);
		if(worldrank == 0)
			cout<<"  = Query: get leaf points done! -> "<<omp_get_wtime() - stage_t<<endl;
		stage_t = omp_get_wtime();
	#endif

	int totalneighbors = k*numof_query_points;
	triple<long, double, long> *tNeighbors = new triple<long, double, long>[totalneighbors];

	long *glb_ref_ids = new long [numof_ref_points];
	#pragma omp parallel for
	for(int i = 0; i < numof_ref_points; i++)
		glb_ref_ids[i] = leaf->data->gids[i];

	//Find k nearest neighbors within this node.
	int leafCommSize;
	MPI_Comm_size(leaf->comm, &leafCommSize);
    if(leafCommSize > 1) {
		double *ref_points = new double [numof_ref_points*dim];
		#pragma omp parallel for
		for(int i = 0; i < numof_ref_points*dim; i++)
			ref_points[i] = leaf->data->X[i];

		pair<double, long> *kneighbors;
        if(numof_query_points > 0) {
			kneighbors = dist_directKQuery( ref_points, &(redistQuery->X[0]), glb_ref_ids,
											numof_ref_points, numof_query_points,
														k, dim, leaf->comm );
		}
		delete [] ref_points;

		//Pack neighbors into array of (queryID, distance, refID) triples.
		#pragma omp parallel for
		for(int i = 0; i < numof_query_points; i++) {
			for(int j = 0; j < k; j++) {
				triple<long, double, long> *currNeigh;
				currNeigh = &(tNeighbors[i*k+j]);
				currNeigh->first = glb_ref_ids[i];		//redistQuery->gids[i];
				currNeigh->second = kneighbors[i*k+j].first;
				currNeigh->third = kneighbors[i*k+j].second;
			}
		}

		delete [] kneighbors;
	}
    else {	// leafCommSize == 1
		vector< pair<double, long> > *kneighbors = new vector< pair<double, long> >;
	    if(numof_query_points > 0) {
			int flag_stree_r = 0;
			if(leaf->options.flag_r == 2) flag_stree_r = 2;
			stTreeSearch_rkdt_a2a_me(redistQuery, k, flag_stree_r,
									 max_points, max_tree_level - leaf->level,
									 kneighbors);

			// redistQuery has been deleted inside the above function
			// since redistQuery point to leaf->data, that means leaf->data has been released. set it to NULL then
			leaf->data = NULL;
		}

		//Pack neighbors into array of (queryID, distance, refID) triples.
		#pragma omp parallel for
		for(int i = 0; i < numof_ref_points; i++) {
			for(int j = 0; j < k; j++) {
				triple<long, double, long> *currNeigh;
				currNeigh = &(tNeighbors[i*k+j]);
				currNeigh->first = glb_ref_ids[i];	//redistQuery->gids[i];
				currNeigh->second = (*kneighbors)[i*k+j].first;
				currNeigh->third = (*kneighbors)[i*k+j].second;
			}
		}

		delete kneighbors;
	}	// end else
	delete [] glb_ref_ids;

	#if STAGE_OUTPUT_VERBOSE
		MPI_Barrier(MPI_COMM_WORLD);
        if(worldrank == 0) {
			cout<<"  = Query: find nn done! -> "<<omp_get_wtime() - stage_t
				<<" leafCommSize: "<<leafCommSize
				<<endl;
		}
		stage_t = omp_get_wtime();
	#endif

    //Sort array of triples and transimit to appropriate home processes.
    if(totalneighbors > 0) omp_par::merge_sort(tNeighbors, &(tNeighbors[totalneighbors]));


    #if COMM_TIMING_VERBOSE
        MPI_Barrier(MPI_COMM_WORLD);
        start_t = omp_get_wtime();
    #endif


    int *sendcounts = new int[worldsize];
    int *rcvcounts = new int[worldsize];
    int *senddisp = new int[worldsize];
    int *rcvdisp = new int[worldsize];
    #pragma omp parallel for
    for(int i = 0; i < worldsize; i++) sendcounts[i] = 0;
    //for(int i = 0; i < totalneighbors; i++) sendcounts[ knn::lsh::idToHomeRank(tNeighbors[i].first, ppn, worldsize) ]++;
    for(int i = 0; i < totalneighbors; i++) sendcounts[ knn::lsh::idToHomeRank(tNeighbors[i].first, inscan_numof_query_points_per_rank) ]++;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_CALL(MPI_Alltoall(sendcounts, 1, MPI_INT, rcvcounts, 1, MPI_INT, MPI_COMM_WORLD));
    omp_par::scan(sendcounts, senddisp, worldsize);
    omp_par::scan(rcvcounts, rcvdisp, worldsize);
    int rcvneighs = rcvdisp[worldsize-1]+rcvcounts[worldsize-1];
    assert(rcvneighs == homepoints*k);
    triple<long, double, long> *homeneighbors = new triple<long, double, long>[rcvneighs];
    MPI_Datatype tripledata;
    MPI_Type_contiguous(sizeof(triple<long, double, long>), MPI_BYTE, &tripledata);
    MPI_Type_commit(&tripledata);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_CALL(MPI_Alltoallv(tNeighbors, sendcounts, senddisp, tripledata, homeneighbors,
			 rcvcounts, rcvdisp, tripledata, MPI_COMM_WORLD));
    delete[] tNeighbors;

#if ALL_TO_ALL_VERBOSE
  int tmpsendsum = 0, tmprecvsum = 0;
  for(int i = 0; i < worldsize; i++)
	  tmpsendsum += sendcounts[i];
  for(int i = 0; i < worldsize; i++)
	  tmprecvsum += rcvcounts[i];
  cout<<"worldrank: "<<worldrank<<" recvcount: "<<tmprecvsum<<" sendcount: "<<tmpsendsum<<endl;
#endif



#if COMM_TIMING_VERBOSE
  Repartition_Query_T_ += omp_get_wtime() - start_t;
#endif

  omp_par::merge_sort(homeneighbors, &(homeneighbors[rcvneighs]));

	#if STAGE_OUTPUT_VERBOSE
		if(worldrank == 0) {
			cout<<"  = Query: repartition results to home process done! -> "<<omp_get_wtime() - stage_t<<endl;
		}
		stage_t = omp_get_wtime();
	#endif


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

  MPI_Type_free(&tripledata);
  //delete redistQuery;
  delete[] homeneighbors;  
  delete[] sendcounts;
  delete[] rcvcounts;
  delete[] senddisp;   
  delete[] rcvdisp;
	
	#if STAGE_OUTPUT_VERBOSE
		if(worldrank == 0) {
			cout<<"  = Query: prepare output done! -> "<<omp_get_wtime() - stage_t<<endl;
		}
		stage_t = omp_get_wtime();
	#endif


}
*/



void bintree::queryK_Greedy( pbinData inData, long rootNpoints,
							 pbinNode searchNode, int k,
							 int traverse_type, int max_points, int max_tree_level,
							 vector<long> *queryIDs,
							 vector< pair<double, long> > *kNN)
{

	double start_t, end_t, stage_t;

	start_t = omp_get_wtime();
	stage_t = omp_get_wtime();

	int worldsize, worldrank;
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);

    if(_RKDT_DEBUG_) {
        cout<<"\t\tgreedy_query: rank "<<worldrank<<" enter queryK_Greedy function"
            <<endl;
    }

	int dim = inData->dim;
	int numof_query_points = inData->numof_points;
	int global_numof_query_points;
	MPI_CALL(MPI_Allreduce( &numof_query_points, &global_numof_query_points, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD ));
	int ppn = global_numof_query_points/worldsize; //Number of points per node
	//int homepoints = (worldrank==worldsize-1) ? ppn+global_numof_query_points%ppn : ppn; //Number of query points "owned" by each process
    long divd = global_numof_query_points / worldsize;
    long rem = global_numof_query_points % worldsize;
    int homepoints = worldrank < rem ? divd+1 : divd;


    if(_RKDT_DEBUG_) {
        cout<<"\t\tgreedy_query: rank "<<worldrank<<" dim = "<<dim
            <<", numof_query_points = "<<numof_query_points
            <<", global_numof_query_points = "<<global_numof_query_points
            <<", ppn = "<<ppn
            <<", homepoints = "<<homepoints
            <<", traverse_type = "<<traverse_type
            <<endl;
    }

	#if STAGE_OUTPUT_VERBOSE
		int max_niq, min_niq;
		//MPI_Reduce(&numof_query_points, &max_niq, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
		//MPI_Reduce(&numof_query_points, &min_niq, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
        if(worldrank == 0) {
			cout<<"  = Query: enter func "
				//<<"-> Nq: min="<<min_niq<<" max="<<max_niq
				<<endl;
		}
	#endif

	pbinData redistQuery;
	pbinNode leaf;
	if(traverse_type == 0) {			// rkdt: new repartition
		redistQuery = new binData();
		double *newX;
		long *newgids;
		long newN;

        if(_RKDT_DEBUG_) {
            cout<<"\t\tgreedy_query: rank "<<worldrank<<" before repartition"
                <<endl;
        }

		repartitionQueryData(&(inData->X[0]), &(inData->gids[0]), inData->numof_points, inData->dim,
							 searchNode, &newX, &newgids, newN);

        if(_RKDT_DEBUG_) {
            cout<<"\t\tgreedy_query: rank "<<worldrank<<" new repartition done"<<endl;
        }


		redistQuery->dim = inData->dim;
		redistQuery->numof_points = newN;
		redistQuery->X.resize(newN*dim);
		#pragma omp parallel for
		for(int i = 0; i < newN*dim; i++)
			redistQuery->X[i] = newX[i];
		redistQuery->gids.resize(newN);
		#pragma omp parallel for
		for(int i = 0; i < newN; i++)
			redistQuery->gids[i] = newgids[i];
		delete [] newX;
		delete [] newgids;

		leaf = searchNode;
		while(leaf->kid != NULL)
			leaf = leaf->kid;
	}
    else if(traverse_type == 1) {	// rkdt: old repartition
		bintree::distributeToNearestLeaf(inData, searchNode, &redistQuery, &leaf);

        if(_RKDT_DEBUG_) {
            cout<<"\t\tgreedy_query: rank "<<worldrank<<" old repartition done"<<endl;
        }
    }
    else {						// rsmt
		bintree::distributeToSampledLeaf(inData, searchNode, &redistQuery, &leaf);
	}

	#if STAGE_OUTPUT_VERBOSE
		MPI_Barrier(MPI_COMM_WORLD);
		if(worldrank == 0) cout<<"  = Query: distribute to leaf done! -> "<<omp_get_wtime() - stage_t<<endl;
		stage_t = omp_get_wtime();
	#endif

	#if TREE_DEBUG_VERBOSE
	for(int r = 0; r < worldsize; r++) {
		if(r == worldrank) {
			for(int i = 0; i < redistQuery->numof_points; i++) {
				cout<<"rank "<<worldrank<<": id="<<redistQuery->gids[i]<<" - ";
				for(int j = 0; j < redistQuery->dim; j++)
					cout<<redistQuery->X[i*redistQuery->dim+j]<<" ";
				cout<<endl;
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
	#endif

	numof_query_points = redistQuery->numof_points;
	int numof_ref_points = leaf->data->numof_points;


	#if LOAD_BALANCE_VERBOSE
		int max_numof_query_points, min_numof_query_points, avg_numof_query_points;
		MPI_Reduce(&numof_query_points, &max_numof_query_points, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&numof_query_points, &min_numof_query_points, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&numof_query_points, &avg_numof_query_points, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		if(worldrank == 0) {
			cout<<"    - numof_query_points: "
				<<" min: "<<min_numof_query_points
				<<" max: "<<max_numof_query_points
				<<" avg: "<<avg_numof_query_points / worldsize
				<<endl;
		}
	#endif
  
	int totalneighbors = k*numof_query_points;
	triple<long, double, long> *tNeighbors = new triple<long, double, long>[totalneighbors];
  
	#if STAGE_OUTPUT_VERBOSE
		if(worldrank == 0) {
			cout<<"  = Query: get leaf points done! -> "<<omp_get_wtime() - stage_t<<endl;
		}
		stage_t = omp_get_wtime();
	#endif

	//Find k nearest neighbors within this node.
	int leafCommSize;
	MPI_Comm_size(leaf->comm, &leafCommSize);
	if(leafCommSize > 1) {
		double *ref_points = (double*)malloc(numof_ref_points*dim*sizeof(double));
		#pragma omp parallel for
		for(int i = 0; i < numof_ref_points*dim; i++)
			ref_points[i] = leaf->data->X[i];
		long *glb_ref_ids = (long*)malloc(numof_ref_points*sizeof(long));
		#pragma omp parallel for
		for(int i = 0; i < numof_ref_points; i++) 
			glb_ref_ids[i] = leaf->data->gids[i];
 	
		pair<double, long> *kneighbors;
		if(numof_query_points > 0) {
			kneighbors = dist_directKQuery( ref_points, &(redistQuery->X[0]), glb_ref_ids,
														numof_ref_points, numof_query_points,
														k, dim, leaf->comm );
		}

		free(ref_points);
		free(glb_ref_ids);
		//Pack neighbors into array of (queryID, distance, refID) triples. 
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
		delete [] kneighbors;
	}	// end if (leafCommSize > 1)

	else {	// leafCommSize == 1
		vector< pair<double, long> > *kneighbors = new vector< pair<double, long> >;
		if(numof_query_points > 0) {
			int flag_stree_r = 0;
			if(leaf->options.flag_r == 2) flag_stree_r = 2;
			stTreeSearch_rkdt_me( leaf->data, redistQuery,
								  k, flag_stree_r,
							      max_points, max_tree_level - leaf->level,
								  kneighbors);
			// since leaf->data has been released inside the above function. set it to NULL then
			leaf->data = NULL;
		}

		//Pack neighbors into array of (queryID, distance, refID) triples. 
		#pragma omp parallel for
		for(int i = 0; i < numof_query_points; i++) {
			for(int j = 0; j < k; j++) {
				triple<long, double, long> *currNeigh;
				currNeigh = &(tNeighbors[i*k+j]);
				currNeigh->first = redistQuery->gids[i];
				currNeigh->second = (*kneighbors)[i*k+j].first;
				currNeigh->third = (*kneighbors)[i*k+j].second;
			}
		}

		delete kneighbors;
	}	// end else
 	
	#if STAGE_OUTPUT_VERBOSE
		if(worldrank == 0) {
			cout<<"  = Query: find nn done! -> "<<omp_get_wtime() - stage_t
				<<" leafCommSize: "<<leafCommSize
				<<endl;
		}
		stage_t = omp_get_wtime();
	#endif

  
	//Sort array of triples and transimit to appropriate home processes.
	if(totalneighbors > 0) omp_par::merge_sort(tNeighbors, &(tNeighbors[totalneighbors]));
  
	#if COMM_TIMING_VERBOSE
		MPI_Barrier(MPI_COMM_WORLD);
		start_t = omp_get_wtime();
	#endif



	int *sendcounts = new int[worldsize];
	int *rcvcounts = new int[worldsize];
	int *senddisp = new int[worldsize];
	int *rcvdisp = new int[worldsize];
	#pragma omp parallel for
	for(int i = 0; i < worldsize; i++) sendcounts[i] = 0;
	for(int i = 0; i < totalneighbors; i++) {
        //sendcounts[ knn::lsh::idToHomeRank(tNeighbors[i].first, ppn, worldsize) ]++;
        sendcounts[ knn::home_rank(global_numof_query_points, worldsize, tNeighbors[i].first) ]++;
    }
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_CALL(MPI_Alltoall(sendcounts, 1, MPI_INT, rcvcounts, 1, MPI_INT, MPI_COMM_WORLD));
	omp_par::scan(sendcounts, senddisp, worldsize);
	omp_par::scan(rcvcounts, rcvdisp, worldsize);
	int rcvneighs = rcvdisp[worldsize-1]+rcvcounts[worldsize-1];

	#if TREE_DEBUG_VERBOSE
	cout<<"rank: "<<worldrank<<" totalneighbors="<<totalneighbors<<endl;
	for(int i = 0; i < totalneighbors; i++)
		cout<<"rank: "<<worldrank<<" "<<tNeighbors[i].first<<" "<<tNeighbors[i].third<<endl;
	#endif


	#if STAGE_OUTPUT_VERBOSE
		if(worldrank == 0) {
			cout<<"  = Query: "
				<<"rcvneighs="<<rcvneighs
				<<" homepoints="<<homepoints
				<<endl;
		}
	#endif

	assert(rcvneighs == homepoints*k);
	triple<long, double, long> *homeneighbors = new triple<long, double, long>[rcvneighs];
	MPI_Datatype tripledata;
	MPI_Type_contiguous(sizeof(triple<long, double, long>), MPI_BYTE, &tripledata);
	MPI_Type_commit(&tripledata);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_CALL(MPI_Alltoallv(tNeighbors, sendcounts, senddisp, tripledata, homeneighbors,
			 rcvcounts, rcvdisp, tripledata, MPI_COMM_WORLD));
	delete[] tNeighbors;
	MPI_Type_free(&tripledata);
 
	#if COMM_TIMING_VERBOSE
		Repartition_Query_T_ += omp_get_wtime() - start_t;
	#endif

	omp_par::merge_sort(homeneighbors, &(homeneighbors[rcvneighs]));
  
	#if STAGE_OUTPUT_VERBOSE
		if(worldrank == 0) cout<<"  = Query: repartition results to home process done! -> "<<omp_get_wtime() - stage_t<<endl;
		stage_t = omp_get_wtime();
	#endif


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
	//delete leaf;	

	#if STAGE_OUTPUT_VERBOSE
		if(worldrank == 0) cout<<"  = Query: prepare output done! -> "<<omp_get_wtime() - stage_t<<endl;
		stage_t = omp_get_wtime();
	#endif

}


// rkdt all to all
pair<double, double> bintree::knnTreeSearch_RandomRotation_a2a( pbinData refData,
							int k,
							int numof_iterations,
							treeParams params,
							int flag_r, int flag_c,
							//output
							vector<long> & queryIDs,
							vector< pair<double, long> >* &kNN,
                            double tol_hit,
                            double tol_err)
{
	double start_t, end_t, dbg_t, max_t, min_t, avg_t;
    double res_hit_rate = -1.0, res_relative_error = -1.0;

	int worldsize, worldrank;
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);

	srand((unsigned)time(NULL)+worldrank);

	int dim = refData->dim;
	int numof_ref_points = refData->numof_points;
	int numof_query_points = numof_ref_points;  //queryData->X.size() / dim;
	long global_numof_ref_points, dummy_numof_ref_points = numof_ref_points;
	MPI_CALL(MPI_Allreduce( &dummy_numof_ref_points, &global_numof_ref_points, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD ));
	long global_numof_query_points = global_numof_ref_points;
	int ppn = global_numof_query_points/worldsize; //Number of points per node
	//int homepoints = (worldrank==worldsize-1) ? ppn+global_numof_query_points%ppn : ppn; //Number of query points "owned" by each process
    long divd = global_numof_query_points /worldsize;
    long rem = global_numof_query_points % worldsize;
    long homepoints = worldrank < rem ? divd+1 : divd;


    vector<long> inscan_numof_query_points_per_rank(worldsize);
    vector<int> numof_query_points_per_rank(worldsize);
    MPI_Allgather(&numof_query_points, 1, MPI_INT, &(numof_query_points_per_rank[0]), 1, MPI_INT, MPI_COMM_WORLD);
    inscan_numof_query_points_per_rank[0] = numof_query_points_per_rank[0];
    for(int i = 1; i < inscan_numof_query_points_per_rank.size(); i++)
        inscan_numof_query_points_per_rank[i] = inscan_numof_query_points_per_rank[i-1] + numof_query_points_per_rank[i];

	// ========== used for correctness check ====== //
	vector<long> sampleIDs;
	vector<double> globalKdist;
	vector<long> globalKid;
    if(params.eval_verbose) {
        start_t = omp_get_wtime();
        if(worldrank == 0)
            cout<<"rank "<<worldrank<<" start find exact knn of sample points "<<endl;
		get_sample_info(&(refData->X[0]), &(refData->X[0]), &(refData->gids[0]), &(refData->gids[0]),
				numof_ref_points, numof_query_points, refData->dim, k,
				sampleIDs, globalKdist, globalKid);
        if(worldrank == 0)
            cout<<"rank "<<worldrank<<" find exact knn of sample points done "<<omp_get_wtime()-start_t<<endl;
	}
	// =========================================== //
    if(_RKDT_ALL2ALL_DEBUG_) {
        MPI_Barrier( MPI_COMM_WORLD );
        if(worldrank == 0) cout<<"    sample validation points done "<<endl;
    }

    if(worldsize == 1) {
        // directly call the shared memory knn search
        queryIDs.resize(numof_query_points);
        for(int i = 0; i < numof_query_points; i++)
            queryIDs[i] = refData->gids[i];

        double srkdt_t = omp_get_wtime();
        if(params.eval_verbose) {
            find_knn_srkdt(refData, refData, k, params.max_points_per_node,
                        params.max_tree_level, numof_iterations, kNN,
                        queryIDs, sampleIDs, globalKdist, globalKid);
        }
        else {
            find_knn_srkdt(refData, refData, k, params.max_points_per_node,
                        params.max_tree_level, numof_iterations, kNN);
        }
        Direct_Kernel_T_ = omp_get_wtime()-srkdt_t;

        //if(params.eval_verbose) {
		//	double hit_rate = 0.0, relative_error = 0.0;
		//	int nmiss = 0;
		//	verify(sampleIDs, globalKdist, globalKid,
        //              queryIDs, *kNN, nmiss, hit_rate, relative_error);
		//	cout<<"    + shared memory knn search "
		//		<<": "<<sampleIDs.size()
        //		<<" samples -- hit rate "<< hit_rate << "%"
		//		<<"  relative error "<< relative_error << "%" << endl;
	    //    res_hit_rate = hit_rate;
        //    res_relative_error = relative_error;
        //}

        return make_pair<double, double>(res_hit_rate, res_relative_error);
    }

    if(worldrank == 0) cout<<endl<<"*************** total iters = "<<numof_iterations<<" ***************** "<<endl;

	double hit_rate = 0.0, relative_error = 1.0;

    for(int iter = 0; iter < numof_iterations; iter++) {

        if(hit_rate > tol_hit || relative_error < tol_err) {
            break;
        }

        double iter_t = omp_get_wtime();

		#if OVERALL_TREE_TIMING_VERBOSE
			MPI_Barrier(MPI_COMM_WORLD);
		#endif

		start_t = omp_get_wtime();

		// 1. build the tree
		pbinData ref_iter = new binData();
		ref_iter->Copy(refData);

		MPI_Collective_T_ = 0.0;

		pbinNode root = new binNode();
		root->options.hypertree = params.hypertree;
		root->options.debug_verbose = params.debug_verbose;
		root->options.timing_verbose = params.timing_verbose;
		root->options.pruning_verbose = params.pruning_verbose;
		root->options.splitter = "rkdt";	// use maxVarSplitter
		root->options.flag_r = flag_r;
		root->options.flag_c = flag_c;

		//root->Insert( NULL, params.max_points_per_node, params.max_tree_level,
		//				params.min_comm_size_per_node,
		//				MPI_COMM_WORLD, refData);

	    root->Insert( NULL, params.max_points_per_node, params.max_tree_level,
						params.min_comm_size_per_node,
						MPI_COMM_WORLD, ref_iter);

		#if OVERALL_TREE_TIMING_VERBOSE
			MPI_Collective_Const_T_ += MPI_Collective_T_;
			Tree_Const_T_ += omp_get_wtime() - start_t;
		#endif

		#if RKDT_ITER_VERBOSE
			MPI_Barrier(MPI_COMM_WORLD);
			end_t = omp_get_wtime() - start_t;
			if(worldrank == 0) cout<<"   == tree construction time: "<<end_t<<endl;
		#endif

        if(_RKDT_ALL2ALL_DEBUG_) {
            MPI_Barrier( MPI_COMM_WORLD );
            if(worldrank == 0) cout<<"\t\ttree construction done"<<endl;
        }

		//release refdata
		delete ref_iter;

		// 2. search the tree
		#if OVERALL_TREE_TIMING_VERBOSE
			MPI_Barrier(MPI_COMM_WORLD);
		#endif

		MPI_Collective_T_ = 0.0;

		start_t = omp_get_wtime();

		bool do_query = true;
        if(iter == 0 && do_query) {
			// use nearest traverse strategy
			int flag_stree_r = 0;
			if(flag_r == 2)	flag_stree_r = 2;
			bintree::queryK_Greedy_a2a(global_numof_ref_points, root, k,
									   params.max_points_per_node, params.max_tree_level,
									   &queryIDs, kNN);

            if(_KNN_MERGE_DEBUG_) {
                MPI_Barrier(MPI_COMM_WORLD);
                if(worldrank == 0) {
                    cout<<"============ "<<iter<<" iteration =========="<<endl;
                    cout<<"kNN_iter: "<<endl;
                }
                for(int r = 0; r < worldsize; r++) {
                    if(r == worldrank) {
                        for(int i = 0; i < queryIDs.size(); i++) {
                            cout<<"rank "<<worldrank<<" ["<<queryIDs[i]<<"] ";
                            for(int j = 0; j < k; j++) {
                                cout<<"("<<(*kNN)[i*k+j].second<<","<<(*kNN)[i*k+j].first<<") ";
                            }
                            cout<<endl;
                        }
                        cout<<endl;
                    }
                    cout.flush();
                    MPI_Barrier( MPI_COMM_WORLD);
                }
            }

        }
		else if(do_query)
        {
			if(params.debug_verbose == 7 && worldrank == 0) dbg_t = omp_get_wtime();
			vector<long> queryIDs_iter;
			vector< pair<double, long> > kNN_iter;
			bintree::queryK_Greedy_a2a(global_numof_ref_points, root, k,
									   params.max_points_per_node, params.max_tree_level,
									   &queryIDs_iter, &kNN_iter);

			vector< pair<double, long> > *tmpkNN = new vector< pair<double, long> >;

            if(_KNN_MERGE_DEBUG_) {
                MPI_Barrier(MPI_COMM_WORLD);
                if(worldrank == 0) {
                    cout<<"============ "<<iter<<" iteration =========="<<endl;
                    cout<<"kNN_iter: "<<endl;
                }
                for(int r = 0; r < worldsize; r++) {
                    if(r == worldrank) {
                        for(int i = 0; i < queryIDs.size(); i++) {
                            cout<<"rank "<<worldrank<<" ["<<queryIDs[i]<<"] ";
                            for(int j = 0; j < k; j++) {
                                cout<<"("<<kNN_iter[i*k+j].second<<","<<kNN_iter[i*k+j].first<<") ";
                            }
                            cout<<endl;
                        }
                        cout<<endl;
                    }
                    cout.flush();
                    MPI_Barrier( MPI_COMM_WORLD);
                }
            }

			knn_merge((*kNN), kNN_iter, homepoints, k, *tmpkNN);

            // see tmpkNN is better than kNN or not
            //int tmp_n = tmpkNN->size() / k;
            //for(int ii = 0; ii < tmp_n; ii++) {
            //    for(int jj = 0; jj < k; jj++) {
            //        if( (*tmpkNN)[ii*k+jj].first > (*kNN)[ii*k+jj].first ) {
            //            cout<<"\terror: tmpknn.first = "<<(*tmpkNN)[ii*k+jj].first
            //                <<" ("<<(*tmpkNN)[ii*k+jj].second<<") > knn.first = "
            //                <<(*kNN)[ii*k+jj].first
            //                <<" ("<<(*kNN)[ii*k+jj].second<<")"<<endl;
            //        }
            //    }
            //}

			delete kNN;
			kNN = tmpkNN;

            if(_KNN_MERGE_DEBUG_) {
                MPI_Barrier(MPI_COMM_WORLD);
                if(worldrank == 0) cout<<"kNN after merge: "<<endl;
                for(int r = 0; r < worldsize; r++) {
                    if(r == worldrank) {
                        for(int i = 0; i < queryIDs.size(); i++) {
                            cout<<"rank "<<worldrank<<" ["<<queryIDs[i]<<"] ";
                            for(int j = 0; j < k; j++) {
                                cout<<"("<<(*kNN)[i*k+j].second<<","<<(*kNN)[i*k+j].first<<") ";
                            }
                            cout<<endl;
                        }
                        cout<<endl;
                    }
                    cout.flush();
                    MPI_Barrier( MPI_COMM_WORLD);
                }
            }

		} else{
		;}// end else

		delete root;

        if(_RKDT_ALL2ALL_DEBUG_) {
            MPI_Barrier( MPI_COMM_WORLD );
            if(worldrank == 0) cout<<"\t\ttree search done"<<endl;
        }


		#if OVERALL_TREE_TIMING_VERBOSE
			MPI_Collective_Query_T_ += MPI_Collective_T_;
			Tree_Search_T_ += omp_get_wtime() - start_t;
		#endif

		#if RKDT_ITER_VERBOSE
		end_t = omp_get_wtime() - start_t;
		MPI_Reduce(&end_t, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&end_t, &min_t, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&end_t, &avg_t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if(worldrank == 0) {
			cout<<"    + min search time: "<<min_t
				<<"  max search time: "<<max_t
				<<"  avg search time: "<<avg_t / worldsize<<endl;
		}
		#endif

        if(params.eval_verbose) {
			//double hit_rate = 0.0, relative_error = 0.0;
			int nmiss = 0;
			verify(sampleIDs, globalKdist, globalKid, queryIDs, *kNN, nmiss, hit_rate, relative_error);
            if(worldrank == 0) {
				cout<<"    + iter "<<iter
					<<": "<<sampleIDs.size()<<" samples -- hit rate "<< hit_rate << "%"
					<<"  relative error "<< relative_error << "%"
                    <<"  elapsed time "<<omp_get_wtime()-iter_t
                    << endl;
			}
            res_hit_rate = hit_rate;
            res_relative_error = relative_error;
		}

	}  // end for(iter < numof_iterations)

	#if RKDT_FINAL_KNN_OUTPUT_VERBOSE
	    for(int i = 0; i < queryIDs.size(); i++) {
			cout<<queryIDs[i]<<" ";
            for(int j = 0; j < k; j++)
				cout<<(*kNN)[i*k+j].second<<" "<<(*kNN)[i*k+j].first<<" ";
			cout<<endl;
		}
	#endif

    return make_pair<double, double>(res_hit_rate, res_relative_error);

}


void bintree::knnTreeSearch_RandomRotation( pbinData refData,
							pbinData queryData,
							int k,
							int numof_iterations,
							treeParams params,
							int flag_r, int flag_c,
							//output
							vector<long> & queryIDs,
							vector< pair<double, long> > *&kNN,
                            double tol_hit, double tol_err)
{

	double start_t, end_t, max_t, min_t, avg_t;

	int worldsize, worldrank;
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);

	srand((unsigned)time(NULL)+worldrank);

	int dim = refData->dim;
	int numof_ref_points = refData->numof_points;
	int numof_query_points = queryData->numof_points;
	int global_numof_query_points;
	int global_numof_ref_points;
	MPI_CALL(MPI_Allreduce( &numof_query_points, &global_numof_query_points, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD ));
	MPI_CALL(MPI_Allreduce( &numof_ref_points, &global_numof_ref_points, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD ));

	int ppn = global_numof_query_points/worldsize; //Number of points per node
	//int homepoints = (worldrank==worldsize-1) ? ppn+global_numof_query_points%ppn : ppn; //Number of query points "owned" by each process
    long divd = global_numof_query_points / worldsize;
    long rem = global_numof_query_points % worldsize;
    long homepoints = worldrank < rem ? divd+1 : divd;

	vector<long> sampleIDs;
	vector<double> globalKdist;
	vector<long> globalKid;
    if(params.eval_verbose) {
		get_sample_info(&(refData->X[0]), &(queryData->X[0]), 
                    &(refData->gids[0]), &(queryData->gids[0]), 
				    numof_ref_points, numof_query_points, refData->dim, k, 
				    sampleIDs, globalKdist, globalKid);
	}


    if(worldsize == 1) {
        // directly call the shared memory knn search
        queryIDs.resize(numof_query_points);
        for(int i = 0; i < numof_query_points; i++)
            queryIDs[i] = queryData->gids[i];
        double srkdt_t = omp_get_wtime();
        find_knn_srkdt(refData, queryData, k, params.max_points_per_node,
                        params.max_tree_level, numof_iterations, kNN);
        Direct_Kernel_T_ = omp_get_wtime() - srkdt_t;

        if(params.eval_verbose) {
			double hit_rate = 0.0, relative_error = 0.0;
			int nmiss = 0;
			verify(sampleIDs, globalKdist, globalKid,
                        queryIDs, *kNN, nmiss, hit_rate, relative_error);
			cout<<"    + shared memory knn search "
				<<": "<<sampleIDs.size()
                <<" samples -- hit rate "<< hit_rate << "%"
				<<"  relative error "<< relative_error << "%" << endl;
		}
        return;
    }


	double hit_rate = 0.0, relative_error = 1.0;

	for(int iter = 0; iter < numof_iterations; iter++) {

        if(hit_rate > tol_hit || relative_error < tol_err) {
            break;
        }


		if(params.debug_verbose == 8 && worldrank == 0) cout<<"  iter "<<iter<<endl;

		#if OVERALL_TREE_TIMING_VERBOSE
			MPI_Barrier(MPI_COMM_WORLD);
		#endif
		start_t = omp_get_wtime();

		// 1. build the tree
		MPI_Collective_T_ = 0.0;
		pbinNode root = new binNode();
		root->options.hypertree = params.hypertree;
		root->options.debug_verbose = params.debug_verbose;
		root->options.timing_verbose = params.timing_verbose;
		root->options.pruning_verbose = params.pruning_verbose;
		root->options.splitter = "rkdt";	// use maxVarSplitter
		root->options.flag_r = flag_r;
		root->options.flag_c = flag_c;
		root->Insert(	NULL, params.max_points_per_node, params.max_tree_level, 
						params.min_comm_size_per_node, MPI_COMM_WORLD, refData);

		#if OVERALL_TREE_TIMING_VERBOSE
			MPI_Collective_Const_T_ += MPI_Collective_T_;
			Tree_Const_T_ += omp_get_wtime() - start_t;
		#endif

		#if RKDT_ITER_VERBOSE
			end_t = omp_get_wtime() - start_t;
			if(worldrank == 0) cout<<"   == tree construction time: "<<end_t<<endl;
		#endif

        if(_RKDT_DEBUG_) {
            cout<<"\t\trkdt: rank "<<worldrank<<" tree construction iter = "<<iter
                <<", elasped time = "<<omp_get_wtime()-start_t
                <<", root->rw.size() = "<<root->rw.size()
                <<endl;
        }

		// 2. search the tree
		#if OVERALL_TREE_TIMING_VERBOSE
			MPI_Barrier(MPI_COMM_WORLD);
		#endif
		start_t = omp_get_wtime();

		MPI_Collective_T_ = 0.0;

		if(iter == 0) {
			// use nearest traverse strategy
			bintree::queryK_Greedy( queryData, global_numof_ref_points, root, k, params.traverse_type,
									params.max_points_per_node, params.max_tree_level,
									&queryIDs, kNN);

			#if RKDT_MERGE_VERBOSE
			if(worldrank == 0) {
				cout<<"iter: "<<iter<<" before merge: "<<endl;
				for(int i = 0; i < queryIDs.size(); i++) {
					cout<<queryIDs[i]<<" ";
					for(int j = 0; j < k; j++) 
						cout<<(*kNN)[i*k+j].second<<" "<<(*kNN)[i*k+j].first<<" ";
					cout<<endl;
				}
				cout<<endl;
			}
			#endif

            if(_RKDT_DEBUG_) {
                cout<<"\t\trkdt: rank "<<worldrank<<" greedy query iter = "<<iter
                    <<", elasped time = "<<omp_get_wtime()-start_t<<endl;
            }

		}
		else {
			vector<long> queryIDs_iter;
			vector< pair<double, long> > kNN_iter;
			bintree::queryK_Greedy( queryData, global_numof_ref_points, root, k, params.traverse_type,
									params.max_points_per_node, params.max_tree_level,
									&queryIDs_iter, &kNN_iter);

			#if RKDT_MERGE_VERBOSE
			if(worldrank == 0) {
				cout<<endl<<"iter: "<<iter<<" before merge: "<<endl;
				for(int i = 0; i < queryIDs_iter.size(); i++) {
					cout<<queryIDs_iter[i]<<" ";
					for(int j = 0; j < k; j++)
						cout<<kNN_iter[i*k+j].second<<" "<<kNN_iter[i*k+j].first<<" ";
					cout<<endl;
				}
			}
			#endif

            if(_RKDT_DEBUG_) {
                cout<<"\t\trkdt: rank "<<worldrank<<" greedy query iter = "<<iter
                    <<", elasped time = "<<omp_get_wtime()-start_t<<endl;
            }

			vector< pair<double, long> > *tmpkNN = new vector< pair<double, long> >;
			knn_merge((*kNN), kNN_iter, homepoints, k, *tmpkNN);

			delete kNN;
			kNN = tmpkNN;

			#if RKDT_MERGE_VERBOSE
			if(worldrank == 0) {
				cout<<"iter: "<<iter<<" after merge: "<<endl;
				for(int i = 0; i < queryIDs.size(); i++) {
					cout<<queryIDs[i]<<" ";
					for(int j = 0; j < k; j++)
						cout<<(*kNN)[i*k+j].second<<" "<<(*kNN)[i*k+j].first<<" ";
					cout<<endl;
				}
			} // end if
			#endif

		} // end else

		delete root;

		#if OVERALL_TREE_TIMING_VERBOSE
			MPI_Collective_Query_T_ += MPI_Collective_T_;
			Tree_Search_T_ += omp_get_wtime() - start_t;
		#endif

		#if RKDT_ITER_VERBOSE
		end_t = omp_get_wtime() - start_t;
		MPI_Reduce(&end_t, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&end_t, &min_t, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&end_t, &avg_t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		if(worldrank == 0) {
			cout<<"    + min search time: "<<min_t
				<<"  max search time: "<<max_t
				<<"  avg search time: "<<avg_t / worldsize<<endl;
		}
		#endif


		if(params.eval_verbose) {
			//double hit_rate = 0.0, relative_error = 0.0;
			int nmiss = 0;
			verify(sampleIDs, globalKdist, globalKid, queryIDs, *kNN, nmiss, hit_rate, relative_error);
			if(worldrank == 0) {
				cout<<"    + iter "<<iter
					<<": "<<sampleIDs.size()<<" samples -- hit rate "<< hit_rate << "%"
					<<"  relative error "<< relative_error << "%" << endl;
			}
		}

	}  // end for(iter < numof_iterations)

	#if RKDT_FINAL_KNN_OUTPUT_VERBOSE
		for(int i = 0; i < queryIDs.size(); i++) {
			cout<<queryIDs[i]<<" ";
			for(int j = 0; j < k; j++) {
				cout<<(*kNN)[i*k+j].second<<" "<<(*kNN)[i*k+j].first<<" ";
			}
			cout<<endl;
		}
	#endif

}


/*
// if some A[i].first == B[j].first, always choose A[i], and remove B[j], the value of B might changed
void bintree::knn_merge( vector< pair<double, long> > &A, vector< pair<double, long> > &B,
						 int n, int k,
						 vector< pair<double, long> > &result)
{
	result.resize(n*k);
	#pragma omp parallel for
    for(int i = 0; i < n; i++) {
		int aloc = i*k;
		int bloc = i*k;
		int resultloc = i*k;

        // check B is garbage or not
        // k >= 3
        bool isgarbage = false;
        if(B[bloc+1].second == 0 && B[bloc+2].second == 0) {
            isgarbage = true;
        }

        if(!isgarbage) {
            for(int j = 0; j < k; j++) {
                if( (A[aloc].second == B[bloc].second) && (bloc == (i+1)*k-1) ) B[bloc] = make_pair(DBL_MAX, -1);
                if( (A[aloc].second == B[bloc].second) && (bloc < (i+1)*k-1) ) bloc++;
                if( A[aloc].first <= B[bloc].first ) {
				    result[resultloc++] = A[aloc++];
			    }
                else {
				    result[resultloc++] = B[bloc++];
			    }
		    }
        } else {
            for(int j = 0; j < k; j++)
                result[resultloc++] = A[aloc++];
        }
	} // end for (i < n)
}
*/

/*
// if some A[i].first == B[j].first, always choose A[i], and remove B[j], the value of B might changed
void bintree::knn_merge( vector< pair<double, long> > &A, vector< pair<double, long> > &B,
						 int n, int k,
						 vector< pair<double, long> > &result)
{
	result.resize(n*k);
	#pragma omp parallel for
    for(int i = 0; i < n; i++) {
		int aloc = i*k;
		int bloc = i*k;
		int resultloc = i*k;

        for(int j = 0; j < k; j++) {
            if( (A[aloc].second == B[bloc].second) && (bloc == (i+1)*k-1) ) B[bloc] = make_pair(DBL_MAX, -1);
            if( (A[aloc].second == B[bloc].second) && (bloc < (i+1)*k-1) ) bloc++;
            if( A[aloc].first <= B[bloc].first ) {
				result[resultloc++] = A[aloc++];
                // for some reason, aloc and bloc can arrive the end but comparison doest not end,
                // so we should set the end element to be an impossible value to make the merge can go further
                if( aloc >= (i+1)*k ) {
                    A[aloc-1] = make_pair(DBL_MAX, -1);
                }
			}
            else {
				result[resultloc++] = B[bloc++];
                if( bloc >= (i+1)*k ) {
                    B[bloc-1] = make_pair(DBL_MAX, -1);
                }
			}
		}
	} // end for (i < n)
}
*/


// ------------- for "rsmt" ----------------
void bintree::knnTreeSearch_RandomSampling( pbinData refData, pbinData queryData,
											int k, int numof_iterations, treeParams params,
											//output
											vector<long> & queryIDs,
											vector< pair<double, long> > &kNN)
{
	double start_t, end_t;
	double max_t, min_t, avg_t;

	int worldsize, worldrank;
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
	
	srand((unsigned)time(NULL)+worldrank);
	
	int dim = refData->dim;
	int numof_ref_points = refData->X.size() / dim;
	int numof_query_points = queryData->X.size() / dim;
	int global_numof_query_points;
	int global_numof_ref_points;
	MPI_CALL(MPI_Allreduce( &numof_query_points, &global_numof_query_points, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD ));
	MPI_CALL(MPI_Allreduce( &numof_ref_points, &global_numof_ref_points, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD ));
		
	// 1. build the tree (use mtree splitter)
	pbinData ref_iter = new binData();
	ref_iter->Copy(refData);
	start_t = omp_get_wtime();
	pbinNode root = new binNode();
	root->options.debug_verbose = params.debug_verbose;
	root->options.timing_verbose = params.timing_verbose;
	root->options.pruning_verbose = params.pruning_verbose;
	//root->options.splitter = params.splitter; // mtree = 0.
	root->options.splitter = "rsmt"; // mtree = 0.
	root->Insert(	NULL, params.max_points_per_node, params.max_tree_level, 
					params.min_comm_size_per_node,
					MPI_COMM_WORLD, ref_iter);
	if(params.debug_verbose == 7 && worldrank == 0) 
		cout<<"tree construction done! - "<<omp_get_wtime() - start_t<<endl;
	if(params.debug_verbose == 8 && worldrank == 0) 
		cout<<"  tree construction - "<<omp_get_wtime() - start_t<<endl;
	if(params.flops_verbose) {
		end_t = omp_get_wtime() - start_t;
		if(worldrank == 0) cout<<"  tree construction time: "<<end_t<<endl;
		start_t = omp_get_wtime();
	}
	
	// 2. traverse the tree several times by sample the nearest kid, and merge the results
	// 2.1 first, do a deterministic search, traverse_type = 0
	if(params.debug_verbose == 8 && worldrank == 0) cout<<"  deterministic search"<<endl;
	bintree::queryK_Greedy( queryData, global_numof_ref_points, 
							root, k, 0, params.max_points_per_node, params.max_tree_level, 
							&queryIDs, &kNN);
	if(params.debug_verbose == 4 && worldrank == 0) {
		cout<<"deterministic search: "<<endl;
		for(int i = 0; i < queryIDs.size(); i++) {
			cout<<queryIDs[i]<<" ";
			for(int j = 0; j < k; j++)
				cout<<kNN[i*k+j].second<<" "<<kNN[i*k+j].first<<" ";
			cout<<endl;
		}
		cout<<endl;
	}
	if(params.debug_verbose == 7 && worldrank == 0) 
		cout<<"deterministic search done! - "<<omp_get_wtime() - start_t<<endl;


	if(params.flops_verbose) {
		end_t = omp_get_wtime() - start_t;
		MPI_Reduce(&end_t, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&end_t, &min_t, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&end_t, &avg_t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		if(worldrank == 0) {
			cout<<"  0th iteration (deterministic search): "<<endl
				<<"    + min search time: "<<min_t
				<<"  max search time: "<<max_t
				<<"  avg search time: "<<avg_t / worldsize<<endl;
		}
		start_t = omp_get_wtime();
	}
	


	// 2.2 do numof_iterations random sampling traverse, traverse_type = 1
	for(int iter = 1; iter < numof_iterations; iter++) {
		if(params.debug_verbose == 8 && worldrank == 0) cout<<"  iter "<<iter<<endl;
		vector<long> queryIDs_iter;
		vector< pair<double, long> > kNN_iter;
		bintree::queryK_Greedy( queryData, global_numof_ref_points, root, k, 2,
								params.max_points_per_node, params.max_tree_level,
								&queryIDs_iter, &kNN_iter);

		if(params.debug_verbose == 7 && worldrank == 0)
			cout<<"iter: "<<iter<<" greedy (random sampling) search done! - "<<omp_get_wtime() - start_t<<endl;

		if(params.debug_verbose == 4 && worldrank == 0) {
			cout<<endl<<"iter: "<<iter<<" before merge: "<<endl;
			for(int i = 0; i < queryIDs_iter.size(); i++) {
				cout<<queryIDs_iter[i]<<" ";
				for(int j = 0; j < k; j++)
					cout<<kNN_iter[i*k+j].second<<" "<<kNN_iter[i*k+j].first<<" ";
				cout<<endl;
			}
		}

		vector< pair<double, long> > tmpkNN;
		knn_merge(kNN, kNN_iter, numof_query_points, k, tmpkNN);

		#pragma omp parallel for
		for(int t = 0; t < tmpkNN.size(); t++)
			kNN[t] = tmpkNN[t];

		if(params.timing_verbose && worldrank == 0)
				cout<<"   iter: "<<iter<<" query time: "<<omp_get_wtime() - start_t<<endl;

		if(params.debug_verbose == 4 && worldrank == 0) {
			cout<<"iter: "<<iter<<" after merge: "<<endl;
			for(int i = 0; i < queryIDs.size(); i++) {
				cout<<queryIDs[i]<<" ";
				for(int j = 0; j < k; j++) {
					cout<<kNN[i*k+j].second<<" "<<kNN[i*k+j].first<<" ";
				}
				cout<<endl;
			}
		} // end if

		if(params.flops_verbose) {
			end_t = omp_get_wtime() - start_t;
			MPI_Reduce(&end_t, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
			MPI_Reduce(&end_t, &min_t, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
			MPI_Reduce(&end_t, &avg_t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			if(worldrank == 0) {
				cout<<"  "<<iter<<"th iteration: "<<endl
					<<"    + min search time: "<<min_t
					<<"  max search time: "<<max_t
					<<"  avg search time: "<<avg_t / worldsize<<endl;
			}
			start_t = omp_get_wtime();
		}
	
	
	}  // end for(iter < numof_iterations)

	delete ref_iter;
	delete root;

	if(params.debug_verbose == 5 && worldrank == 0) {
		for(int i = 0; i < queryIDs.size(); i++) {
			cout<<queryIDs[i]<<" ";
			for(int j = 0; j < k; j++) {
				cout<<kNN[i*k+j].second<<" "<<kNN[i*k+j].first<<" ";
			}
			cout<<endl;
		}
	}
}


// super charging
// Note: To apply super charging, we need to know the suspects of all nearest neighbors 
//	     of a query points, i. e. we should perform an all-to-all knn search at first.
//       At the all-to-all case, ref and query on each process should be the exact same. 
//		 i.e. query = ref, numof_query_points = numof_ref_points in all-to-all knn
// to use supercharging, we assume that ref points are number from rank 0 to rank size-1
// from 0 - numof_points -1
void bintree::superCharging_p2p(	double *ref, int numof_ref_points, int dim, int k,
					vector< pair<double, long> > &all2all_kNN, 
					// output
					vector< pair<double, long> > &sckNN,
					MPI_Comm comm)
{
	int size, rank;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	int max_numof_ref_points, global_numof_ref_points;
	MPI_CALL(MPI_Allreduce(&numof_ref_points, &max_numof_ref_points, 1, MPI_INT, MPI_MAX, comm));
	MPI_CALL(MPI_Allreduce(&numof_ref_points, &global_numof_ref_points, 1, MPI_INT, MPI_SUM, comm));
	int ppn = global_numof_ref_points / size;	// average points per process	

	vector<int> send_count(size);
	vector<int> send_disp(size);
	vector<int> recv_count(size);
	vector<int> recv_disp(size); 
	
	vector<long> send_knn_ids;
	vector<long> recv_knn_ids;
	vector<long> suspect_ids(k*k);
	vector<double> suspect_coords(k*k*dim);
	vector<double> send_suspect_coords;
	send_disp[0] = 0;
	recv_disp[0] = 0;


	vector< pair<double, long> > tmpkNN(all2all_kNN.size());

	if(SC_DEBUG_VERBOSE) {
		max_numof_ref_points = 1;
		cout<<"enter superCharging ... "<<endl;
	}

	for(int i = 0; i < max_numof_ref_points; i++) {

		// 1. request all suspect ids of q's neighbors
		for(int l = 0; l < size; l++) send_count[l] = 0;
		int n_total_recv = 0;
		if(i < numof_ref_points) {
			send_knn_ids.resize(k);			// dynamic allocation, maybe slow
			for(int j = 0; j < k; j++) {
				//int target_rank = knn::lsh::idToHomeRank( all2all_kNN[i*k+j].second, ppn, size );
				int target_rank = knn::home_rank(global_numof_ref_points, size, all2all_kNN[i*k+j].second);
				send_count[ target_rank ]++;
				send_knn_ids[j] = all2all_kNN[i*k+j].second;
			}
			sort(send_knn_ids.begin(), send_knn_ids.end());
		}
		else {
			memset(&(send_count[0]), 0, sizeof(int)*size);
		}
		MPI_Barrier(comm);
		MPI_CALL(MPI_Alltoall( &(send_count[0]), 1, MPI_INT, &(recv_count[0]), 1, MPI_INT, comm ));
		for(int l = 0; l < size; l++) n_total_recv += recv_count[l];
		recv_knn_ids.resize(n_total_recv);		// dynamic allocation, maybe slow
		for(int l = 1; l < size; l++) {
			send_disp[l] = send_disp[l-1] + send_count[l-1];
			recv_disp[l] = recv_disp[l-1] + recv_count[l-1];
		}
		MPI_Barrier(comm);
		MPI_CALL(MPI_Alltoallv( &(send_knn_ids[0]), &(send_count[0]), &(send_disp[0]), MPI_LONG,
					&(recv_knn_ids[0]), &(recv_count[0]), &(recv_disp[0]), MPI_LONG, comm));
		

	if(SC_DEBUG_VERBOSE) {
		cout<<"rank: "<<rank<<" 1. request all suspect's ids of q's neighbors\n";
		cout<<"rank: "<<rank<<" query neighbors: ";
		for(int j = 0; j < k; j++) {
			cout<<all2all_kNN[i*k+j].second<<"-"
				<<knn::home_rank(global_numof_ref_points, size, all2all_kNN[i*k+j].second)<<" ";
		}
		cout<<endl;
		cout<<"rank: "<<rank<<" send_count: ";
		for(int j = 0; j < size; j++)
			cout<<send_count[j]<<" ";
		cout<<endl;
		cout<<"rank: "<<rank<<" send_knn_ids: ";
		for(int j = 0; j < k; j++)
			cout<<send_knn_ids[j]<<" ";
		cout<<endl;
		cout<<"rank: "<<rank<<" recv_knn_ids: ";
		for(int j = 0; j < recv_knn_ids.size(); j++)
			cout<<recv_knn_ids[j]<<" ";
		cout<<endl;
		cout<<"rank: "<<rank<<" recv_count: ";
		for(int j = 0; j < size; j++)
			cout<<recv_count[j]<<" ";
		cout<<endl;
	}
		
		// 2. send suspect ids to the requested process.
		if(i < numof_ref_points) {
			send_knn_ids.resize( k * recv_knn_ids.size() );
			for(int t = 0; t < recv_knn_ids.size(); t++) {
				int local_id = recv_knn_ids[t] - rank*ppn;
				for(int j = 0; j < k; j++) {
					send_knn_ids[t*k+j] = all2all_kNN[ local_id*k+j ].second;
				}
			}
			for(int t = 0; t < size; t++)
				send_count[t] = recv_count[t] * k;
		}
		else {
			memset(&(send_count[0]), 0, sizeof(int)*size);
		}
		MPI_Barrier(comm);
		MPI_CALL(MPI_Alltoall( &(send_count[0]), 1, MPI_INT, &(recv_count[0]), 1, MPI_INT, comm ));
		for(int l = 1; l < size; l++) {
			send_disp[l] = send_disp[l-1] + send_count[l-1];
			recv_disp[l] = recv_disp[l-1] + recv_count[l-1];
		}
		MPI_Barrier(comm);
		MPI_CALL(MPI_Alltoallv( &(send_knn_ids[0]), &(send_count[0]), &(send_disp[0]), MPI_LONG,
					&(suspect_ids[0]), &(recv_count[0]), &(recv_disp[0]), MPI_LONG, comm));

	if(SC_DEBUG_VERBOSE) {
		cout<<endl;
		cout<<"rank: "<<rank<<" 2. send suspect ids to the requested process\n";
		cout<<"rank: "<<rank<<" send_knn_ids: ";
		for(int j = 0; j < send_knn_ids.size(); j++)
			cout<<send_knn_ids[j]<<" ";
		cout<<endl;
		
		cout<<"rank: "<<rank<<" send_count: ";
		for(int j = 0; j < size; j++)
			cout<<send_count[j]<<" ";
		cout<<endl;
		
		cout<<"rank: "<<rank<<" suspect_ids: ";
		for(int j = 0; j < suspect_ids.size(); j++)
			cout<<suspect_ids[j]<<" ";
		cout<<endl;
		cout<<"rank: "<<rank<<" recv_count: ";
		for(int j = 0; j < size; j++)
			cout<<recv_count[j]<<" ";
		cout<<endl;
	}

		// 3. request suspect coordinates
		// + 3.1 remove duplicate
		//for(int l = 0; l < size; l++) send_count[l] = 0;
		int ns = 0;
		memset(&(send_count[0]), 0, sizeof(int)*size);
		if(i < numof_ref_points) {
			sort(suspect_ids.begin(), suspect_ids.end());
			vector<long>::iterator it = unique(suspect_ids.begin(), suspect_ids.end());
			ns = it - suspect_ids.begin();
			for(int j = 0; j < ns; j++) {
				//int target_rank = knn::lsh::idToHomeRank( suspect_ids[j], ppn, size );
				int target_rank = knn::home_rank( suspect_ids[j], ppn, size );
				send_count[ target_rank ]++;
			}
		}
		MPI_Barrier(comm);
		MPI_CALL(MPI_Alltoall( &(send_count[0]), 1, MPI_INT, &(recv_count[0]), 1, MPI_INT, comm ));
		n_total_recv = 0;
		for(int l = 0; l < size; l++) n_total_recv += recv_count[l];
		recv_knn_ids.resize(n_total_recv);		// dynamic allocation, maybe slow
		for(int l = 1; l < size; l++) {
			send_disp[l] = send_disp[l-1] + send_count[l-1];
			recv_disp[l] = recv_disp[l-1] + recv_count[l-1];
		}
		MPI_Barrier(comm);
		MPI_CALL(MPI_Alltoallv( &(suspect_ids[0]), &(send_count[0]), &(send_disp[0]), MPI_LONG,
					&(recv_knn_ids[0]), &(recv_count[0]), &(recv_disp[0]), MPI_LONG, comm));
	
	if(SC_DEBUG_VERBOSE) {
		cout<<endl;
		cout<<"rank: "<<rank<<" 3. request suspects' coordinates\n";
		cout<<"rank: "<<rank<<" suspect_ids: ";
		for(int j = 0; j < ns; j++)
			cout<<suspect_ids[j]<<" ";
		cout<<endl;
		
		cout<<"rank: "<<rank<<" send_count: ";
		for(int j = 0; j < size; j++)
			cout<<send_count[j]<<" ";
		cout<<endl;
		
		cout<<"rank: "<<rank<<" recv_knn_ids: ";
		for(int j = 0; j < recv_knn_ids.size(); j++)
			cout<<recv_knn_ids[j]<<" ";
		cout<<endl;
		
		cout<<"rank: "<<rank<<" recv_count: ";
		for(int j = 0; j < size; j++)
			cout<<recv_count[j]<<" ";
		cout<<endl;
	
	}

		// 4. send suspect coordinates
		if(i < numof_ref_points) {
			send_suspect_coords.resize( dim * recv_knn_ids.size() );
			for(int t = 0; t < recv_knn_ids.size(); t++) {
				int local_id = recv_knn_ids[t] - rank*ppn;
				for(int j = 0; j < dim; j++) {
					send_suspect_coords[t*dim+j] = ref[ local_id*dim+j ];
				}
			}
			for(int t = 0; t < size; t++)
				send_count[t] = recv_count[t] * dim;
		}
		else {
			memset(&(send_count[0]), 0, sizeof(int)*size);
		}
		MPI_Barrier(comm);
		MPI_CALL(MPI_Alltoall( &(send_count[0]), 1, MPI_INT, &(recv_count[0]), 1, MPI_INT, comm ));
		for(int l = 1; l < size; l++) {
			send_disp[l] = send_disp[l-1] + send_count[l-1];
			recv_disp[l] = recv_disp[l-1] + recv_count[l-1];
		}
		MPI_Barrier(comm);
		MPI_CALL(MPI_Alltoallv( &(send_suspect_coords[0]), &(send_count[0]), &(send_disp[0]), MPI_DOUBLE,
					&(suspect_coords[0]), &(recv_count[0]), &(recv_disp[0]), MPI_DOUBLE, comm));
		
	
	if(SC_DEBUG_VERBOSE) {
		cout<<endl;
		cout<<"rank: "<<rank<<" 4. send suspects' coordinates\n";
		cout<<"rank: "<<rank<<" send_suspect_coords: ";
		for(int j = 0; j < recv_knn_ids.size(); j++) {
			for(int t = 0; t < dim; t++)
				cout<<send_suspect_coords[j*dim+t]<<" ";
			cout<<"; ";
		}
		cout<<endl;
		
		cout<<"rank: "<<rank<<" send_count: ";
		for(int j = 0; j < size; j++)
			cout<<send_count[j]<<" ";
		cout<<endl;
		
		cout<<"rank: "<<rank<<" suspect_coords: ";
		for(int j = 0; j < ns; j++) {
			for(int t = 0; t < dim; t++)
				cout<<suspect_coords[j*dim+t]<<" ";
			cout<<"; ";
		}
		cout<<endl;
		
		cout<<"rank: "<<rank<<" recv_count: ";
		for(int j = 0; j < size; j++)
			cout<<recv_count[j]<<" ";
		cout<<endl;
	
	}
	
		// 5. find nn within suspect_coords (suspect_ids)
		// + 5.1 find knn within all suspects;
		if(i < numof_ref_points) {
			if(ns < k) {
				knn::directKQueryLowMem(&(suspect_coords[0]), ref+i*dim, ns, 1, ns, dim, &(tmpkNN[i*k]) );
				for(int ii = 0; ii < ns; ii++)
					tmpkNN[i*k+ii].second = suspect_ids[ tmpkNN[i*k+ii].second ];
				for(int t = ns; t < k; t++)
					sckNN[i*k+t] = make_pair(DBL_MAX, -1);
			}
			else {
				knn::directKQueryLowMem(&(suspect_coords[0]), ref+i*dim, ns, 1, k, dim, &(tmpkNN[i*k]) );
				for(int ii = 0; ii < k; ii++)
					tmpkNN[i*k+ii].second = suspect_ids[ tmpkNN[i*k+ii].second ];
			}
		}

	}	// end for (i < numof_ref_points)
	
	knn_merge(all2all_kNN, tmpkNN, numof_ref_points, k, sckNN);

}







// super charging
// Note: To apply super charging, we need to know the suspects of all nearest neighbors 
//	     of a query points, i. e. we should perform an all-to-all knn search at first.
//       At the all-to-all case, ref and query on each process should be the exact same. 
//		 i.e. query = ref, numof_query_points = numof_ref_points in all-to-all knn
// to use supercharging, we assume that ref points are number from rank 0 to rank size-1
// from 0 - numof_points -1
void bintree::superCharging(	double *ref, int numof_ref_points, int dim, int k,
					vector< pair<double, long> > &all2all_kNN, 
					// output
					vector< pair<double, long> > &sckNN,
					MPI_Comm comm)
{
	int size, rank;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	double start_t, start_t2;
	start_t = omp_get_wtime();

	int max_numof_ref_points, global_numof_ref_points;
	MPI_CALL(MPI_Allreduce(&numof_ref_points, &max_numof_ref_points, 1, MPI_INT, MPI_MAX, comm));
	MPI_CALL(MPI_Allreduce(&numof_ref_points, &global_numof_ref_points, 1, MPI_INT, MPI_SUM, comm));
	int ppn = global_numof_ref_points / size;	// average points per process	
	int max_blocksize;
	if( max_numof_ref_points*k*k*dim*8 > 1e9 ) {
		max_blocksize = 1e9 / (k*k*dim*8);
	}
	else {
		max_blocksize = max_numof_ref_points;
	}

	assert(max_blocksize > 0);
	int nblocks = (int) max_numof_ref_points / max_blocksize;
	int iters = (int) ceil( (double)max_numof_ref_points / (double)max_blocksize );

if(SC_TIMING_VERBOSE) {
	cout<<"rank("<<rank<<") max_blocksize "<<max_blocksize<<" nblocks "<<nblocks<<" iters "<<iters<<endl;
	cout<<"rank("<<rank<<") initialization time: "<<omp_get_wtime() - start_t<<endl;
	start_t = omp_get_wtime();
}

	vector<int> send_count(size);
	vector<int> send_disp(size);
	vector<int> recv_count(size);
	vector<int> recv_disp(size); 
	vector<long> suspect_ids(k*k*max_blocksize);
	vector<long> suspect_ids_clone(k*k*max_blocksize);
	vector<double> suspect_coords(k*k*dim*max_blocksize);
	vector<long> send_knn_ids(k*max_blocksize);
	vector<long> recv_knn_ids;
	vector<long> send_suspect_ids;
	vector<double> send_suspect_coords;
	send_disp[0] = 0;
	recv_disp[0] = 0;
	
	#pragma omp parallel if(suspect_ids.size() > 1000)
	{
		#pragma omp for
		for(int i = 0; i < suspect_ids.size(); i++) {
			suspect_ids[i] = global_numof_ref_points;	
		}
	}
	#pragma omp parallel if(suspect_ids.size() > 1000)
	{
		#pragma omp for
		for(int i = 0; i < send_knn_ids.size(); i++) {
			send_knn_ids[i] = global_numof_ref_points;	
		}
	}

	vector< pair<double, long> > tmpkNN(all2all_kNN.size());
	
	int blocksize = (int)ceil((double)numof_ref_points / (double)iters);
	

if(SC_TIMING_VERBOSE) {
	cout<<"rank("<<rank<<") initialization time (vector allocation): "<<omp_get_wtime() - start_t<<endl;
	start_t = omp_get_wtime();
}
	

	for(int it = 0; it < iters; it++) {
		double *currRef = ref + it*blocksize*dim;
		
		int numof_curr_points = blocksize;
		if( (it == iters -1) && (numof_ref_points % blocksize) ) {
			numof_curr_points = numof_ref_points % blocksize;
		}

if(SC_TIMING_VERBOSE) start_t2 = omp_get_wtime();
	
		// 1. request all suspect ids of q's neighbors
		memset(&(send_count[0]), 0, sizeof(int)*size);
		for(int i = 0; i < numof_curr_points; i++) {
			for(int j = 0; j < k; j++) {
				//int target_rank = knn::lsh::idToHomeRank(all2all_kNN[it*blocksize*k+i*k+j].second, ppn, size);
				int target_rank = knn::home_rank(global_numof_ref_points, size, all2all_kNN[it*blocksize*k+i*k+j].second);
				send_count[target_rank]++;
				send_knn_ids[i*k+j] = all2all_kNN[it*blocksize*k+i*k+j].second;
			}
		}


if(SC_TIMING_VERBOSE) {
	cout<<"  - rank("<<rank<<")"<<" iter("<<it<<") step 1 - send_count time: "<<omp_get_wtime() - start_t2<<endl;
	start_t2 = omp_get_wtime();
}


		int tmpK = numof_curr_points*k;
		pair<long, int> *p_sorted_knn_ids = new pair<long, int> [numof_curr_points*k];
		#pragma omp parallel if(tmpK > 1000)
		{
			#pragma omp for
			for(int j = 0; j < tmpK; j++) {
				p_sorted_knn_ids[j].first = send_knn_ids[j];
				p_sorted_knn_ids[j].second = j;
			}
		}
		omp_par::merge_sort( &(p_sorted_knn_ids[0]), &(p_sorted_knn_ids[tmpK]) );

		//sort(send_knn_ids.begin(), send_knn_ids.end());
		#pragma omp parallel if(tmpK > 1000)
		{
			#pragma omp for
			for(int j = 0; j < tmpK; j++) {
				send_knn_ids[j] = p_sorted_knn_ids[j].first;
			}
		}


if(SC_TIMING_VERBOSE) {
	cout<<"  - rank("<<rank<<")"<<" iter("<<it<<") step 1 - merge_sort time: "<<omp_get_wtime() - start_t2<<endl;
	start_t2 = omp_get_wtime();
}

		MPI_Barrier(comm);
		MPI_CALL(MPI_Alltoall( &(send_count[0]), 1, MPI_INT, &(recv_count[0]), 1, MPI_INT, comm ));
		int n_total_recv = recv_count[0];
		for(int l = 1; l < size; l++) {
			n_total_recv += recv_count[l];
			send_disp[l] = send_disp[l-1] + send_count[l-1];
			recv_disp[l] = recv_disp[l-1] + recv_count[l-1];
		}
		recv_knn_ids.resize(n_total_recv);		// dynamic allocation, maybe slow
		MPI_Barrier(comm);
		MPI_CALL(MPI_Alltoallv( &(send_knn_ids[0]), &(send_count[0]), &(send_disp[0]), MPI_LONG,
					&(recv_knn_ids[0]), &(recv_count[0]), &(recv_disp[0]), MPI_LONG, comm));


if(SC_TIMING_VERBOSE) {
	cout<<"  - rank("<<rank<<")"<<" iter("<<it<<") step 1 - all2allv time: "<<omp_get_wtime() - start_t2<<endl;
	start_t2 = omp_get_wtime();
	MPI_Barrier(MPI_COMM_WORLD);
}

if(SC_TIMING_VERBOSE) {
	cout<<"rank("<<rank<<")"<<" iter("<<it<<") step 1 request suspecd ids time: "<<omp_get_wtime() - start_t<<endl;
	start_t = omp_get_wtime();
}

		if(SC_DEBUG_VERBOSE) {
			cout<<"rank: "<<rank<<" 1. request all suspect's ids of q's neighbors\n";
			cout<<"rank: "<<rank<<" send_count: ";
			for(int j = 0; j < size; j++)
				cout<<send_count[j]<<" ";
			cout<<endl;
			cout<<"rank: "<<rank<<" send_knn_ids: ";
			for(int j = 0; j < send_knn_ids.size(); j++)
				cout<<send_knn_ids[j]<<" ";
			cout<<endl;
			cout<<"rank: "<<rank<<" recv_knn_ids: ";
			for(int j = 0; j < recv_knn_ids.size(); j++)
				cout<<recv_knn_ids[j]<<" ";
			cout<<endl;
			cout<<"rank: "<<rank<<" recv_count: ";
			for(int j = 0; j < size; j++)
				cout<<recv_count[j]<<" ";
			cout<<endl;
		}

		// 2. send suspect ids to the requested process.
		send_suspect_ids.resize( k * recv_knn_ids.size() );
		#pragma omp parallel if(recv_knn_ids.size() > 1000)
		{
			#pragma omp for
			for(int t = 0; t < recv_knn_ids.size(); t++) {
				int local_id = recv_knn_ids[t] - rank*ppn;
				for(int j = 0; j < k; j++) 
					send_suspect_ids[t*k+j] = all2all_kNN[ local_id*k+j ].second;
			}
		}
		for(int t = 0; t < size; t++)
			send_count[t] = recv_count[t] * k;

if(SC_TIMING_VERBOSE) {
	cout<<"  - rank("<<rank<<")"<<" iter("<<it<<") step 2 - send_count time: "<<omp_get_wtime() - start_t2
		<<" recv_knn_ids.size() "<<recv_knn_ids.size()<<endl;
	start_t2 = omp_get_wtime();
}

		MPI_Barrier(comm);
		MPI_CALL(MPI_Alltoall( &(send_count[0]), 1, MPI_INT, &(recv_count[0]), 1, MPI_INT, comm ));
		for(int l = 1; l < size; l++) {
			send_disp[l] = send_disp[l-1] + send_count[l-1];
			recv_disp[l] = recv_disp[l-1] + recv_count[l-1];
		}
		MPI_Barrier(comm);
		MPI_CALL(MPI_Alltoallv( &(send_suspect_ids[0]), &(send_count[0]), &(send_disp[0]), MPI_LONG,
					&(suspect_ids[0]), &(recv_count[0]), &(recv_disp[0]), MPI_LONG, comm));


if(SC_TIMING_VERBOSE) {
	cout<<"  - rank("<<rank<<")"<<" iter("<<it<<") step 2 - all2allv time: "<<omp_get_wtime() - start_t2<<endl;
	start_t2 = omp_get_wtime();
}
		
		#pragma omp parallel if(suspect_ids.size() > 1000)
		{
			#pragma omp for
			for(int j = 0; j < suspect_ids.size(); j++)
				suspect_ids_clone[j] = suspect_ids[j];
		}
		//vector<long> suspect_ids_clone(suspect_ids);


if(SC_TIMING_VERBOSE) {
	cout<<"  - rank("<<rank<<")"<<" iter("<<it<<") step 2 - copy suspect_ids_clone time: "<<omp_get_wtime() - start_t2<<endl;
	start_t2 = omp_get_wtime();
}

if(SC_TIMING_VERBOSE) {
	cout<<"rank("<<rank<<")"<<" iter("<<it<<") step 2 send suspect ids time: "<<omp_get_wtime() - start_t<<endl;
	start_t = omp_get_wtime();
}

		if(SC_DEBUG_VERBOSE) {
			cout<<endl;
			cout<<"rank: "<<rank<<" 2. send suspect ids to the requested process\n";
			cout<<"rank: "<<rank<<" send_suspect_ids: ";
			for(int j = 0; j < send_suspect_ids.size(); j++)
				cout<<send_suspect_ids[j]<<" ";
			cout<<endl;
			cout<<"rank: "<<rank<<" send_count: ";
			for(int j = 0; j < size; j++)
				cout<<send_count[j]<<" ";
			cout<<endl;
			cout<<"rank: "<<rank<<" suspect_ids: ";
			for(int j = 0; j < suspect_ids.size(); j++)
				cout<<suspect_ids[j]<<" ";
			cout<<endl;
			cout<<"rank: "<<rank<<" recv_count: ";
			for(int j = 0; j < size; j++)
				cout<<recv_count[j]<<" ";
			cout<<endl;
		}

		// 3. request suspect coordinates
		// + 3.1 remove duplicate
		//sort(suspect_ids.begin(), suspect_ids.end());
		omp_par::merge_sort( suspect_ids.begin(), suspect_ids.end() );

if(SC_TIMING_VERBOSE) {
	cout<<"  - rank("<<rank<<")"<<" iter("<<it<<") step 3 - sort time: "<<omp_get_wtime() - start_t2<<endl;
	start_t2 = omp_get_wtime();
}
		
		
		vector<long>::iterator it1 = unique(suspect_ids.begin(), suspect_ids.end());
	
		int ns = it1 - suspect_ids.begin();

if(SC_TIMING_VERBOSE) {
	cout<<"  - rank("<<rank<<")"<<" iter("<<it<<") step 3 - unique time: "<<omp_get_wtime() - start_t2<<endl;
	start_t2 = omp_get_wtime();
}
		
		// + 3.2 alltoall communication
		memset(&(send_count[0]), 0, sizeof(int)*size);
		for(int j = 0; j < ns; j++) {
			//int target_rank = knn::lsh::idToHomeRank( suspect_ids[j], ppn, size );
			int target_rank = knn::home_rank(global_numof_ref_points, size, suspect_ids[j]);
			send_count[ target_rank ]++;
		}
		MPI_Barrier(comm);
		MPI_CALL(MPI_Alltoall( &(send_count[0]), 1, MPI_INT, &(recv_count[0]), 1, MPI_INT, comm ));
		n_total_recv = recv_count[0];
		for(int l = 1; l < size; l++) {
			n_total_recv += recv_count[l];
			send_disp[l] = send_disp[l-1] + send_count[l-1];
			recv_disp[l] = recv_disp[l-1] + recv_count[l-1];
		}
		recv_knn_ids.resize(n_total_recv);		// dynamic allocation, maybe slow
		MPI_Barrier(comm);
		MPI_CALL(MPI_Alltoallv( &(suspect_ids[0]), &(send_count[0]), &(send_disp[0]), MPI_LONG,
					&(recv_knn_ids[0]), &(recv_count[0]), &(recv_disp[0]), MPI_LONG, comm));


if(SC_TIMING_VERBOSE) {
	cout<<"  - rank("<<rank<<")"<<" iter("<<it<<") step 3 - alltoall time: "<<omp_get_wtime() - start_t2<<endl;
	start_t2 = omp_get_wtime();
}

if(SC_TIMING_VERBOSE) {
	cout<<"rank("<<rank<<")"<<" iter("<<it<<") step 3 request coord time: "<<omp_get_wtime() - start_t<<endl;
	start_t = omp_get_wtime();
}



		if(SC_DEBUG_VERBOSE) {
			cout<<endl;
			cout<<"rank: "<<rank<<" 3. request suspects' coordinates\n";
			cout<<"rank: "<<rank<<" suspect_ids: ";
			for(int j = 0; j < ns; j++)
				cout<<suspect_ids[j]<<" ";
			cout<<endl;
			cout<<"rank: "<<rank<<" send_count: ";
			for(int j = 0; j < size; j++)
				cout<<send_count[j]<<" ";
			cout<<endl;
			cout<<"rank: "<<rank<<" recv_knn_ids: ";
			for(int j = 0; j < recv_knn_ids.size(); j++)
				cout<<recv_knn_ids[j]<<" ";
			cout<<endl;
			cout<<"rank: "<<rank<<" recv_count: ";
			for(int j = 0; j < size; j++)
				cout<<recv_count[j]<<" ";
			cout<<endl;
		}
		
		// 4. send suspect coordinates
		send_suspect_coords.resize( dim * recv_knn_ids.size() );
		for(int t = 0; t < recv_knn_ids.size(); t++) {
			int local_id = recv_knn_ids[t] - rank*ppn;
			for(int j = 0; j < dim; j++) 
				send_suspect_coords[t*dim+j] = ref[ local_id*dim+j ];
		}
		for(int t = 0; t < size; t++)
			send_count[t] = recv_count[t] * dim;
		MPI_Barrier(comm);
		MPI_CALL(MPI_Alltoall( &(send_count[0]), 1, MPI_INT, &(recv_count[0]), 1, MPI_INT, comm ));
		for(int l = 1; l < size; l++) {
			send_disp[l] = send_disp[l-1] + send_count[l-1];
			recv_disp[l] = recv_disp[l-1] + recv_count[l-1];
		}
		MPI_Barrier(comm);
		MPI_CALL(MPI_Alltoallv( &(send_suspect_coords[0]), &(send_count[0]), &(send_disp[0]), MPI_DOUBLE,
					&(suspect_coords[0]), &(recv_count[0]), &(recv_disp[0]), MPI_DOUBLE, comm));

if(SC_TIMING_VERBOSE) {
	cout<<"rank("<<rank<<")"<<" iter("<<it<<") step 4 send coord time: "<<omp_get_wtime() - start_t<<endl;
	start_t = omp_get_wtime();
}


		if(SC_DEBUG_VERBOSE) {
			cout<<endl;
			cout<<"rank: "<<rank<<" 4. send suspects' coordinates\n";
			cout<<"rank: "<<rank<<" send_suspect_coords: ";
			for(int j = 0; j < recv_knn_ids.size(); j++) {
				for(int t = 0; t < dim; t++)
					cout<<send_suspect_coords[j*dim+t]<<" ";
				cout<<"; ";
			}
			cout<<endl;
			cout<<"rank: "<<rank<<" send_count: ";
			for(int j = 0; j < size; j++)
				cout<<send_count[j]<<" ";
			cout<<endl;
			cout<<"rank: "<<rank<<" suspect_coords: ";
			for(int j = 0; j < ns; j++) {
				for(int t = 0; t < dim; t++)
					cout<<suspect_coords[j*dim+t]<<" ";
				cout<<"; ";
			}
			cout<<endl;
			cout<<"rank: "<<rank<<" recv_count: ";
			for(int j = 0; j < size; j++)
				cout<<recv_count[j]<<" ";
			cout<<endl;
		}

		// 5. find nn within suspects for every point
		// 5.1 construct a hash map to find suspect of each query point
		//unordered_map<long, int> suspect_local_map;	// suspect id to local id in arr suspect_coords
		//suspect_local_map.reserve(suspect_ids.size());
		map<long, int> suspect_local_map;	// suspect id to local id in arr suspect_coords
		for(int j = 0; j < ns; j++)
			suspect_local_map[ suspect_ids[j] ] = j;
	
		if(SC_DEBUG_VERBOSE) {
			cout<<"rank: "<<rank<<" 5. 1  map\n";
			cout<<"rank: "<<rank<<" suspect_local_map: ";
			for(map<long, int>::iterator iitt = suspect_local_map.begin(); iitt != suspect_local_map.end(); iitt++) {
				cout<<(*iitt).first<<"-"<<(*iitt).second<<"  ";
			}
			cout<<endl;
		}


if(SC_TIMING_VERBOSE) {
	cout<<"rank("<<rank<<")"<<" iter("<<it<<") step 5.1 construct map time: "<<omp_get_wtime() - start_t<<endl;
	start_t = omp_get_wtime();
}



		// 5.2 rearrange suspect ids in the order of local query ids
		vector<long> suspect_ids_in_order(numof_curr_points*k*k);
		#pragma omp parallel if(numof_curr_points*k > 1000)
		{
			#pragma omp for
			for(int i = 0; i < numof_curr_points*k; i++) {
				for(int j = 0; j < k; j++)
					suspect_ids_in_order[ p_sorted_knn_ids[i].second*k+j] = suspect_ids_clone[ i*k+j ];
			}
		}
	

if(SC_TIMING_VERBOSE) {
	cout<<"rank("<<rank<<")"<<" iter("<<it<<") step 5.2 rearrange suspect: "<<omp_get_wtime() - start_t<<endl;
	start_t = omp_get_wtime();
}


		if(SC_DEBUG_VERBOSE) {
			cout<<"rank: "<<rank<<" 5. 2  rearrange suspect\n";
			cout<<"rank: "<<rank<<" suspect_ids_clone: ";
			for(int j = 0; j < suspect_ids_clone.size(); j++) {
					cout<<suspect_ids_clone[j]<<" ";
			}
			cout<<endl;
			cout<<"rank: "<<rank<<" p_sorted_knn_ids: ";
			for(int j = 0; j < numof_curr_points*k; j++) {
					cout<<p_sorted_knn_ids[j].second<<" ";
			}
			cout<<endl;
			
			cout<<"rank: "<<rank<<" suspect_ids_in_order: ";
			for(int j = 0; j < suspect_ids_in_order.size(); j++) {
					cout<<suspect_ids_in_order[j]<<" ";
			}
			cout<<endl;
		}

		delete [] p_sorted_knn_ids;
	

		// 5.3 find nn for every query
		int pos_offset = it*blocksize*k;
		
		//vector<double> candidates_coords(k*k*dim);
		//vector<long> candidates_ids(k*k);
		//double *tmpdiff = new double [dim];
		//pair<double, long> * tmpdist = new pair<double, long> [k*k];
		

		#pragma omp parallel if(numof_curr_points > 500)
		{
			vector<double> candidates_coords(k*k*dim);
			vector<long> candidates_ids(k*k);
			double *tmpdiff = new double [dim];
			pair<double, long> * tmpdist = new pair<double, long> [k*k];
			
			#pragma omp for
			for(int i = 0; i < numof_curr_points; i++) {
				//vector<double> candidates_coords(k*k*dim);
				//vector<long> candidates_ids(k*k);
			
				copy( suspect_ids_in_order.begin()+i*k*k, suspect_ids_in_order.begin()+(i+1)*k*k, candidates_ids.begin() );
			
				sort(candidates_ids.begin(), candidates_ids.end());
				vector<long>::iterator it2 = unique(candidates_ids.begin(), candidates_ids.end());
				int nd = it2 - candidates_ids.begin();
			//int nd = k*k;
				for(int j = 0; j < nd; j++) {
					for(int t = 0; t < dim; t++) 
						candidates_coords[j*dim+t] = suspect_coords[ suspect_local_map[candidates_ids[j]]*dim+t ];
				}
	
				/*if(nd < k) {
					knn::directKQueryLowMem(&(candidates_coords[0]), currRef+i*dim, nd, 1, nd, dim, &(tmpkNN[pos_offset+i*k]) );
					for(int ii = 0; ii < nd; ii++)
						tmpkNN[pos_offset+i*k+ii].second = candidates_ids[ tmpkNN[pos_offset+i*k+ii].second ];
					for(int t = nd; t < k; t++)
						tmpkNN[pos_offset+i*k+t] = make_pair(DBL_MAX, -1);
				}
				else {
					knn::directKQueryLowMem(&(candidates_coords[0]), currRef+i*dim, nd, 1, k, dim, &(tmpkNN[pos_offset+i*k]) );
					for(int ii = 0; ii < k; ii++)
						tmpkNN[pos_offset+i*k+ii].second = candidates_ids[ tmpkNN[pos_offset+i*k+ii].second ];
				}*/

				bintree::find_knn_single_query(&(candidates_coords[0]), currRef+i*dim, &(candidates_ids[0]), 
												nd, dim, k, &(tmpkNN[pos_offset+i*k]), 
												tmpdiff, tmpdist);
				//bintree::find_knn_single_query(&(candidates_coords[0]), currRef+i*dim, &(candidates_ids[0]), 
				//								nd, dim, k, &(tmpkNN[pos_offset+i*k]) );
				
			}	// end for (i < numof_curr_points)
	
			delete [] tmpdiff;
			delete [] tmpdist;
		}

if(SC_TIMING_VERBOSE) {
	cout<<"rank("<<rank<<")"<<" iter("<<it<<") step 5.3 find nn time: "<<omp_get_wtime() - start_t<<endl;
	start_t = omp_get_wtime();
}



	} // end for (it < iters)

	if(SC_DEBUG_VERBOSE) {
		for(int i = 0; i < numof_ref_points; i++) {
			cout<<"rank "<<rank<<" i: "<<i<<" - ";
			for(int j = 0; j < k; j++) {
				cout<<tmpkNN[i*k+j].second
					<<"-"<<tmpkNN[i*k+j].first
					<<" ";
			}
			cout<<endl;
		}
		cout<<endl;
	}

	knn_merge(all2all_kNN, tmpkNN, numof_ref_points, k, sckNN);
}

void bintree::find_knn_single_query(double *ref, double *query, long *ref_ids, int numof_ref_points, int dim, int k,
						   // output
						   pair<double, long> *result,
						   // auxliary 
						   double *diff, pair<double, long> *dist)
{

	bool dealloc_dist = false;
	bool dealloc_diff = false;
	if(!dist) {
		dist = new pair<double, long> [numof_ref_points];
		dealloc_dist = true;
	}
	if(!diff) {
		diff = new double [dim];
		dealloc_diff = true;
	}

	double alpha = -1.0;
	int ONE = 1;

	for(int i = 0; i < numof_ref_points; i++) {
		memcpy(diff, query, sizeof(double)*dim);
		daxpy(&dim, &alpha, &(ref[i*dim]), &ONE, &(diff[0]), &ONE);
		dist[i].first = ddot(&dim, &(diff[0]), &ONE, &(diff[0]), &ONE);
		dist[i].second = ref_ids[i];
	}

	sort(dist, dist+numof_ref_points);

	result[0].first = dist[0].first;
	result[0].second = dist[0].second;
	for(int i = 1; i < k; i++) {
		result[i].first = DBL_MAX;
		result[i].second = -1L;
	}
	int nk = 1, curr = 1;
	while(nk < k && curr < numof_ref_points) {
		if(dist[curr].second != result[nk-1].second) {
			result[nk].first = dist[curr].first;
			result[nk].second = dist[curr].second;
			nk++;
		}
		curr++;
	}

	if(dealloc_dist) 
		delete [] dist;
	if(dealloc_diff)
		delete [] diff;

}











/*
void bintree::queryR( pbinData inData, long rootNpoints, double dupFactor,
	     pbinNode searchNode, 
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
  MPI_CALL(MPI_Allreduce( &numof_query_points, &total_query_points, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD ));
  }


  binData* leafData;
  pbinNode leaf;
  bintree::distributeToLeaves(inData, rootNpoints, dupFactor, searchNode, range,  &leafData, &leaf);
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

  MPI_CALL(MPI_Allreduce( &numof_query_points, &global_numof_query_points, 
  1, MPI_INT, MPI_SUM, comm ));
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



void bintree::queryR(	pbinData inData, long rootNpoints, 
				double dupFactor,
				pbinNode searchNode, 
				vector< pair<double, long> > *&neighbors, 
				int *nvectors)
{
  int size, rank;
	
  binData* leafData;
  pbinNode leaf;
 
  int total_query_points; 
  int dim = inData->dim;
  int numof_query_points = inData->X.size() / dim;
  assert( inData->radii.size() == numof_query_points );

  if(searchNode->options.pruning_verbose) {
  MPI_CALL(MPI_Allreduce( &numof_query_points, &total_query_points, 1, MPI_INT, MPI_SUM, searchNode->comm ));
   }


  bintree::distributeToLeaves(inData, rootNpoints, dupFactor, searchNode, &leafData, &leaf);
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
  MPI_CALL(MPI_Allreduce( &numof_query_points, &global_numof_query_points, 1, MPI_INT, MPI_SUM, comm ));
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
*/



// get neighbors' coordinates
// note: to use this function, ref are assumed to be sorted according to its global id.
void bintree::getNeighborCoords(int numof_ref_points, double *ref, int dim,
								int numof_query_points, int k,
								vector< pair<double, long> > &all2all_kNN,
								// output
								double *neighborCoords,	// numof_ref_points * k * dim
								MPI_Comm comm)
{
	int size, rank;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	int max_numof_ref_points, global_numof_ref_points;
	MPI_CALL(MPI_Allreduce(&numof_ref_points, &max_numof_ref_points, 1, MPI_INT, MPI_MAX, comm));
	MPI_CALL(MPI_Allreduce(&numof_ref_points, &global_numof_ref_points, 1, MPI_INT, MPI_SUM, comm));
	int ppn = global_numof_ref_points / size;	// average points per process

	vector<int> send_count(size);
	vector<int> send_disp(size);
	vector<int> recv_count(size);
	vector<int> recv_disp(size);

	vector<long> send_knn_ids(numof_query_points*k);
	vector<double> send_knn_coords(numof_query_points*k);
	vector<long> recv_knn_ids;
	vector<double> recv_knn_coords;

	// 1. copy knn ids of all query points to send_knn_ids
	#pragma omp parallel if (numof_query_points > 1000)
    {
		#pragma omp for
		for(int i = 0; i < all2all_kNN.size(); i++)
			send_knn_ids[i] = all2all_kNN[i].second;
	}

    if( _COLLECT_COORD_DEBUG_ & SC_DEBUG_VERBOSE ) {
	    for(int r = 0; r < size; r++) {
		    if(rank == r) {
			    cout<<"rank: "<<rank<<" I need these ids (send_knn_ids): ";
			    for(int i = 0; i < send_knn_ids.size(); i++)
				    cout<<send_knn_ids[i]<<" ";
			    cout<<endl;
		    }
		    MPI_Barrier(comm);
	    }
	}

    if(_COLLECT_COORD_DEBUG_) {
        MPI_Barrier(comm);
        if(rank == 0) {
            cout<<"rank "<<rank<<": 1. copy knn ids to send_knn_ids done, all2all_kNN.size = "<<all2all_kNN.size()<<", nquery*k = "<<send_knn_ids.size()<<endl;
        }
    }

	// 2. remove duplicate ids to reduce alltoallv communication cost
	if(send_knn_ids.size() > 0)
		omp_par::merge_sort( send_knn_ids.begin(), send_knn_ids.end() );
	vector<long>::iterator it = unique(send_knn_ids.begin(), send_knn_ids.end());
	send_knn_ids.resize(it-send_knn_ids.begin());

    if(_COLLECT_COORD_DEBUG_ | SC_DEBUG_VERBOSE) {
        for(int r = 0; r < size; r++) {
            if(rank == r) {
			    cout<<"rank: "<<rank<<" I need these ids after rm dup (send_knn_ids): ";
			    for(int i = 0; i < send_knn_ids.size(); i++)
				    cout<<send_knn_ids[i]<<" ";
			    cout<<endl;
		    }
		    MPI_Barrier(comm);
	    }
	}

    if(_COLLECT_COORD_DEBUG_) {
        MPI_Barrier(comm);
        if(rank == 0) {
            cout<<"rank "<<rank<<": 2. remove duplicates done. "<<endl;
        }
    }

	// 3. construct a mapping from the point id to its position in ref array
	map<int, int> mapping;	// map<id, pos>
    for(int i = 0; i < send_knn_ids.size(); i++) {
		mapping.insert( make_pair(send_knn_ids[i], i) );
	}

    if(SC_DEBUG_VERBOSE) {
	    for(int r = 0; r < size; r++) {
		    if(rank == r) {
			    cout<<"rank: "<<rank<<" mapping: ";
			    for(map<int,int>::iterator it = mapping.begin(); it != mapping.end(); it++) {
				    cout<<it->first<<" - "<<it->second<<" ";
			    }
			    cout<<endl;
		    }
		    MPI_Barrier(comm);
	    }
	}

    if(_COLLECT_COORD_DEBUG_) {
        MPI_Barrier(comm);
        if(rank == 0) {
            cout<<"rank "<<rank<<": 3. construct a mapping done. "<<endl;
        }
    }

	// 4. send knn ids to home process
	memset(&(send_count[0]), 0, sizeof(int)*size);
    for(int j = 0; j < send_knn_ids.size(); j++) {
		//int target_rank = knn::lsh::idToHomeRank( send_knn_ids[j], ppn, size );
		int target_rank = knn::home_rank(global_numof_ref_points, size, send_knn_ids[j]);
		send_count[ target_rank ]++;
	}

    if(_COLLECT_COORD_DEBUG_) {
        MPI_Barrier(comm);
        if(rank == 0) {
            cout<<"rank "<<rank<<": 4.0 cal home process done. "<<endl;
        }
    }

	MPI_CALL(MPI_Alltoall( &(send_count[0]), 1, MPI_INT, &(recv_count[0]), 1, MPI_INT, comm ));
	int n_total_recv = recv_count[0];
	send_disp[0] = 0;
	recv_disp[0] = 0;
    for(int l = 1; l < size; l++) {
		n_total_recv += recv_count[l];
		send_disp[l] = send_disp[l-1] + send_count[l-1];
		recv_disp[l] = recv_disp[l-1] + recv_count[l-1];
	}
	recv_knn_ids.resize(n_total_recv);		// dynamic allocation, maybe slow

    if(_COLLECT_COORD_DEBUG_ & 0) {
	    for(int r = 0; r < size; r++) {
		    if(rank == r) {
			    cout<<"rank: "<<rank<<": nrecv = "<<n_total_recv<<", recv_count: ";
                for(int i = 0; i < size; i++)
                    cout<<recv_count[i]<<" ";
			    cout<<endl;
		    }
            cout.flush();
		    MPI_Barrier(comm);
	    }
	}

	MPI_CALL(MPI_Alltoallv( &(send_knn_ids[0]), &(send_count[0]), &(send_disp[0]), MPI_LONG,
					&(recv_knn_ids[0]), &(recv_count[0]), &(recv_disp[0]), MPI_LONG, comm));

    if(_COLLECT_COORD_DEBUG_) {
        MPI_Barrier(comm);
        if(rank == 0) {
            cout<<"rank "<<rank<<": 4. send ids to home process (alltoallv) done. "<<endl;
        }
    }

	// 5. copy data points to send buffer.
	send_knn_coords.resize(dim*recv_knn_ids.size());
	recv_knn_coords.resize(dim*send_knn_ids.size());

    if(_COLLECT_COORD_DEBUG_) {
        if(rank == 0) cout<<endl;
        MPI_Barrier(comm);
	    for(int r = 0; r < size; r++) {
		    if(rank == r) {
			    cout<<"rank: "<<rank<<": send_knn_coords.size = "<<send_knn_coords.size()
                    <<", recv_knn_coords.size = "<<recv_knn_coords.size()
                    <<", nref = "<<numof_ref_points<<", ppn = "<<ppn
                    <<", recv_knn_ids: ";
                for(int i = 0; i < recv_knn_ids.size(); i++) {
		            int local_id = recv_knn_ids[i] - rank*ppn;
                    cout<<"["<<recv_knn_ids[i]<<","<<recv_knn_ids[i] - rank*ppn<<"]  ";
                    if(local_id >= numof_ref_points) {
                        cout<<"rank "<<rank<<": !!!!! found error, recv_knn_ids[i] = "<<recv_knn_ids[i]
                            <<", local_id = "<<local_id<<", i = "<<i<<endl;
                    }
                }
                cout<<endl;
		    }
            cout.flush();
		    MPI_Barrier(comm);
	    }
	}

    //#pragma omp parallel for
    for(int i = 0; i < recv_knn_ids.size(); i++) {
		int local_id = recv_knn_ids[i] - rank*ppn;
		memcpy( &(send_knn_coords[i*dim]), &(ref[local_id*dim]), sizeof(double)*dim );
	}


    if(_COLLECT_COORD_DEBUG_) {
        MPI_Barrier(comm);
        if(rank == 0) {
            cout<<"rank "<<rank<<": 5. prepare send data done. "<<endl;
        }
    }


    if(SC_DEBUG_VERBOSE) {
	    for(int r = 0; r < size; r++) {
		    if(rank == r) {
			    cout<<"rank: "<<rank<<" send_knn_coords: ";
			    for(int i = 0; i < send_knn_coords.size()/dim; i++) {
				    for(int j = 0; j < dim; j++)
					    cout<<send_knn_coords[i*dim+j]<<" ";
				    cout<<"   /   ";
			    }
		    }
		    MPI_Barrier(comm);
	    }
	}


    for(int i = 0; i < size; i++)
		send_count[i] = recv_count[i] * dim;
	MPI_CALL(MPI_Alltoall( &(send_count[0]), 1, MPI_INT, &(recv_count[0]), 1, MPI_INT, comm ));
    for(int l = 1; l < size; l++) {
		send_disp[l] = send_disp[l-1] + send_count[l-1];
		recv_disp[l] = recv_disp[l-1] + recv_count[l-1];
	}
	MPI_CALL(MPI_Alltoallv( &(send_knn_coords[0]), &(send_count[0]), &(send_disp[0]), MPI_DOUBLE,
				&(recv_knn_coords[0]), &(recv_count[0]), &(recv_disp[0]), MPI_DOUBLE, comm));

    if(SC_DEBUG_VERBOSE) {
	    for(int r = 0; r < size; r++) {
		    if(rank == r) {
			    cout<<"rank: "<<rank<<" recv_knn_coords: ";
			    for(int i = 0; i < recv_knn_coords.size()/dim; i++) {
				    for(int j = 0; j < dim; j++) 
					    cout<<recv_knn_coords[i*dim+j]<<" ";
				    cout<<"   /   ";
			    }
		    }
		    MPI_Barrier(comm);
	    }
	}

    if(_COLLECT_COORD_DEBUG_) {
        MPI_Barrier(comm);
        if(rank == 0) {
            cout<<"rank "<<rank<<": 5.2 alltoallv send data done. "<<endl;
        }
    }


    // 6. copy data into output array
    for(int i = 0; i < numof_query_points; i++) {
		for(int j = 0; j < k; j++) {
			memcpy( &(neighborCoords[(i*k+j)*dim]), &(recv_knn_coords[mapping[all2all_kNN[i*k+j].second]*dim]), dim*sizeof(double) );
		}
	}

    if(_COLLECT_COORD_DEBUG_) {
        MPI_Barrier(comm);
        if(rank == 0) {
            cout<<"rank "<<rank<<": 6. copy data into output done. "<<endl;
        }
    }

}





