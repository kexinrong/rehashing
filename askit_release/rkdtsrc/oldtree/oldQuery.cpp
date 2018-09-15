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
#include "binTree.h"
#include "generator.h"
#include "stTree.h"
#include "stTreeSearch.h"
#include "verbose.h"
#include "eval.h"
#include "oldTree.h"
#include "oldQuery.h"
#include "distributeToLeaf_ot.h"
#include "gatherTree_ot.h"

using namespace std;
using namespace knn;
using namespace knn::repartition;

// ============= the following functions are used for exact search =============

/**
 * Performs first pass to select initial guesses for R for k-query.  For internal use only.
 * \param inData Query points.
 * \param searchNode Root of current search.
 * \param k Number of near neighbors to find.
 * \param R [out] Initial guesses for search radii (k nearest neighbors are /at most/ this far away).
 */
void oldtree::queryKSelectRs( pbinData redistQuery, pbinData homeData,
					   poldNode searchNode, int global_numof_query_points,
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

  //Collect search radii at "home" process of each query point (rank query_id/worldsize in MPI_COMM_WORLD).
  int *sendcounts = new int[worldsize];
  int *rcvcounts = new int[worldsize];
  int *senddisp = new int[worldsize];
  int *rcvdisp = new int[worldsize];
  for(int i = 0; i < worldsize; i++) sendcounts[i] = 0;
  for(int i = 0; i < numof_query_points; i++) sendcounts[ knn::lsh::idToHomeRank(kthdist[i].first, ppn, worldsize) ]++;
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


void oldtree::queryK(	pbinData inData, long rootNpoints, double dupFactor,
						poldNode searchNode, 
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
		homeproc[i] = knn::lsh::idToHomeRank(inData->gids[i], ppn, worldsize);
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
	poldNode leaf;
	oldtree::distributeToNearestLeaf(querycopy, searchNode, &redistQuery, &leaf);
	
	#if STAGE_DTREE_OUTPUT_VERBOSE
		MPI_Barrier(MPI_COMM_WORLD);
		if(worldrank == 0) {
			cout<<"   + distribute points to nearest leaf done -> "<<omp_get_wtime() - start_t
				<<" redistQuery->numof_points="<<redistQuery->numof_points<<endl;
		}
		start_t = omp_get_wtime();
	#endif
 

	oldtree::queryKSelectRs(redistQuery, inData, leaf, global_numof_query_points, k, &R);
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
	oldtree::queryRK(querycopy, rootNpoints, dupFactor, k, searchNode, rneighbors, &nvectors);
	
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
					cout<<"id: "<<querycopy->gids[i]<<"  ";
					for(int j = 0; j < neighcount[i]; j++)
						cout<<"("<<rneighbors[i][j].second<<" "<<rneighbors[i][j].first<<")  ";
					cout<<endl;
				}
			}
			cout.flush();
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
	for(int i = 0; i < totalneighbors; i++) sendcounts[ knn::lsh::idToHomeRank(tNeighbors[i].first, ppn, worldsize) ]++;
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_CALL(MPI_Alltoall(sendcounts, 1, MPI_INT, rcvcounts, 1, MPI_INT, MPI_COMM_WORLD));
	omp_par::scan(sendcounts, senddisp, worldsize);
	omp_par::scan(rcvcounts, rcvdisp, worldsize);
	int rcvneighs = rcvdisp[worldsize-1]+rcvcounts[worldsize-1];
	
	#if STAGE_DTREE_OUTPUT_VERBOSE
		MPI_Barrier(MPI_COMM_WORLD);
		//cout<<"   +  worldrank "<<worldrank<<" rcvneighs="<<rcvneighs
		//	<<" numof_query_points="<<numof_query_points
		//	<<" k="<<k
		//	<<endl;
		if(worldrank == 0) {
			cout<<"   + repartition results done -> "<<omp_get_wtime() - start_t
				<<" nvectors="<<nvectors<<endl;
		}
		start_t = omp_get_wtime();
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


	#if PCL_DEBUG_VERBOSE
		if(worldrank == 0) cout<<endl<<"homeneighbors: "<<endl;
		for(int i = 0; i < rcvneighs; i++)
			cout<<"("<<homeneighbors[i].first<<" "<<homeneighbors[i].third<<" "<<homeneighbors[i].second<<")  ";
		cout<<endl;
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

	delete[] homeneighbors;  
	delete[] sendcounts;
	delete[] rcvcounts;
	delete[] senddisp;   
	delete[] rcvdisp;
	delete[] offsets;

}



void oldtree::queryRK( pbinData inData, long rootNpoints, 
				double dupFactor, int k,
				poldNode searchNode, 
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
  poldNode leaf;
  oldtree::distributeToLeaves(inData, rootNpoints, dupFactor, 
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

void oldtree::queryK_Greedy_a2a( long rootNpoints, 
							 poldNode searchNode, int k,
							 int max_points, int max_tree_level,
							 vector<long> *queryIDs,
							 vector< pair<double, long> > *kNN){
	double start_t, end_t;
	double stage_t;

	int worldrank, worldsize;
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);

	#if STAGE_OUTPUT_VERBOSE
		stage_t = omp_get_wtime();
	#endif

	poldNode leaf = searchNode;
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
		if(worldrank == 0) {
			cout<<"  = Query: get leaf points done! -> "<<omp_get_wtime() - stage_t<<endl;
		}
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
	}	// end if (leafCommSize > 1)
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
  for(int i = 0; i < totalneighbors; i++) sendcounts[ knn::lsh::idToHomeRank(tNeighbors[i].first, ppn, worldsize) ]++;
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



void oldtree::queryK_Greedy( pbinData inData, long rootNpoints, 
							 poldNode searchNode, int k,
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

	int dim = inData->dim;
	int numof_query_points = inData->numof_points;
	int global_numof_query_points;
	MPI_CALL(MPI_Allreduce( &numof_query_points, &global_numof_query_points, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD ));
	int ppn = global_numof_query_points/worldsize; //Number of points per node
	int homepoints = (worldrank==worldsize-1) ? ppn+global_numof_query_points%ppn : ppn; //Number of query points "owned" by each process
  

	pbinData redistQuery;
	poldNode leaf;
	if(traverse_type == 0) {
		redistQuery = new binData();
		double *newX;
		long *newgids;
		long newN;
		oldtree::repartitionQueryData(&(inData->X[0]), &(inData->gids[0]), inData->numof_points, inData->dim,
							searchNode, &newX, &newgids, newN);
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
	else {
		oldtree::distributeToNearestLeaf(inData, searchNode, &redistQuery, &leaf);
	}


	#if STAGE_OUTPUT_VERBOSE
		MPI_Barrier(MPI_COMM_WORLD);
		if(worldrank == 0) cout<<"  = Query: distribute to leaf done! -> "<<omp_get_wtime() - stage_t<<endl;
		stage_t = omp_get_wtime();
	#endif


  numof_query_points = redistQuery->numof_points;
  int numof_ref_points = leaf->data->numof_points;

  if(searchNode->options.debug_verbose == 8) {
	int max_numof_query_points, min_numof_query_points, avg_numof_query_points;
	MPI_Reduce(&numof_query_points, &max_numof_query_points, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&numof_query_points, &min_numof_query_points, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&numof_query_points, &avg_numof_query_points, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	if(worldrank == 0) {
		cout<<"    - numof_query_points_per_rank: "
			<<" min: "<<min_numof_query_points
			<<" max: "<<max_numof_query_points
			<<" avg: "<<avg_numof_query_points / worldsize
			<<endl;
	}
  }
  
 
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
			//stTreeSearch_rkdt(&(leaf->data->X[0]), &(redistQuery->X[0]), 
			//				  &(leaf->data->gids[0]), &(redistQuery->gids[0]), 
			//				  numof_ref_points, numof_query_points, dim, max_points, 
			//				  max_tree_level - leaf->level, k, 1, flag_stree_r, kneighbors);
		
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
  for(int i = 0; i < totalneighbors; i++) sendcounts[ knn::lsh::idToHomeRank(tNeighbors[i].first, ppn, worldsize) ]++;
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
  MPI_Type_free(&tripledata);
 
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

  delete redistQuery;
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

  if(searchNode->options.debug_verbose == 8) {
	end_t = omp_get_wtime() - start_t;
	double max_end_t, min_end_t, avg_end_t;
	double max_numof_query_points, min_numof_query_points, avg_numof_query_points;
	MPI_Reduce(&end_t, &max_end_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&end_t, &min_end_t, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&end_t, &avg_end_t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if(worldrank == 0) {
		cout<<"    - queryK_Greedy search time : "
			<<" min: "<<min_end_t
			<<" max: "<<max_end_t
			<<" avg: "<<avg_end_t / worldsize
			<<endl;
	}
  }

}


// rkdt all to all 
void oldtree::knnTreeSearch_RandomRotation_a2a( pbinData refData,
							int k,
							int numof_iterations,
							treeParams params,
							int flag_r, int flag_c,
							//output
							vector<long> & queryIDs,
							vector< pair<double, long> >* & kNN)
{
	double start_t, end_t, dbg_t, max_t, min_t, avg_t;

	int worldsize, worldrank;
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
	
	srand((unsigned)time(NULL)+worldrank);
	
	int dim = refData->dim;
	int numof_ref_points = refData->numof_points;
	int numof_query_points = numof_ref_points;  //queryData->X.size() / dim;
	int global_numof_ref_points;
	MPI_CALL(MPI_Allreduce( &numof_ref_points, &global_numof_ref_points, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD ));
	int global_numof_query_points = global_numof_ref_points;
	int ppn = global_numof_query_points/worldsize; //Number of points per node
	int homepoints = (worldrank==worldsize-1) ? ppn+global_numof_query_points%ppn : ppn; //Number of query points "owned" by each process
	
	// ========== used for correctness check ====== //
	vector<long> sampleIDs;
	vector<double> globalKdist;
	vector<long> globalKid;
	if(params.eval_verbose) {
		get_sample_info(&(refData->X[0]), &(refData->X[0]), &(refData->gids[0]), &(refData->gids[0]), 
				numof_ref_points, numof_query_points, refData->dim, k, 
				sampleIDs, globalKdist, globalKid);
	}
	// =========================================== // 

	for(int iter = 0; iter < numof_iterations; iter++) {
		
		#if OVERALL_TREE_TIMING_VERBOSE
			MPI_Barrier(MPI_COMM_WORLD);
		#endif

		start_t = omp_get_wtime();
		
		// 1. build the tree
		//pbinData ref_iter = new binData();
		//ref_iter->Copy(refData);
		
		MPI_Collective_T_ = 0.0;

		poldNode root = new oldNode();
		root->options.hypertree = params.hypertree;
		root->options.debug_verbose = params.debug_verbose;
		root->options.timing_verbose = params.timing_verbose;
		root->options.pruning_verbose = params.pruning_verbose;
		root->options.splitter = "rkdt";	// use maxVarSplitter
		root->options.flag_r = flag_r;
		root->options.flag_c = flag_c;

		root->Insert(	NULL, params.max_points_per_node, params.max_tree_level, 
						params.min_comm_size_per_node,
						MPI_COMM_WORLD, refData);
		
		#if OVERALL_TREE_TIMING_VERBOSE
			MPI_Collective_Const_T_ += MPI_Collective_T_;
			Tree_Const_T_ += omp_get_wtime() - start_t;
		#endif

		#if RKDT_ITER_VERBOSE
			MPI_Barrier(MPI_COMM_WORLD);
			end_t = omp_get_wtime() - start_t;
			if(worldrank == 0) cout<<"   == tree construction time: "<<end_t<<endl;
		#endif
	
		//release refdata
		//delete ref_iter;
		

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
			oldtree::queryK_Greedy_a2a( global_numof_ref_points, root, k,
									params.max_points_per_node, params.max_tree_level,
									&queryIDs, kNN);
		
		}
		else if(do_query)
		  {
			if(params.debug_verbose == 7 && worldrank == 0) dbg_t = omp_get_wtime();
			vector<long> queryIDs_iter;
			vector< pair<double, long> > kNN_iter;
			oldtree::queryK_Greedy_a2a( global_numof_ref_points, root, k,
									params.max_points_per_node, params.max_tree_level,
									&queryIDs_iter, &kNN_iter);
			
			vector< pair<double, long> > *tmpkNN = new vector< pair<double, long> >;
			oldtree::knn_merge((*kNN), kNN_iter, homepoints, k, *tmpkNN);

			delete kNN;
			kNN = tmpkNN;

		  } else{
		  ;}// end else
	
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
			double hit_rate = 0.0, relative_error = 0.0;
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




void oldtree::knnTreeSearch_RandomRotation( pbinData refData,
							pbinData queryData,
							int k,
							int numof_iterations,
							treeParams params,
							int flag_r, int flag_c,
							//output
							vector<long> & queryIDs,
							vector< pair<double, long> >* & kNN)
{

	double start_t, end_t, dbg_t, max_t, min_t, avg_t;

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
	int homepoints = (worldrank==worldsize-1) ? ppn+global_numof_query_points%ppn : ppn; //Number of query points "owned" by each process
	
	vector<long> sampleIDs;
	vector<double> globalKdist;
	vector<long> globalKid;
	if(params.eval_verbose) {
		get_sample_info(&(refData->X[0]), &(queryData->X[0]),
				&(refData->gids[0]), &(queryData->gids[0]), 
				numof_ref_points, numof_query_points,
				refData->dim, k, 
				sampleIDs, globalKdist, globalKid);
	}


	for(int iter = 0; iter < numof_iterations; iter++) {
		
		if(params.debug_verbose == 8 && worldrank == 0) cout<<"  iter "<<iter<<endl;
		
		#if OVERALL_TREE_TIMING_VERBOSE
			MPI_Barrier(MPI_COMM_WORLD);
		#endif
		start_t = omp_get_wtime();
		
		// 1. build the tree
	
		MPI_Collective_T_ = 0.0;

		poldNode root = new oldNode();
		root->options.hypertree = params.hypertree;
		root->options.debug_verbose = params.debug_verbose;
		root->options.timing_verbose = params.timing_verbose;
		root->options.pruning_verbose = params.pruning_verbose;
		root->options.splitter = "rkdt";	// use maxVarSplitter
		root->options.flag_r = flag_r;
		root->options.flag_c = flag_c;

		root->Insert(	NULL, params.max_points_per_node, params.max_tree_level, 
						params.min_comm_size_per_node,
						MPI_COMM_WORLD, refData);

		#if OVERALL_TREE_TIMING_VERBOSE
			MPI_Collective_Const_T_ += MPI_Collective_T_;
			Tree_Const_T_ += omp_get_wtime() - start_t;
		#endif

		#if RKDT_ITER_VERBOSE
			end_t = omp_get_wtime() - start_t;
			if(worldrank == 0) {
				//cout<<"  "<<iter<<"th iteration: "<<endl
				cout<<"   == tree construction time: "<<end_t<<endl;
			}
		#endif

		//delete ref_iter;


		// 2. search the tree
		#if OVERALL_TREE_TIMING_VERBOSE
			MPI_Barrier(MPI_COMM_WORLD);
		#endif
		start_t = omp_get_wtime();
		
		MPI_Collective_T_ = 0.0;

		//cout<<"RKDT: worldrank: "<<worldrank
		//	<<" queryData->gids.size(): "<<queryData->gids.size()
		//	<<" queryData->X.size(): "<<queryData->X.size()
		//	<<" queryData->numof_points: "<<queryData->numof_points
		//	<<" queryData->dim: "<<queryData->dim
		//	<<endl;

		if(iter == 0) {
			// use nearest traverse strategy
			if(params.debug_verbose == 7 && worldrank == 0) dbg_t = omp_get_wtime();
			
			oldtree::queryK_Greedy( queryData, global_numof_ref_points, root, k, params.traverse_type,
									params.max_points_per_node, params.max_tree_level,
									&queryIDs, kNN);
			
			
			#if RKDT_MERGE_VERBOSE
			if(worldrank == 0) {
				cout<<"iter: "<<iter<<" before merge: "<<endl;
				for(int i = 0; i < queryIDs.size(); i++) {
					cout<<queryIDs[i]<<" ";
					for(int j = 0; j < k; j++) {
						cout<<(*kNN)[i*k+j].second<<" "<<(*kNN)[i*k+j].first<<" ";
					}
					cout<<endl;
				}
				cout<<endl;
			}
			#endif
		
		}
		else {
			if(params.debug_verbose == 7 && worldrank == 0) dbg_t = omp_get_wtime();
			vector<long> queryIDs_iter;
			vector< pair<double, long> > kNN_iter;
			oldtree::queryK_Greedy( queryData, global_numof_ref_points, root, k, params.traverse_type,
									params.max_points_per_node, params.max_tree_level,
									&queryIDs_iter, &kNN_iter);
		
			#if RKDT_MERGE_VERBOSE
			if(worldrank == 0) {
				cout<<endl<<"iter: "<<iter<<" before merge: "<<endl;
				for(int i = 0; i < queryIDs_iter.size(); i++) {
					cout<<queryIDs_iter[i]<<" ";
					for(int j = 0; j < k; j++) {
						cout<<kNN_iter[i*k+j].second<<" "<<kNN_iter[i*k+j].first<<" ";
					}
					cout<<endl;
				}
			}
			#endif
			
			vector< pair<double, long> > *tmpkNN = new vector< pair<double, long> >;
			oldtree::knn_merge((*kNN), kNN_iter, homepoints, k, *tmpkNN);

			delete kNN;
			kNN = tmpkNN;
			
			#if RKDT_MERGE_VERBOSE
			if(worldrank == 0) {
				cout<<"iter: "<<iter<<" after merge: "<<endl;
				for(int i = 0; i < queryIDs.size(); i++) {
					cout<<queryIDs[i]<<" ";
					for(int j = 0; j < k; j++) {
						cout<<(*kNN)[i*k+j].second<<" "<<(*kNN)[i*k+j].first<<" ";
					}
					cout<<endl;
				}
			} // end if
			#endif

		} // end else
			
		//delete ref_iter;
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
			double hit_rate = 0.0, relative_error = 0.0;
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



// if some A[i].first == B[j].first, always choose A[i], and remove B[j], the value of B might changed
void oldtree::knn_merge( vector< pair<double, long> > &A, vector< pair<double, long> > &B,
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
			if( A[aloc] <= B[bloc] ) {
				result[resultloc++] = A[aloc++];
			}
			else {
				result[resultloc++] = B[bloc++];
			}
		}
	} // end for (i < n)
}



