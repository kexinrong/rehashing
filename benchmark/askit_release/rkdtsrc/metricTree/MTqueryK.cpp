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

#include "metrictree.h"

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
void MTqueryKSelectRs( pMetricData redistQuery, pMetricData homeData,
					   pMetricNode searchNode, int global_numof_query_points,
					   int k, double **R) 
{

  double stage_distk_t, stage_sort1_t, stage_sort2_t, stage_collect_t, stage_findmin_t;
  double start_t;

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

  
  if(searchNode->options.timing_verbose) start_t = MPI_Wtime();
  //Find k nearest neighbors within this node.
  pair<double, long> *kneighbors = dist_directKQuery( ref_points, &(redistQuery->X[0]), 
                                              glb_ref_ids,
                                              numof_ref_points, numof_query_points,
                                              k, dim, comm );
  if(searchNode->options.timing_verbose) stage_distk_t = MPI_Wtime() - start_t;
  
  
  if(searchNode->options.timing_verbose) start_t = MPI_Wtime();
  //Sort in order of ascending query point ID.
  pair<long, double> *kthdist = new pair<long, double>[numof_query_points];
  #pragma omp parallel for
  for(int i = 0; i < numof_query_points; i++) {
    kthdist[i].first = redistQuery->gids[i];
    kthdist[i].second = kneighbors[(i+1)*k-1].first;
  }
  delete[] kneighbors;
  free(ref_points);
  free(glb_ref_ids);
  omp_par::merge_sort(&(kthdist[0]), &(kthdist[numof_query_points]));
  if(searchNode->options.timing_verbose) stage_sort1_t = MPI_Wtime() - start_t;
  
  
  
  if(searchNode->options.timing_verbose) start_t = MPI_Wtime();
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
  if(searchNode->options.timing_verbose) stage_collect_t = MPI_Wtime() - start_t;
  
  
  if(searchNode->options.timing_verbose) start_t = MPI_Wtime();
  //Sort k-distances by query ID
  omp_par::merge_sort(homedistances, &(homedistances[rcvdists]));
  if(searchNode->options.timing_verbose) stage_sort2_t = MPI_Wtime() - start_t;
  
  
  if(searchNode->options.timing_verbose) start_t = MPI_Wtime();
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
  
  if(searchNode->options.timing_verbose) stage_findmin_t = MPI_Wtime() - start_t;

 if(searchNode->options.timing_verbose) {
	char ptrOutputName[256] = {0};
	sprintf(ptrOutputName, "proc%05d_dim%03d_rank%05d_r%02d.info", 
			worldsize, dim, worldrank, searchNode->options.timing_verbose);
	ofstream fout(ptrOutputName, ios::app|ios::out);
	fout<<"rselect: worldrank  commsize  distk_t  sort1_t collect_t"
		<<"  sort2_t  findmin_t"
		<<endl;
	fout<<worldrank	<<" "
		<<size<<" " 
		<<stage_distk_t<<" "
		<<stage_sort1_t<<" "
		<<stage_collect_t<<" "
		<<stage_sort2_t<<" "
		<<stage_findmin_t
		<<endl;
	fout<<"r_select_problem_size:(#ref,#query) "<<numof_ref_points<<" "<<numof_query_points<<endl;
	fout.flush();
	fout.close();
	MPI_Barrier(MPI_COMM_WORLD);
  }
 
  delete[] homedistances;
  delete[] kthdist;
  delete[] sendcounts;
  delete[] rcvcounts;
  delete[] senddisp;   
  delete[] rcvdisp;
    
  return;  
}




void MTqueryK( pMetricData inData, long rootNpoints, double dupFactor,
                pMetricNode searchNode, 
                int k, vector<long> *queryIDs,
                vector< pair<double, long> > *kNN){
 
  double start_t;
  double stage_relocation_t, stage_rselect_t, stage_query_ph1_t;
  double stage_pack_t, stage_merge_t, stage_tonearestleaf_t;

  if(searchNode->options.timing_verbose) start_t = MPI_Wtime();
 
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
  knn::repartition::repartition(gid_copy, data_copy, numof_query_points, sendcounts, dim,
              &new_ids, &new_data, &new_n, MPI_COMM_WORLD);

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
  pMetricData querycopy = new MetricData();
  querycopy->Copy(inData);
  
  if(searchNode->options.timing_verbose) stage_relocation_t = MPI_Wtime() - start_t;
  
  //Initial pass to determine search radii.
  if(searchNode->options.timing_verbose) start_t = MPI_Wtime();
  double *R;
  pMetricData redistQuery;
  pMetricNode leaf;
  MTdistributeToNearestLeaf(querycopy, searchNode, &redistQuery, &leaf);
  if(searchNode->options.timing_verbose) stage_tonearestleaf_t = MPI_Wtime() - start_t;
  

  if(searchNode->options.timing_verbose) start_t = MPI_Wtime();
  MTqueryKSelectRs(redistQuery, inData, leaf, global_numof_query_points, k, &R);
  delete redistQuery;
  if(searchNode->options.timing_verbose) stage_rselect_t = MPI_Wtime() - start_t;

  
  
  //Perform r-query with individual radii
  if(searchNode->options.timing_verbose) start_t = MPI_Wtime();
  inData->radii.resize(numof_query_points);
  #pragma omp parallel for
  for(int i = 0; i < numof_query_points; i++)
    inData->radii[i] = R[i];

  delete[] R;
  vector< pair<double, long> > *rneighbors;
  querycopy->Copy(inData);
  int nvectors;
  double rquerytime = MPI_Wtime();

  if(searchNode->options.timing_verbose) stage_query_ph1_t = MPI_Wtime() - start_t;
  
  MTqueryRK(querycopy, rootNpoints, dupFactor, k, searchNode, rneighbors, &nvectors);

  if(searchNode->options.timing_verbose) start_t = MPI_Wtime();
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
  triple<long, double, long> *homeneighbors = new triple<long, double, long>[rcvneighs];
  MPI_Datatype tripledata;
  MPI_Type_contiguous(sizeof(triple<long, double, long>), MPI_BYTE, &tripledata);
  MPI_Type_commit(&tripledata);
  MPI_Alltoallv(tNeighbors, sendcounts, senddisp, tripledata, homeneighbors,
                rcvcounts, rcvdisp, tripledata, MPI_COMM_WORLD);

  delete[] tNeighbors;
  
  if(searchNode->options.timing_verbose) stage_pack_t = MPI_Wtime() - start_t;
  
  
  if(searchNode->options.timing_verbose) start_t = MPI_Wtime();
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
    double last_min = 0.0;
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

  if(searchNode->options.timing_verbose) stage_merge_t = MPI_Wtime() - start_t;

  /*if(searchNode->options.timing_verbose) {
  	MPI_Barrier(MPI_COMM_WORLD);
  	if(worldrank == 0) {
  		cout<<"rank - #query points - merge time - overall time (except nn search)"<<endl;
		fflush(stdout);
		fsync(STDOUT_FILENO);
  	}
  	MPI_Barrier(MPI_COMM_WORLD);
  	for(int t = 0; t < worldsize; t++) {
  		if(worldrank == t) {
		    cout<<worldrank
		    	<<" - "<<nvectors
		    	<<" - "<<stage_merge_t
		    	<<" - "<<stage_relocation_t + stage_rselect_t + stage_query_ph1_t + stage_merge_t + stage_pack_t
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

 if(searchNode->options.timing_verbose) {
	char ptrOutputName[256] = {0};
	sprintf(ptrOutputName, "proc%05d_dim%03d_rank%05d_r%02d.info", 
			worldsize, dim, worldrank, searchNode->options.timing_verbose);
	ofstream fout(ptrOutputName, ios::app|ios::out);
	fout<<"overall: worldrank  #query  merge_t  relocation_t  tonearestleaf_t"
		<<"  rselect_t  query_ph1_t  pack_t"
		<<endl;
	fout<<worldrank	<<" "
		<<nvectors <<" "
		<<stage_merge_t<<" "
		<<stage_relocation_t<<" "
		<<stage_tonearestleaf_t<<" "
		<<stage_rselect_t<<" "
		<<stage_query_ph1_t<<" "
		<<stage_pack_t
		<<endl;
	fout.flush();
	fout.close();
	MPI_Barrier(MPI_COMM_WORLD);
  }


  //Optionally validate results against direct search
  if(searchNode->options.debug_verbose) {
     double *ref = (double*)malloc(numof_query_points*dim*sizeof(double));
     long *glob_ids = (long*)malloc(numof_query_points*sizeof(long));
     #pragma omp parallel for
     for(int i = 0; i < numof_query_points*dim; i++)
       ref[i] = inData->X[i];
     #pragma omp parallel for
     for(int i = 0; i < numof_query_points; i++)
       glob_ids[i] = inData->gids[i];

     std::pair<double, long> *directKNN = dist_directKQuery
                    ( ref, &(inData->X[0]), glob_ids, numof_query_points,
                      numof_query_points, k, dim, MPI_COMM_WORLD);

     pair<double, long>* arr = &((*kNN)[0]);


     //Now, compare output and print results
     int id_mismatches = 0, dist_mismatches = 0; 
     for(int i = 0; i < numof_query_points*k; i++) {
       if( abs(directKNN[i].first - (*kNN)[i].first) > directKNN[i].first*1.0e-5) 
          dist_mismatches++;
       if(directKNN[i].second != (*kNN)[i].second) {
          id_mismatches++;
          cout << inData->gids[i/k] << ": correct= " << directKNN[i].second << "," 
               << directKNN[i].first
               << "  incorrect= " << (*kNN)[i].second << "," << (*kNN)[i].first
               << "  R = " << inData->radii[i/k]  <<endl;
          cout.flush();
         for(int j = 0; j < rcvneighs; j++)  { 
           if( homeneighbors[j].third == directKNN[i].second
               && homeneighbors[j].first == inData->gids[i/k] ) {
             cout << inData->gids[i/k] << ": found neighbor " << homeneighbors[j].third <<endl;
             cout.flush();
           }
         }
        
       }
     }

     cout << worldrank << ": " << "Incorrect IDs: " << id_mismatches
          << ", Incorrect distances: " << dist_mismatches 
          << ", Percent correct: " 
          << (1.0-((double)id_mismatches)/(numof_query_points*k))*100.0
          << endl; cout.flush();
     

  }


  delete[] homeneighbors;  
  delete[] sendcounts;
  delete[] rcvcounts;
  delete[] senddisp;   
  delete[] rcvdisp;
  delete[] offsets;

}


