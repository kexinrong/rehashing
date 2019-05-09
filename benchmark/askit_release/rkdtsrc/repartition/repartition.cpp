#include <cstdlib>
#include <cmath>
#include <mpi.h>
#include <omp.h>
#include <vector>
#include <queue>
#include <map>
#include <cstring>
#include <cassert>
#include <iostream>
#include "ompUtils.h"
#include "repartition.h"
#include "direct_knn.h"
#include "verbose.h"
#include <algorithm>
#include <cassert>

using namespace std;


void knn::repartition::Collect_Query_Results(long *query_ids,
					 int *neighbors_per_point, 
					 int *neighbor_ids,
					 int nPts,
					 MPI_Comm comm)
{
	int rank, nproc;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nproc);
	
	// gather the number of query points on each node
	int *nPts_per_proc = new int [nproc];
	int *pnpts = &nPts;
	MPI_Gather(pnpts, 1, MPI_INT, nPts_per_proc, 1, MPI_INT, 0, comm);
	//std::cout<<" - rank "<<rank<<" gather # of query on each proc\n";

	// gather the number of neighbors of each query point
	int * collect_neighbors_per_point;
	int total_nPts = 0;
	
	if(0 == rank) {
		for(int i = 0; i < nproc; i++)
			total_nPts += nPts_per_proc[i];
		//std::cout<<"    + rank "<<rank<<" total # of Pts "<<total_nPts<<std::endl;
		
		collect_neighbors_per_point = new int [total_nPts];
	}
	int *displacement = new int [nproc];
	displacement[0] = 0;
	for(int i = 1; i < nproc; i++)
		displacement[i] = displacement[i-1] + nPts_per_proc[i-1];
	MPI_Gatherv(neighbors_per_point, nPts, MPI_INT, collect_neighbors_per_point, nPts_per_proc, displacement, MPI_INT, 0, comm);
	
	// gather the ids of each query point
	long *collect_query_ids = new long [total_nPts];
	MPI_Gatherv(query_ids, nPts, MPI_LONG, collect_query_ids, nPts_per_proc, displacement, MPI_LONG, 0, comm);
	//std::cout<<" - rank "<<rank<<" gather ids of query on each proc\n";


	// gather the number of neighbors on each node
	int *nNeighs_per_proc = new int [nproc];
	int nNeighs = 0;
	for(int i = 0; i < nPts; i++) 
		nNeighs += neighbors_per_point[i];
	MPI_Gather(&nNeighs, 1, MPI_INT, nNeighs_per_proc, 1, MPI_INT, 0, comm);
	//std::cout<<" - rank "<<rank<<" gather # of neighbors on each proc\n";

	// gather the id of neighbors of each query point
	int * collect_neighbor_ids;
	long total_nNeighbors = 0;
	
	if(0 == rank) {
		for(int i = 0; i < nproc; i++)
			total_nNeighbors += nNeighs_per_proc[i];
		//std::cout<<"    + rank "<<rank<<" total # of neighbors "<<total_nNeighbors<<std::endl;
		
		collect_neighbor_ids = new int [total_nNeighbors];
	}

	for(int i = 1; i < nproc; i++)
		displacement[i] = displacement[i-1] + nNeighs_per_proc[i-1];
	
	MPI_Gatherv(neighbor_ids, nNeighs, MPI_INT, collect_neighbor_ids, 
				nNeighs_per_proc, displacement, MPI_INT, 0, comm);
	//std::cout<<" - rank "<<rank<<" gather id of neighbors on each proc\n";

	//output the result
	if(rank == 0 && 1 == 2) {

/*
		std::cout<<"collect_query_ids ";
		for(int i = 0; i < total_nPts; i++)
			std::cout<<collect_query_ids[i]<<" ";
		std::cout<<std::endl<<std::endl;

		std::cout<<"collect_neighbors_per_point ";
		for(int i = 0; i < total_nPts; i++)
			std::cout<<collect_neighbors_per_point[i]<<" ";
		std::cout<<std::endl<<std::endl;

		std::cout<<"collect_neighbor_ids ";
		for(int i = 0; i < total_nNeighbors; i++)
			std::cout<<collect_neighbor_ids[i]<<" ";
		std::cout<<std::endl<<std::endl;
*/

		pair<map<long, vector<int> >::iterator, bool> ret;
		map<long, vector<int> > NN_OUT;
		int offset = 0;
		for(int i = 0; i < total_nNeighbors; i++) {
			int tmp_n = collect_neighbors_per_point[i];
			vector<int> ids;	// neighbor ids
			ids.reserve(tmp_n);
			for(int j  = 0; j < tmp_n; j++) {
				ids.push_back(collect_neighbor_ids[offset + j]);
				ret = NN_OUT.insert(make_pair<long, vector<int> >(collect_query_ids[i], ids));
				if( false == ret.second ) { // query id already exists
					ids.reserve(tmp_n+ids.size());
					for(int jj = 0; jj < tmp_n; jj++) 
						(ret.first)->second.push_back(collect_neighbor_ids[offset+jj]);
			
				}
			}
			offset += tmp_n;
		}

/*	
		map<long, vector<int> >::iterator iter;
		for(iter = NN_OUT.begin(); iter != NN_OUT.end(); iter++) {
			std::cout<<"query id: "<<iter->first<<" neighbor id: ";
			for(int i = 0; i < iter->second.size(); i++) {
				std::cout<<iter->second.at(i)<<" ";
			}
			std::cout<<std::endl;
		}
*/
	}
	
}



/**
 * repartition query points according the critieria dist(center, query) < clusterRadius + range.
 * this code at first duplicate query points if we need to search for knn on several different processors,
 * and then using 'repartition' function to exchange data among processors.
 */
void knn::repartition::query_repartition(long *queryIDs,		// in: query ids
					   double *queryPts,	// in: query data
					   double *centers,		// in: cluster centers
					   long nPts,		// in: no. of query points
					   long nClusters,		// in: no. of clusters
					   int dim,			// in: dimensionality of data
					   map<int, vector<int> > clusterPos,		// in: define each cluster is on which processor
					   double *Radius, 		// in: radius of each cluster [nClusters]
					   double search_range,	// in: search knn within radius 'range'
					   long **new_queryIDs,	// out: query ids after repartition (local)
					   double **new_queryPts,	// out: query points after repartition (local)
					   long *new_nPts,		// out: no. of query points after repartition (local)
					   MPI_Comm comm)
{
	int nproc, rank;
	MPI_Comm_size(comm, &nproc);
	MPI_Comm_rank(comm, &rank);
	

	for(int i = 0; i < nClusters; i++)
		Radius[i] += search_range;
	double *dist = new double [ nPts * nClusters ];
	knn::compute_distances(centers, queryPts, nClusters, nPts, dim, dist);
	long tmp_n = 0;
	for(int i = 0; i < nPts; i++) {
		for(int j = 0; j < nClusters; j++) {
			if(sqrt(dist[i*nClusters+j])<Radius[j]) {
				tmp_n += clusterPos[j].size();
			}
		}
	}

	int *send_count = new int [nproc];
	memset(send_count, 0, nproc*sizeof(int));
	double *tmp_data = new double [tmp_n*dim];
	long *tmp_ids = new long [tmp_n];
	int *tmp_own = new int [tmp_n];
	long p = 0;
	for(int j = 0; j < nClusters; j++) {
		for(long i = 0; i < nPts; i++) {
			if(sqrt(dist[i*nClusters+j])<Radius[j]) { 
				for(int kk = 0; kk < clusterPos[j].size(); kk++) {
					send_count[ clusterPos[j].at(kk) ]++;
					memcpy(tmp_data+p*dim, queryPts+i*dim, dim*sizeof(double));
					tmp_ids[p] = queryIDs[i];
					tmp_own[p] = clusterPos[j].at(kk);
					p++;
				}
			}
		}
	}
	
	delete [] dist;
	//delete [] queryIDs;
	//delete [] queryPts;

	local_rearrange(&tmp_ids, &tmp_own, &tmp_data, tmp_n, dim);
	repartition(tmp_ids, tmp_data, tmp_n, send_count, dim, new_queryIDs, new_queryPts, new_nPts, comm);
	
	delete [] tmp_own;
	delete [] tmp_data;
	delete [] tmp_ids;
	delete [] send_count;
	
}




void knn::repartition::Calculate_Cluster_Radius(double * centers,		// in: cluster centers
				  double * points,		// in: points
				  int dim,			// in: dimensionality of each point
				  int * localClusterSize,	// in: indicate each cluster has how many points locally !!!
				  int numClusters,		// in: no. of clusters
				  double ** Radius,		// out: radius of each cluster
				  MPI_Comm comm) 		// mpi communicator
{
	int rank, nproc;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nproc);
	
	double * local_radius = new double [ numClusters ];
	int offset = 0, p = 0;
	for(int i = 0; i < numClusters; i++) {
		if(localClusterSize[i] > 0) {
			double * dist = new double [ localClusterSize[i] ];
			knn::compute_distances(points+offset, centers+i*dim, localClusterSize[i], 1, dim, dist);
			//local_radius[p] = cblas_idamax(totalClusterSize[i], dist, 1);
			double tmp_max = dist[0];
			for(int ii = 1; ii < localClusterSize[i]; ii++)
				tmp_max = max(tmp_max, dist[ii]);
			local_radius[i] = sqrt(tmp_max);
			offset += localClusterSize[i]*dim;
			delete [] dist;
		}
		else {	// localClusterSize [i] == 0
			local_radius[i] = 0;
		}
	}
	
	*Radius = new double [numClusters];
	MPI_CALL(MPI_Allreduce(local_radius, *Radius, numClusters, MPI_DOUBLE, MPI_MAX, comm));
	
	delete [] local_radius;
}


int knn::repartition::repartition(long *ids,
		 double *data,
		 long n,
		 int *send_count,
		 int dim,
		 long **new_ids,
		 double **new_data, 
		 long *new_n,
		 MPI_Comm comm, int maxNewPoints)
{
  int nproc, rank;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nproc);
  int *recv_count = new int[nproc];

  //if(!rank) cout << "In repartition" << endl;
  //MPI_Barrier(comm);
  //if(!rank) cout << "sendcount alltoall" << endl;
  MPI_CALL(MPI_Alltoall(send_count, 1, MPI_INT, recv_count, 1, MPI_INT, comm));
  //if(!rank) cout << "sendcount alltoall complete" << endl;
	
  long sum = 0;
  #pragma omp parallel for reduction(+:sum)
  for( int i = 0; i < nproc; i++ ) {
    sum += recv_count[i];
  }
  (*new_n) = sum;

  int *send_displacement = new int[nproc]; 
  int *recv_displacement = new int[nproc]; 
  
  send_displacement[0] = 0;
  recv_displacement[0] = 0;

  for(int i = 1; i < nproc; i++) {
    send_displacement[i] = send_displacement[i-1] + send_count[i-1];
    recv_displacement[i] = recv_displacement[i-1] + recv_count[i-1]; 
  }

/*
  for(int i = 0; i < nproc; i++) {
    if(i == rank) cout << "rank " << rank << " points to receive: " << *new_n << endl;
    MPI_Barrier(comm);
  }
*/

  //MPI_Barrier(comm);
  int totalNewPoints = recv_displacement[nproc-1] + recv_count[nproc-1];
  int maxrecv, minrecv;
  //MPI_Reduce( &sum, &maxrecv, 1, MPI_INT, MPI_MAX, 0, comm);
  //MPI_Reduce( &sum, &minrecv, 1, MPI_INT, MPI_MIN, 0, comm);
  //if(!rank) cout << "Min points to receive: " << minrecv << ", max: " << maxrecv << endl;



/*
  int globalMaxPoints;
  if( maxNewPoints > 0){
    MPI_Barrier(comm);
    MPI_CALL(MPI_Allreduce(&totalNewPoints, &globalMaxPoints, 1, MPI_INT, MPI_MAX, comm));
  }
  if( maxNewPoints > 0 && globalMaxPoints > maxNewPoints ) { //Return error.  Don't transfer anything.
    delete[] send_displacement;
    delete[] recv_displacement;
    delete[] recv_count;
    
    return -1;
  }
*/

  *new_ids = new long[*new_n];
  
  //if(!rank) cout << "id alltoallv barrier" << endl;
  //MPI_Barrier(comm);
  //if(!rank) cout << "id alltoallv" << endl;
  MPI_CALL(MPI_Alltoallv(ids, send_count, send_displacement, MPI_LONG,
			 *new_ids, recv_count, recv_displacement, MPI_LONG, comm));
  //if(!rank) cout << "id alltoallv complete" << endl;
  
  #pragma omp parallel for 
  for(int i = 0; i < nproc; i++) {
    send_count[i] = dim * send_count[i];
    recv_count[i] = dim * recv_count[i];
  }

  #pragma omp parallel for
  for(int i = 1; i < nproc; i++) {
    send_displacement[i] = send_displacement[i]*dim; 
    recv_displacement[i] = recv_displacement[i]*dim;
  }
  
  *new_data = new double[(*new_n)*dim];

  //if(!rank) cout << "point data alltoallv barrier" << endl;
  //MPI_Barrier(comm);
  //if(!rank) cout << "point data alltoallv" << endl;
  MPI_CALL(MPI_Alltoallv(data, send_count, send_displacement, MPI_DOUBLE, 
			 *new_data, recv_count, recv_displacement, MPI_DOUBLE, comm));

  //if(!rank) cout << "point data alltoallv complete" << endl;
  //delete [] send_count;
  delete [] send_displacement;
  delete [] recv_count;
  delete [] recv_displacement;
  
  return 0;
}


void knn::repartition::repartition(long *ids,
		 unsigned int *secondaryIDs,
		 double *data,
		 long n,
		 int *send_count,
		 int dim,
		 long **new_ids,
		 unsigned int **new_secondaryIDs,
		 double **new_data, 
		 long *new_n,
		 MPI_Comm comm)
{
  int nproc, rank;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nproc);
  int *recv_count = new int[nproc];

  MPI_Barrier(comm);
  MPI_CALL(MPI_Alltoall(send_count, 1, MPI_INT, recv_count, 1, MPI_INT, comm));
	
  long sum = 0;
  #pragma omp parallel for reduction(+:sum)
  for( int i = 0; i < nproc; i++ ) {
    sum += recv_count[i];
  }
  (*new_n) = sum;

  int *send_displacement = new int[nproc]; 
  int *recv_displacement = new int[nproc]; 
  
  send_displacement[0] = 0;
  recv_displacement[0] = 0;

  for(int i = 1; i < nproc; i++) {
    send_displacement[i] = send_displacement[i-1] + send_count[i-1];
    recv_displacement[i] = recv_displacement[i-1] + recv_count[i-1]; 
  }

  *new_ids = new long[*new_n];
  *new_secondaryIDs = new unsigned int[*new_n];
  
  MPI_Barrier(comm);
  MPI_CALL(MPI_Alltoallv(ids, send_count, send_displacement, MPI_LONG, 
			 *new_ids, recv_count, recv_displacement, MPI_LONG, comm));
  MPI_CALL(MPI_Alltoallv(secondaryIDs, send_count, send_displacement, MPI_UNSIGNED, 
			 *new_secondaryIDs, recv_count, recv_displacement, MPI_UNSIGNED, comm));
  
  #pragma omp parallel for 
  for(int i = 0; i < nproc; i++) {
    send_count[i] = dim * send_count[i];
    recv_count[i] = dim * recv_count[i];
  }

  #pragma omp parallel for
  for(int i = 1; i < nproc; i++) {
    send_displacement[i] = send_displacement[i]*dim; 
    recv_displacement[i] = recv_displacement[i]*dim;
  }
  
  *new_data = new double[(*new_n)*dim];
 
  MPI_Barrier(comm);
  MPI_CALL(MPI_Alltoallv(data, send_count, send_displacement, MPI_DOUBLE, 
			 *new_data, recv_count, recv_displacement, MPI_DOUBLE, comm));

  //delete [] send_count;
  delete [] send_displacement;
  delete [] recv_count;
  delete [] recv_displacement;
  
}







void knn::repartition::repartition(long *ids,
                                   double *data,
                                   double *radii,
                                   long n,
                                   int *send_count,
                                   int dim,
                                   long **new_ids,
                                   double **new_data, 
                                   double **new_radii,
                                   long *new_n,
                                   MPI_Comm comm)
{
  int nproc, rank;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nproc);
  int *recv_count = new int[nproc];
  
  MPI_Barrier(comm);
  MPI_Alltoall(send_count, 1, MPI_INT, recv_count, 1, MPI_INT, comm);
	
  long sum = 0;
#pragma omp parallel for reduction(+:sum)
  for( int i = 0; i < nproc; i++ ) {
    sum += recv_count[i];
  }
  (*new_n) = sum;
  
  int *send_displacement = new int[nproc]; 
  int *recv_displacement = new int[nproc]; 
  
  send_displacement[0] = 0;
  recv_displacement[0] = 0;
  
  for(int i = 1; i < nproc; i++) {
    send_displacement[i] = send_displacement[i-1] + send_count[i-1];
    recv_displacement[i] = recv_displacement[i-1] + recv_count[i-1]; 
  }
  
  *new_ids = new long[*new_n];
  *new_radii = new double[*new_n];
  
  MPI_Barrier(comm);
  MPI_Alltoallv(ids, send_count, send_displacement, MPI_LONG, 
                *new_ids, recv_count, recv_displacement, MPI_LONG, comm);
  
  MPI_Alltoallv(radii, send_count, send_displacement, MPI_DOUBLE, 
                *new_radii, recv_count, recv_displacement, MPI_DOUBLE, comm);
  
#pragma omp parallel for 
  for(int i = 0; i < nproc; i++) {
    send_count[i] = dim * send_count[i];
    recv_count[i] = dim * recv_count[i];
  }
  
#pragma omp parallel for
  for(int i = 1; i < nproc; i++) {
    send_displacement[i] = send_displacement[i]*dim; 
    recv_displacement[i] = recv_displacement[i]*dim;
  }
  
  *new_data = new double[(*new_n)*dim];
  
  MPI_Barrier(comm);
  MPI_Alltoallv(data, send_count, send_displacement, MPI_DOUBLE, 
                *new_data, recv_count, recv_displacement, MPI_DOUBLE, comm);
  
  //delete [] send_count;
  delete [] send_displacement;
  delete [] recv_count;
  delete [] recv_displacement;
  
}





void knn::repartition::rearrange_data(int *clusterMembership, double *data, long n, int dim, double *re_data){

  std::pair<long, long> *pobj_clusterMembership = new std::pair<long, long>[n];

  #pragma omp parallel for
  for(long int i = 0; i < n; i++) {
    pobj_clusterMembership[i].first = clusterMembership[i];
    pobj_clusterMembership[i].second = i;
  }

  omp_par::merge_sort( &(pobj_clusterMembership[0]), &(pobj_clusterMembership[n]) );

  #pragma omp parallel for
  for( long i = 0; i < n; i ++ ) {
    for( int j = 0; j < dim; j++ )
      re_data[i*dim+j] = data[ pobj_clusterMembership[i].second*dim + j ];
  }

  delete[] pobj_clusterMembership;

}

// use less copy
void knn::repartition::pre_all2all(long *ids, int *membership, double *data, long n, int dim){

  std::pair<long, long> *pobj_clusterMembership = new std::pair<long, long>[n];

  #pragma omp parallel for
  for(long int i = 0; i < n; i++) {
    pobj_clusterMembership[i].first = membership[i];
    pobj_clusterMembership[i].second = i;
  }

  omp_par::merge_sort( &(pobj_clusterMembership[0]), &(pobj_clusterMembership[n]) );


  int xsize = n*dim;

  long *ids_clone = new long[n];
  double *data_clone = new double[n*dim];
  int *membership_clone = new int[n];
  #pragma omp parallel for
  for(int i = 0; i < n; i++) {
	ids_clone[i] = ids[i];
	membership_clone[i] = membership[i];
  }
  #pragma omp parallel for
  for(int i = 0; i < xsize; i++)
	  data_clone[i] = data[i];

  #pragma omp parallel for
  for( long i = 0; i < n; i ++ ) {
    ids[i] = ids_clone[ pobj_clusterMembership[i].second ];
    membership[i] = membership_clone[ pobj_clusterMembership[i].second ];
    //for( int j = 0; j < dim; j++ )
    //  data[i*dim+j] = data_clone[ pobj_clusterMembership[i].second*dim + j ];
	memcpy(data+i*dim, data_clone+pobj_clusterMembership[i].second*dim, sizeof(double)*dim);
  }


  delete[] pobj_clusterMembership;
  delete[] ids_clone;
  delete[] data_clone;
  delete[] membership_clone;
}




void knn::repartition::local_rearrange(long **ids, int **clusterMembership, double **data, long n, int dim){

  std::pair<long, long> *pobj_clusterMembership = new std::pair<long, long>[n];

  #pragma omp parallel for
  for(long int i = 0; i < n; i++) {
    pobj_clusterMembership[i].first = (*clusterMembership)[i];
    pobj_clusterMembership[i].second = i;
  }

  omp_par::merge_sort( &(pobj_clusterMembership[0]), &(pobj_clusterMembership[n]) );

  long *rearranged_ids = new long[n];
  double *rearranged_data = new double[n*dim];
  int *rearranged_clusterMembership = new int[n];

  #pragma omp parallel for
  for( long i = 0; i < n; i ++ ) {
    rearranged_ids[i] = (*ids)[ pobj_clusterMembership[i].second ];
    rearranged_clusterMembership[i] = (*clusterMembership)[ pobj_clusterMembership[i].second ];
    for( int j = 0; j < dim; j++ )
      rearranged_data[i*dim+j] = (*data)[ pobj_clusterMembership[i].second*dim + j ];
  }

  long *temp_ids = (*ids);
  double *temp_data = *data;
  int * temp_clusterMembership = *clusterMembership;

  *ids = rearranged_ids;
  *data = rearranged_data;
  *clusterMembership = rearranged_clusterMembership;

  delete[] pobj_clusterMembership;
  delete[] temp_ids;
  delete[] temp_data;
  delete[] temp_clusterMembership;
}


void knn::repartition::local_rearrange(long **ids, unsigned int **clusterMembership, double **data, long n, int dim){

  std::pair<unsigned int, int> *pobj_clusterMembership = new std::pair<unsigned int, int>[n];

  #pragma omp parallel for
  for(int i = 0; i < n; i++) {
    pobj_clusterMembership[i].first = (*clusterMembership)[i];
    pobj_clusterMembership[i].second = i;
  }

  omp_par::merge_sort( &(pobj_clusterMembership[0]), &(pobj_clusterMembership[n]) );

  long *rearranged_ids = new long[n];
  double *rearranged_data = new double[n*dim];
  unsigned int *rearranged_clusterMembership = new unsigned int[n];

  #pragma omp parallel for
  for( long i = 0; i < n; i ++ ) {
    rearranged_ids[i] = (*ids)[ pobj_clusterMembership[i].second ];
    rearranged_clusterMembership[i] = (*clusterMembership)[ pobj_clusterMembership[i].second ];
    for( int j = 0; j < dim; j++ )
      rearranged_data[i*dim+j] = (*data)[ pobj_clusterMembership[i].second*dim + j ];
  }

  long *temp_ids = (*ids);
  double *temp_data = *data;
  unsigned int * temp_clusterMembership = *clusterMembership;

  *ids = rearranged_ids;
  *data = rearranged_data;
  *clusterMembership = rearranged_clusterMembership;

  delete[] pobj_clusterMembership;
  delete[] temp_ids;
  delete[] temp_data;
  delete[] temp_clusterMembership;
}




void knn::repartition::local_rearrange(long **ids, int **clusterMembership, unsigned int **secondaryIDs,
                                         double **data, long n, int dim){

  std::pair<unsigned int, int> *pobj_clusterMembership = new std::pair<unsigned int, int>[n];

  #pragma omp parallel for
  for(int i = 0; i < n; i++) {
    pobj_clusterMembership[i].first = (*clusterMembership)[i];
    pobj_clusterMembership[i].second = i;
  }

  omp_par::merge_sort( &(pobj_clusterMembership[0]), &(pobj_clusterMembership[n]) );

  long *rearranged_ids = new long[n];
  double *rearranged_data = new double[n*dim];
  int *rearranged_clusterMembership = new int[n];
  unsigned int *rearranged_secondaryIDs = new unsigned int[n];

  #pragma omp parallel for
  for( long i = 0; i < n; i ++ ) {
    rearranged_ids[i] = (*ids)[ pobj_clusterMembership[i].second ];
    rearranged_secondaryIDs[i] = (*secondaryIDs)[ pobj_clusterMembership[i].second ];
    rearranged_clusterMembership[i] = (*clusterMembership)[ pobj_clusterMembership[i].second ];
    for( int j = 0; j < dim; j++ )
      rearranged_data[i*dim+j] = (*data)[ pobj_clusterMembership[i].second*dim + j ];
  }

  long *temp_ids = (*ids);
  double *temp_data = *data;
  int * temp_clusterMembership = *clusterMembership;
  unsigned int * temp_secondaryIDs = *secondaryIDs;

  *ids = rearranged_ids;
  *data = rearranged_data;
  *clusterMembership = rearranged_clusterMembership;
  *secondaryIDs = rearranged_secondaryIDs;

  delete[] pobj_clusterMembership;
  delete[] temp_ids;
  delete[] temp_data;
  delete[] temp_clusterMembership;
  delete[] temp_secondaryIDs;
}





int *knn::repartition::calculate_send_counts(int *totalClusterSize, 
                    int numClusters,
		    int *clusterMembership,
		    int **clusterLocations,
                    long n,
                    MPI_Comm comm)
{
  int rank, nproc;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nproc);

  *clusterLocations = new int[numClusters];
  int *prefix_sum = *clusterLocations;

  omp_par::scan(totalClusterSize, prefix_sum, numClusters);

  int avgObj = (prefix_sum[numClusters-1]+totalClusterSize[numClusters-1])/nproc;
  int *sendCount = new int[nproc];

  //Not quite an optimal partitioning, but close and fast.
  #pragma omp parallel for schedule(static)
  for(int i = 0; i < numClusters; i++)
    prefix_sum[i] = prefix_sum[i] / avgObj; //overwrite with process assignment

  //With integer division, we may end up with invalid owners, so we need to fix them.
  int curr = numClusters - 1;
  while( prefix_sum[curr] > (nproc-1) )
    prefix_sum[curr--] = nproc-1;

  #pragma omp parallel for
  for( int i = 0; i < nproc; i++ )
    sendCount[i] = 0;

//Each thread computes a portion of the sendCount, and then they 
    //are combined sequentially
  int *privateSendCount;
  int threads;
  #pragma omp parallel
  {
    int t = omp_get_thread_num();

    if( t == 0 ) {
      threads = omp_get_num_threads();
      privateSendCount = new int[nproc * threads];
    }

    #pragma omp barrier

    #pragma omp for
    for( int i = 0; i < nproc*threads; i++ )
      privateSendCount[i] = 0;
    #pragma omp for
    for( int i = 0; i < n; i++ )
      privateSendCount[  t*nproc + prefix_sum[clusterMembership[i]] ] ++;
  }

  for( int i = 0; i < threads; i++ )
    for( int j = 0; j < nproc; j++ )
      sendCount[j] += privateSendCount[i*nproc + j];

  delete [] privateSendCount;

  return sendCount;
}


void knn::repartition::local_rearrange(long **ids, double **clusterMembership, double **data, long n, int dim){

  std::pair<double, long> *pobj_clusterMembership = new std::pair<double, long>[n];

  #pragma omp parallel for
  for(long int i = 0; i < n; i++) {
    pobj_clusterMembership[i].first = (*clusterMembership)[i];
    pobj_clusterMembership[i].second = i;
  }

  omp_par::merge_sort( &(pobj_clusterMembership[0]), &(pobj_clusterMembership[n]) );

  long *rearranged_ids = new long[n];
  double *rearranged_data = new double[n*dim];
  double *rearranged_clusterMembership = new double[n];

  #pragma omp parallel for
  for( long i = 0; i < n; i ++ ) {
    rearranged_ids[i] = (*ids)[ pobj_clusterMembership[i].second ];
    rearranged_clusterMembership[i] = (*clusterMembership)[ pobj_clusterMembership[i].second ];
    for( int j = 0; j < dim; j++ )
      rearranged_data[i*dim+j] = (*data)[ pobj_clusterMembership[i].second*dim + j ];
  }

  long *temp_ids = (*ids);
  double *temp_data = *data;
  double * temp_clusterMembership = *clusterMembership;

  *ids = rearranged_ids;
  *data = rearranged_data;
  *clusterMembership = rearranged_clusterMembership;

  delete[] pobj_clusterMembership;
  delete[] temp_ids;
  delete[] temp_data;
  delete[] temp_clusterMembership;
}


void knn::repartition::loadBalance(double *points, long *gids, int numof_points, int dim, int n_over_p,
				 //output
				 int &new_numof_points,
				 MPI_Comm comm)
{

	int rank, size;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	int worldrank;
	MPI_Comm_rank( MPI_COMM_WORLD, &worldrank );
	MPI_Barrier(MPI_COMM_WORLD);
 	if(!worldrank) cout << worldrank << ": Beginning load balance" << endl;	

	
	int * load_per_rank = new int [size];
	MPI_Allgather(&numof_points, 1, MPI_INT, load_per_rank, 1, MPI_INT, comm);
	
	#if DETAIL_VERBOSE
		MPI_Barrier(MPI_COMM_WORLD);
		if(worldrank == 0) cout<<"      - 1st MPI_Allgather() done "<<endl;
	#endif
	
	pair<int, int> *plr = new pair<int, int> [size];
	#pragma omp parallel if(size > 1000)
	{
		#pragma omp for
		for(int i = 0; i < size; i++) {
			plr[i].first = load_per_rank[i];
			plr[i].second = i;
		}
	}
	std::sort(&(plr[0]), &(plr[size]));

	int maxIter = std::ceil(log10((double)size)/log10(2.0)) + 2;
	int niter = 0;
	MPI_Bcast(&maxIter, 1, MPI_INT, 0, comm);

	while(niter++ < maxIter /* && plr[size-1].first > n_over_p */) {
		
		int mypos = -1;
		for(int i = 0; i < size; i++) {
			if(plr[i].second == rank) {
					mypos = i;
				break;
			}
		}
		
		int mypartner_rank = plr[size - 1 - mypos].second;
		
		// check whether find the right partner or not
		int mypartnerpos = -1;
		for(int i = 0; i < size; i++) {
			if(plr[i].second == mypartner_rank) {
					mypartnerpos = i;
				break;
			}
		}
		assert( rank == plr[size-1-mypartnerpos].second );
		

		assert( mypartner_rank >= 0 && mypartner_rank < size );
		int mypartner_nump = load_per_rank[mypartner_rank];
	
		int n_send = 0;
		if(mypos >= size / 2 ) {	// send points to its partner
			n_send = (numof_points - mypartner_nump) / 2;
		}
	
		int n_recv = knn::repartition::pairwise_exchange(mypartner_rank, n_over_p, n_send, dim,
											points+(numof_points-n_send)*dim, gids+(numof_points-n_send), 
											points+numof_points*dim, gids+numof_points, comm);

		if(mypos >= size / 2) {	// send points to its partner
			new_numof_points = numof_points - n_send;
		}
		else {
			new_numof_points = numof_points + n_recv;
		}

		numof_points = new_numof_points;

		MPI_Allgather(&numof_points, 1, MPI_INT, load_per_rank, 1, MPI_INT, comm);
		
		#pragma omp parallel if(size > 1000)
		{
			#pragma omp for
			for(int i = 0; i < size; i++) {
				plr[i].first = load_per_rank[i];
				plr[i].second = i;
			}
		}
		std::sort(&(plr[0]), &(plr[size]));
	
	}

	assert( numof_points == n_over_p );
	


	delete[] plr;
	delete[] load_per_rank;


	MPI_Barrier(MPI_COMM_WORLD);
 	if(!worldrank) cout << worldrank << ": Finished load balance" << endl;	

}





int knn::repartition::pairwise_exchange( int partner_rank, int maxpts, int n_send, int dim, double *point_sendbuf,
                           long *id_sendbuf, double *point_recvbuf, long *id_recvbuf, MPI_Comm comm ){

  MPI_Request req[4]; //0 = idrecv, 1 = pointrecv, 2 = idsend, 3 = pointsend
  MPI_Status stat[4];
  int tag_ids = 11111;
  int tag_points = 22222;

  assert(n_send <= maxpts);

        int worldrank;
        MPI_Comm_rank( MPI_COMM_WORLD, &worldrank );
/*
        cout << worldrank << ": Beginning pairwise_exchange" << endl;
*/


  MPI_Irecv(id_recvbuf, maxpts, MPI_LONG, partner_rank, tag_ids, comm, &(req[0]));
  MPI_Irecv(point_recvbuf, maxpts*dim, MPI_DOUBLE, partner_rank, tag_points, comm, &(req[1]));

  MPI_Isend(id_sendbuf, n_send, MPI_LONG, partner_rank, tag_ids, comm, &(req[2]));
  MPI_Isend(point_sendbuf, n_send*dim, MPI_DOUBLE, partner_rank, tag_points, comm, &(req[3]));

  assert( MPI_Waitall(4, req, stat) == MPI_SUCCESS);  

/*
  MPI_Sendrecv( id_sendbuf, n_send, MPI_LONG, partner_rank, tag_ids, 
                id_recvbuf, maxpts,  MPI_LONG, partner_rank, tag_ids, comm, &(stat[0]) );
  MPI_Sendrecv( point_sendbuf, n_send*dim, MPI_DOUBLE, partner_rank, tag_points, 
                point_recvbuf, maxpts*dim,  MPI_DOUBLE, partner_rank, tag_points, comm, &(stat[1]) );
*/

  //Find out how many points we received.a
  int recv_count, recvid_count, nNew;
  MPI_Get_count( &(stat[0]), MPI_LONG, &recvid_count );
  MPI_Get_count( &(stat[1]), MPI_DOUBLE, &recv_count );
  nNew = recv_count / dim;
  assert( recvid_count == nNew );

/*
        cout << worldrank << ": Finished pairwise_exchange" << endl;
*/

  return nNew;
}






int knn::repartition::tree_repartition(long *ids,
		 double *data,
                 int nLocal,
                 int maxpts,
		 int *child_tag,
		 int dim,
		 MPI_Comm comm) {

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

	int worldrank;
	MPI_Comm_rank( MPI_COMM_WORLD, &worldrank );
	MPI_Barrier(MPI_COMM_WORLD);
 	if(!worldrank) cout << worldrank << ": Beginning tree_repartition" << endl;	

  int partner_rank;
 
  if( rank < size/2 ) //Process belongs to left child
    partner_rank = size/2 + rank;
  else
    partner_rank = rank - size/2;


  //Find location of first point to send to other side and number of points to send.
  int first_send; 
  int n_send;
  if(rank >= size/2) {
     first_send = 0;  //Points destined for right child will allways be at beginning.
     int *last = std::lower_bound( child_tag, &(child_tag[nLocal]), 1 );
     n_send = (int) (last - child_tag); //last is 1 past last element to send.
  } else {
     int *first = std::lower_bound( child_tag, &(child_tag[nLocal]), 1 );
     first_send = (int) (first - child_tag);
     n_send = nLocal - first_send;
  }

  int nNew = pairwise_exchange( partner_rank, maxpts, n_send, dim, &(data[first_send*dim]),
                                         &(ids[first_send]), &(data[maxpts*dim]), &(ids[maxpts]), comm );

  int newnLocal = nLocal - n_send + nNew;
  int nOld = nLocal - n_send;

  //Compress and concatenate data within data array.
  int first_new;
  if( rank < size/2 ) {
    first_new = first_send;  
  } else { 
    //Compress remaining original data to beginning of arrays.
      for(int i = 0; i < nOld; i++) ids[i] = ids[n_send+i];
      int nOldDim = nOld*dim;
      int n_senddim = n_send*dim;
      for(int i = 0; i < nOldDim; i++) data[i] = data[n_senddim+i];
  }


    for(int i = 0; i < nNew; i++) ids[nOld+i] = ids[maxpts+i];
    int nNewdim = nNew*dim;
    int nOlddim = nOld*dim;
    int maxptsdim = maxpts * dim;
    for(int i = 0; i < nNewdim; i++) data[nOlddim+i] = data[maxptsdim+i];

  MPI_Barrier(comm);

	MPI_Barrier(MPI_COMM_WORLD);
 	if(!worldrank) cout << worldrank << ": Finished tree_repartition" << endl;	

  return newnLocal;
}






















int knn::repartition::tree_repartition_arbitraryN(vector<long> &ids,
		 vector<double> &data,
                 int nLocal,
		 int *child_tag,
                 int *rank_colors,
		 int dim,
		 MPI_Comm comm) {

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  int worldrank;
  MPI_Comm_rank( MPI_COMM_WORLD, &worldrank );
/*
  MPI_Barrier(MPI_COMM_WORLD);
  if(!worldrank) cout << worldrank << ": Beginning tree_repartition" << endl;	
*/

  int partner_rank;

/*
  if( rank < size/2 ) //Process belongs to left child
    partner_rank = size/2 + rank;
  else
    partner_rank = rank - size/2;
*/


  int first_left, first_right, p_left, p_right;
  first_left = 0; 
  for(int i = 0; i < size; i++) {
    if(rank_colors[i] == 1) {
      first_right = i;
      p_left  = i; 
      p_right = size-i;
      break;
    }
  }

  assert(p_left >= p_right);
  assert(p_left + p_right == size);

  
  if( rank_colors[rank] == 0 ) //Process belongs to left child
    partner_rank = first_right + (rank % p_right);
  else
    partner_rank = rank - p_left; //Parter for first exchange


  //Find location of first point to send to other side and number of points to send.
  int first_send; 
  int n_send;
  if(rank_colors[rank] == 1) {
     first_send = 0;  //Points destined for left child will allways be at beginning.
     int *last = std::lower_bound( child_tag, &(child_tag[nLocal]), 1 );
     n_send = (int) (last - child_tag); //last is 1 past last element to send.
  } else {
     int *first = std::lower_bound( child_tag, &(child_tag[nLocal]), 1 );
     first_send = (int) (first - child_tag);
     n_send = nLocal - first_send;
  }
  

  int newnLocal;
  //First exchange
  if( rank < p_right || rank >= first_right ) {  
    vector<double> pointRecvBuffer;
    vector<long> idRecvBuffer;
    //Find out how many points to receive.
    int n_recv, count;
    MPI_Status s;
    MPI_Sendrecv(&n_send, 1, MPI_INT, partner_rank, 12345, &n_recv, 1, MPI_INT, partner_rank, 12345, comm, &s);
    MPI_Get_count(&s, MPI_INT, &count);
    assert(count == 1);

/*
    pointRecvBuffer.resize(n_recv*dim);
    idRecvBuffer.resize(n_recv);
*/
    int maxpts = std::max(n_send, n_recv);
    pointRecvBuffer.resize(maxpts*dim);
    idRecvBuffer.resize(maxpts);
  
    int nNew = pairwise_exchange( partner_rank, maxpts, n_send, dim, &(data[first_send*dim]),
                                           &(ids[first_send]), &(pointRecvBuffer[0]), &(idRecvBuffer[0]), comm );
  
  
    newnLocal = nLocal - n_send + nNew;
    int nOld = nLocal - n_send;
  
    data.resize(newnLocal*dim);
    ids.resize(newnLocal);
    
  
    //Compress and concatenate data within data array.
    if( rank >= size/2 ) {
      //Compress remaining original data to beginning of arrays.
        for(int i = 0; i < nOld; i++) ids[i] = ids[n_send+i];
        int nOldDim = nOld*dim;
        int n_senddim = n_send*dim;
        for(int i = 0; i < nOldDim; i++) data[i] = data[n_senddim+i];
    }
  
  
    for(int i = 0; i < nNew; i++) ids[nOld+i] = idRecvBuffer[i];
    int nNewdim = nNew*dim;
    int nOlddim = nOld*dim;
    for(int i = 0; i < nNewdim; i++) data[nOlddim+i] = pointRecvBuffer[i];
    nLocal = newnLocal;
  }
  MPI_Barrier(comm);

  //Second exchange (if needed)
  if( p_left != p_right && rank >= p_right && rank < first_right + (p_left-p_right) ) {  
    vector<double> pointRecvBuffer;
    vector<long> idRecvBuffer;
    if( rank_colors[rank] == 0 ) //Process belongs to left child
      partner_rank = first_right + (rank % p_right);
    else {
      partner_rank = rank - p_left + p_right; //Parter for second exchange
      n_send = 0; //Already sent points to other side 
    }

    //Find out how many points to receive.
    int n_recv, count;
    MPI_Status s;
    MPI_Sendrecv(&n_send, 1, MPI_INT, partner_rank, 12345, &n_recv, 1, MPI_INT, partner_rank, 12345, comm, &s);
    MPI_Get_count(&s, MPI_INT, &count);
    assert(count == 1);
    int maxpts = std::max(n_send, n_recv);
    pointRecvBuffer.resize(maxpts*dim);
    idRecvBuffer.resize(maxpts);
 
    int nNew = pairwise_exchange( partner_rank, maxpts, n_send, dim, &(data[first_send*dim]),
                                           &(ids[first_send]), &(pointRecvBuffer[0]), &(idRecvBuffer[0]), comm );
 
    if(rank_colors[rank] == 0)
      assert(nNew == 0); 
  
    newnLocal = nLocal - n_send + nNew;
    int nOld = nLocal - n_send;
 
 
    data.resize(newnLocal*dim);
    ids.resize(newnLocal);
    
  
    //Compress and concatenate data within data array.
    //if( rank >= size/2 ) {
    if( rank_colors[rank] == 1 ) {
      //Compress remaining original data to beginning of arrays.
        for(int i = 0; i < nOld; i++) ids[i] = ids[n_send+i];
        int nOldDim = nOld*dim;
        int n_senddim = n_send*dim;
        for(int i = 0; i < nOldDim; i++) data[i] = data[n_senddim+i];
    }
  
  
    for(int i = 0; i < nNew; i++) ids[nOld+i] = idRecvBuffer[i];
    int nNewdim = nNew*dim;
    int nOlddim = nOld*dim;
    for(int i = 0; i < nNewdim; i++) data[nOlddim+i] = pointRecvBuffer[i];
  }
  MPI_Barrier(comm);




/*
  MPI_Barrier(MPI_COMM_WORLD);
  if(!worldrank) cout << worldrank << ": Finished tree_repartition" << endl;	
*/

  return newnLocal;
}





void knn::repartition::loadBalance_arbitraryN(std::vector<double> &points, std::vector<long> &gids, int numof_points, int dim, 
				 //output
				 int &new_numof_points,
				 MPI_Comm comm)
{

	int rank, size;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	int worldrank;
	MPI_Comm_rank( MPI_COMM_WORLD, &worldrank );
/*
	MPI_Barrier(MPI_COMM_WORLD);
 	if(!worldrank) cout << worldrank << ": Beginning load balance" << endl;	
*/
	
	int * load_per_rank = new int [size];
	MPI_Allgather(&numof_points, 1, MPI_INT, load_per_rank, 1, MPI_INT, comm);
	
/*
	#if DETAIL_VERBOSE
		MPI_Barrier(MPI_COMM_WORLD);
		if(worldrank == 0) cout<<"      - 1st MPI_Allgather() done "<<endl;
	#endif
*/
	
	pair<int, int> *plr = new pair<int, int> [size];
	#pragma omp parallel if(size > 1000)
	{
		#pragma omp for
		for(int i = 0; i < size; i++) {
			plr[i].first = load_per_rank[i];
			plr[i].second = i;
		}
	}
	omp_par::merge_sort(&(plr[0]), &(plr[size]));

	int maxIter = std::ceil(log10((double)size)/log10(2.0)) + 2;
	int niter;
	MPI_Bcast(&maxIter, 1, MPI_INT, 0, comm);

	for(niter = 0; niter < maxIter; niter++) {
		
		int mypos = -1;
		for(int i = 0; i < size; i++) {
			if(plr[i].second == rank) {
					mypos = i;
				break;
			}
		}
		
		int mypartner_rank = plr[size - 1 - mypos].second;
		
		// check whether find the right partner or not
		int mypartnerpos = -1;
		for(int i = 0; i < size; i++) {
			if(plr[i].second == mypartner_rank) {
					mypartnerpos = i;
				break;
			}
		}
		assert( rank == plr[size-1-mypartnerpos].second );
		assert( mypartner_rank >= 0 && mypartner_rank < size );
		int mypartner_nump = load_per_rank[mypartner_rank];
	
		int n_send = 0;
		if(mypos >= size / 2 ) {	// send points to its partner
			n_send = (numof_points - mypartner_nump) / 2;
		}

		//Determine number of points to receive, and adjust vector sizes.
		int n_recv, count;
		MPI_Status s;
		MPI_Sendrecv(&n_send, 1, MPI_INT, mypartner_rank, 12345, &n_recv, 1, MPI_INT, mypartner_rank, 12345, comm, &s);
		MPI_Get_count(&s, MPI_INT, &count);
		assert(count == 1);
		points.resize(points.size()+n_recv*dim);
		gids.resize(gids.size()+n_recv);
		int maxpts = std::max(n_send, n_recv);

		n_recv = knn::repartition::pairwise_exchange(mypartner_rank, maxpts, n_send, dim,
							&(points[(numof_points-n_send)*dim]), &(gids[(numof_points-n_send)]), 
							&(points[numof_points*dim]), &(gids[numof_points]), comm);


		if(mypos >= size / 2) {	// sent points to its partner
			new_numof_points = numof_points - n_send;
		}
		else {
			new_numof_points = numof_points + n_recv;
		}

		numof_points = new_numof_points;

		MPI_Allgather(&numof_points, 1, MPI_INT, load_per_rank, 1, MPI_INT, comm);
		
		#pragma omp parallel if(size > 1000)
		{
			#pragma omp for
			for(int i = 0; i < size; i++) {
				plr[i].first = load_per_rank[i];
				plr[i].second = i;
			}
		}
		std::sort(&(plr[0]), &(plr[size]));
	
	}


	delete[] plr;
	delete[] load_per_rank;

/*
	MPI_Barrier(MPI_COMM_WORLD);
 	if(!worldrank) cout << worldrank << ": Finished load balance" << endl;	
*/

}





