#include "metrictree.h"

#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <stdio.h>

#include "repartition.h"
#include "CmdLine.h"
#include "generator.h"
#include "clustering.h"
#include "direct_knn.h"
#include "mpitree.h"
#include <ompUtils.h>
//#include "parUtils.h"

using namespace std;
using namespace knn;
using namespace knn::repartition;


void MetricNode::Insert(  pMetricNode in_parent, int maxp, int maxLevel, int minCommSize, MPI_Comm inComm, pMetricData inData, int seedType)
{

	int worldsize, worldrank;
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);

	// input checks
	int numof_kids = 2;
	int cluster_factor = 1; //Do not overcluster
	assert( maxp > 1 );
	assert( maxLevel >= 0 && maxLevel <= options.max_max_treelevel);
		
	// Initializations
	int its_child_id = 0;
	if (in_parent!=NULL)  { level = in_parent->level + 1; parent = in_parent; 
		its_child_id = chid;
	}

	comm = inComm;
	int size; iM( MPI_Comm_size(comm, &size));	  
	int rank; iM( MPI_Comm_rank(comm, &rank)); // needed to print information
	int dim = inData->dim;
	vector<double> &X = inData->X;
	vector<long> &gids= inData->gids;
	int N = X.size()/dim;
				
	MPI_Allreduce(&N, &Nglobal, 1, MPI_INT, MPI_SUM, comm);

	// BASE CASE TO TERMINATE RECURSION
	if (size <= minCommSize ||                // don't want smaller processor partitions
			level == maxLevel ||              // reached max level
			Nglobal <= maxp // not enough points to further partition
			) { 
		data = new MetricData;
		data->Copy(inData);
		// compute centroid and radius. 
		// When a node is a leaf, pruning will be done by its centroid. 
		// Otherwise, use its children clusters


		MPI_Barrier(MPI_COMM_WORLD);
		int globalsize, globalrank;
		MPI_Comm_size(MPI_COMM_WORLD, &globalsize);
		MPI_Comm_rank(MPI_COMM_WORLD, &globalrank);
		int npoints = data->X.size()/dim;

		return;
	}// end of base case 


	// CLUSTERING
	// 1. calculate cluster centers and point memberships
	int numof_clusters = 2;
	vector<int> point_to_cluster_membership(N);
	vector<int> local_numof_points_per_cluster(numof_clusters);
	vector<int> global_numof_points_per_cluster(numof_clusters);

	proj.resize(dim);
	getProjection(&(X[0]), N, dim, &(proj[0]), comm);
	medianPartition( &X[0], N, dim, &(proj[0]),
				&point_to_cluster_membership[0],
				median,
				&global_numof_points_per_cluster[0],
				&local_numof_points_per_cluster[0], 
				comm);

	if(options.debug_verbose) {
		cout<<"worldrank: "<<worldrank
			<<" level: "<<level
			<<" median: "<<median
			<<" proj: ";
		for(int i = 0; i < dim; i++)
			cout<<proj[i]<<" ";
		cout<<endl;
	}
	


	// 2. assign clusters and points to the kids of the node 
	int *dummy_buffer;
	cluster_to_kid_membership.resize(numof_clusters);
	vector<int> point_to_kid_membership(N);
	redistribute( &(global_numof_points_per_cluster[0]), 
					numof_clusters,
					&(point_to_cluster_membership[0]),
					N,	  
					numof_kids,
					// output below
					&(point_to_kid_membership[0]),
					&(cluster_to_kid_membership[0]),
					&dummy_buffer, 
					//input
					comm);  

	delete[] dummy_buffer;
	
	
	
	//4. given cluster and kid membership, fingure out processors to redistribute points
	int *point_to_rank_membership = new int[N];
	int my_rank_color; // used to assign this rank to a processor
	groupDistribute( &point_to_kid_membership[0], 
					 N, numof_kids, 
					 comm, 
					 my_rank_color, 
					 point_to_rank_membership);
		


	rank_colors.resize(size);
	int * all_my_rank_colors = new int [size];
	MPI_Allgather(&my_rank_color, 1, MPI_INT, all_my_rank_colors, 1, MPI_INT, comm);
	for(int i = 0; i < size; i++) rank_colors[i] = all_my_rank_colors[i];
	delete [] all_my_rank_colors;
	its_child_id = numof_kids * its_child_id + my_rank_color;


	//5. repartition points given the new processor ids
	double *ra_X = new double[N*dim];
	long *ra_gids = new long[N]; 
	for (int i=0; i<N*dim; i++)	ra_X[i] = X[i];	
	for (int i=0; i<N; i++)	ra_gids[i] = gids[i];
	local_rearrange(&ra_gids,
			&point_to_rank_membership,
			&ra_X, N, dim);
	double *new_X;
	long *new_gids;
	long new_N;
	vector<int> send_count(size);
	for(int i=0;i<N;i++) send_count[ point_to_rank_membership[i] ] ++;
	knn::repartition::repartition( ra_gids, ra_X, long(N), &send_count[0], 
			dim, &new_gids, &new_X, &new_N, comm);

	delete [] ra_X;
	delete [] ra_gids;	
	delete [] point_to_rank_membership;

	//6. Assign new data to the inData information
	inData->X.resize(new_N*dim);
	inData->gids.resize(new_N);
	#pragma omp parallel for
	for (int i=0; i<new_N*dim; i++) 
		inData->X[i] = new_X[i];
	#pragma omp parallel for
	for (int i=0; i<new_N; i++) 
		inData->gids[i] = new_gids[i]; 
	delete [] new_X;
	delete [] new_gids;
	
	//7. create new communicator
	int new_rank  = rank % numof_kids; 
	MPI_Comm new_comm = MPI_COMM_NULL;
	
	if( MPI_SUCCESS != MPI_Comm_split( comm, my_rank_color, rank, &new_comm) )
		assert(NULL);
	

	if(options.timing_verbose) {
		int newCommSize;
		MPI_Comm_size(new_comm, &newCommSize);
		char ptrOutputName[256] = {0};
		sprintf(ptrOutputName, "proc%05d_dim%03d_rank%05d_r%02d.info", 
				worldsize, dim, worldrank, options.timing_verbose);
		ofstream fout(ptrOutputName, ios::app|ios::out);
		if(level == 0) fout<<"rank, level, id, #points, comm_size, #points_per_cluster, cluster_2_children_membership"<<endl;
		fout<<worldrank<<" "<<level<<" "<<chid<<" "<<new_N<<" "<<newCommSize<<" ";
		for(int ii = 0; ii < numof_clusters; ii++)
			fout<<global_numof_points_per_cluster[ii]<<" ";
		for(int ii = 0; ii < numof_clusters; ii++)
			fout<<cluster_to_kid_membership[ii]<<" ";
		fout<<endl;
		fout.flush();
		fout.close();
	}

	
	//8. Create new node and insert new data
	kid = new MetricNode(its_child_id);
	kid->options.pruning_verbose = options.pruning_verbose;
	kid->options.timing_verbose = options.timing_verbose;
	kid->Insert( this, maxp, maxLevel, minCommSize, new_comm, inData, seedType);


};


void medianPartition(double * points, int numof_points, int dim,
		            double *projDirection,
					// output
					int* point_to_hyperplane_membership,
					double &medianValue,
					int* global_numof_points_per_hyperplane,
					int* local_numof_points_per_hyperplane, 
					MPI_Comm comm)
{
	int rank, nproc;
	MPI_Comm_rank(comm, &rank);

	double *projValue = new double [numof_points];
	//#pragma omp parallel for
	for(int i = 0; i < numof_points; i++) {
		projValue[i] = 0.0;
		for(int j = 0; j < dim; j++) 
			projValue[i] += projDirection[j] * points[i*dim+j];
	}

	int glb_numof_points;
	MPI_Allreduce(&numof_points, &glb_numof_points, 1, MPI_INT, MPI_SUM, comm);
	//double *pmedian = centroids(projValue, numof_points, glb_numof_points, 1, comm);
	//medianValue = pmedian[0];
	//delete [] pmedian;
	//cout<<"median value: "<<medianValue<<endl;
	
	vector<double> tmp(numof_points);
	#pragma omp parallel for
	for(int i = 0; i < numof_points; i++) tmp[i] = projValue[i];
	medianValue = distributeSelect(tmp, glb_numof_points/2, comm);

	local_numof_points_per_hyperplane[0] = 0;
	local_numof_points_per_hyperplane[1] = 0;

	for(int i = 0; i < numof_points; i++) {
		if(projValue[i] < medianValue) {
			point_to_hyperplane_membership[i] = 0;
			local_numof_points_per_hyperplane[0]++;
		}
		else {
			point_to_hyperplane_membership[i] = 1;
			local_numof_points_per_hyperplane[1]++;
		}
	}
	MPI_Allreduce(local_numof_points_per_hyperplane, global_numof_points_per_hyperplane, 2, MPI_INT, MPI_SUM, comm);

//cout<<"rank "<<rank<<" member per pt: ";
//	for(int i = 0;i < numof_points; i++)
//			cout<<point_to_hyperplane_membership[i]<<" ";
//cout<<endl;

	delete [] projValue;
}


void furthestPoint(// input
		   double *points, int numof_points, int dim, double *query,
		   // output
		   double *furP,
		   MPI_Comm comm)
{
	int rank, nproc;
	MPI_Comm_size(comm, &nproc);
	MPI_Comm_rank(comm, &rank);

	double * dist = new double [numof_points];
	knn::compute_distances(points, query, numof_points, 1, dim, dist);
	
	//cout<<"rank "<<rank<<" dist: ";
	//for(int i = 0; i < numof_points; i++)
	//	cout<<sqrt(dist[i])<<" ";
	//cout<<endl;
	
	double * pdmax = max_element(dist, dist+numof_points);
	int idmax = pdmax - dist;
	for(int i = 0; i < dim; i++)
		furP[i] = points[idmax*dim+i];
	
	//cout<<"rank "<<rank<<" max dist: "<<sqrt(*pdmax)<<endl;
	
	double * dmaxg = new double [nproc];
	MPI_Allgather(pdmax, 1, MPI_DOUBLE, dmaxg, 1, MPI_DOUBLE, comm);
	double *pm = max_element(dmaxg, dmaxg+nproc);
	
	//cout<<"rank "<<rank<<" dmaxg: ";
	//for(int i = 0; i < nproc; i++) cout<<dmaxg[i]<<" ";
	//cout<<endl;
	
	int rankmax = pm - dmaxg;
	//cout<<"rank "<<rank<<" rankmax: "<<rankmax<<endl;

	MPI_Bcast(furP, dim, MPI_DOUBLE, rankmax, comm);

	delete [] dist;
	delete [] dmaxg;
}


void getProjection(// input
		   double * points, int numof_points, int dim,
		   // output
		   double * proj,
		   MPI_Comm comm)
{
	int rank;
	MPI_Comm_rank(comm, &rank);

	int global_numof_points;
	double *p1 = new double [dim];
	double *p2 = new double [dim];
	
	MPI_Allreduce(&numof_points, &global_numof_points, 1, MPI_INT, MPI_SUM, comm);
	double *global_mean = centroids(points, numof_points, 
									global_numof_points, dim, comm);
	MPI_Barrier(comm);
	furthestPoint(points, numof_points, dim, global_mean, p1, comm);
	//MPI_Barrier(comm);
	furthestPoint(points, numof_points, dim, p1, p2, comm);

	for(int i = 0; i < dim; i++)
		proj[i] = p1[i] - p2[i];
	double norm = 0.0;
	for(int i = 0; i < dim; i++)
		norm += proj[i] * proj[i];
	norm = sqrt(norm);
	for(int i = 0; i < dim; i++)
		proj[i] /= norm;

	/*
	cout<<"rank: "<<rank<<" mean: ";
	for(int i = 0; i < dim; i++)
		cout<<global_mean[i]<<" ";
	cout<<endl;
	cout<<"rank: "<<rank<<" p1: ";
	for(int i = 0; i < dim; i++)
		cout<<p1[i]<<" ";
	cout<<endl;
	cout<<"rank: "<<rank<<" p2: ";
	for(int i = 0; i < dim; i++)
		cout<<p2[i]<<" ";
	cout<<endl;
*/

	delete [] p1;
	delete [] p2;
	delete [] global_mean;
}



// select the kth smallest element in arr
// for median, ks = glb_N / 2
double distributeSelect(vector<double> &arr, int ks, MPI_Comm comm)
{
	vector<double> S_less;
	//vector<double> S_equal;
	vector<double> S_great;
	S_less.reserve(arr.size());
	S_great.reserve(arr.size());
	
	int N = arr.size();
	int glb_N;
	MPI_Allreduce(&N, &glb_N, 1, MPI_INT, MPI_SUM, comm);
	
	double *pmean = centroids(&(arr[0]), N, glb_N, 1, comm);
	double mean = *pmean;
	delete pmean;
	
	for(int i = 0; i < arr.size(); i++) {
		if(arr[i] > mean) S_great.push_back(arr[i]);
		else S_less.push_back(arr[i]);
	}

	int N_less, N_great, glb_N_less, glb_N_great;
	N_less = S_less.size();
	N_great = S_great.size();
	MPI_Allreduce(&N_less, &glb_N_less, 1, MPI_INT, MPI_SUM, comm);
	MPI_Allreduce(&N_great, &glb_N_great, 1, MPI_INT, MPI_SUM, comm);
	
	if( glb_N_less == ks || glb_N == 1 || glb_N == glb_N_less ) return mean;
	else if(glb_N_less > ks) {
		return distributeSelect(S_less, ks, comm);
	}
	else {
		return distributeSelect(S_great, ks-glb_N_less, comm);
	}


}




void MTdistributeToLeaves( pMetricData inData, long rootNpoints, 
						   double dupFactor, pMetricNode searchNode, 
						   double range, 
						   pMetricData *outData, pMetricNode *leaf)
{
	int size, rank;
	MPI_Comm comm = searchNode->comm;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);
	
	int numof_kids = 2;
	int dim = inData->dim;
	int numof_query_points = inData->X.size() / dim;
	int worldsize, worldrank;
	int globalnumof_query_points;
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
	MPI_Allreduce( &numof_query_points, &globalnumof_query_points, 
				   1, MPI_INT, MPI_SUM, comm);

	//Check for excessive point duplication
	int globalqppproc = rootNpoints / worldsize;
	int myqppproc = globalnumof_query_points / size;
	double currDuplication = (double)myqppproc / (double)globalqppproc; 
	
	if(NULL == searchNode->kid || currDuplication > dupFactor ) {
		*outData = new MetricData();
		(*outData)->Copy(inData);
		*leaf = searchNode;
		return;
    
	}
	else {		// not a leaf node
    	int numof_clusters = 2;
		double * projValue = new double [numof_query_points];
		//#pragma omp parallel for
		for(int i = 0; i < numof_query_points; i++) {
			projValue[i] = 0.0;
			for(int j = 0; j < dim; j++) {
				projValue[i]+= searchNode->proj[j]*inData->X[i*dim+j];
			}
		}
		
		vector<long> * members_in_kid = new vector<long> [numof_kids];
		for(int i = 0; i < numof_kids; i++)
			members_in_kid[i].reserve(numof_query_points/numof_kids);
    
		double medValue = searchNode->median;
		for(int i = 0; i < numof_query_points; i++) {
			if(projValue[i] < medValue) {
				members_in_kid[ searchNode->cluster_to_kid_membership[0] ].push_back(i);
				if( (medValue - projValue[i]) < range ) {
					members_in_kid[ searchNode->cluster_to_kid_membership[1] ].push_back(i);
				}
			}
			else {
				members_in_kid[ searchNode->cluster_to_kid_membership[1] ].push_back(i);
				if( (projValue[i] - medValue) < range ) {
					members_in_kid[ searchNode->cluster_to_kid_membership[0] ].push_back(i);
				}
			}
		}
		
		delete [] projValue;
		
		
		// remove duplicate points
		for(int i = 0; i < numof_kids; i++) {
			sort(members_in_kid[i].begin(), members_in_kid[i].end());
			vector<long>::iterator it;
			it = unique(members_in_kid[i].begin(), members_in_kid[i].end());
			members_in_kid[i].resize(it - members_in_kid[i].begin());
		}
    
		int new_numof_query_points = 0;
		for(int i = 0; i < numof_kids; i++) {
			new_numof_query_points += members_in_kid[i].size();
		}
    
		double *new_queries = new double [new_numof_query_points * dim];
		long *new_query_gids = new long [new_numof_query_points];
		int *point_to_kid_membership = new int [new_numof_query_points];
		int p = 0;
		for(int i = 0; i < numof_kids; i++) {
			for(int j = 0; j < members_in_kid[i].size(); j++) {
				point_to_kid_membership[p] = i;
				int local_ind = members_in_kid[i].at(j);
				new_query_gids[p] = inData->gids[ local_ind ];
				//#pragma omp parallel for
				for(int k = 0; k < dim; k++)
					new_queries [p*dim+k] = inData->X[local_ind*dim+k];
				p++;
			}
		}
		for(int i = 0; i < numof_kids; i++)
			members_in_kid[i].clear();
		delete [] members_in_kid;
		inData->X.clear();
		inData->gids.clear();
		
		int *send_count = new int [size];
		memset(send_count, 0, size*sizeof(int));
		group2rankDistribute(new_numof_query_points, 
						     &(searchNode->rank_colors[0]), 
                             size, point_to_kid_membership, 
                             send_count);
    
		double *re_X;
		long *re_gids;
		long re_N;
		knn::repartition::repartition( new_query_gids, new_queries, 
                                       (long)new_numof_query_points, 
									   send_count, dim, 
                                       &re_gids, &re_X, &re_N, comm);
		delete [] new_queries;
		delete [] new_query_gids;
		delete [] point_to_kid_membership;
		delete [] send_count;
		
		inData->X.resize(dim*re_N);
		inData->gids.resize(re_N);
		#pragma omp parallel for
		for(int i = 0; i < re_N*dim; i++) { inData->X[i] = re_X[i]; }
		#pragma omp parallel for
		for(int i = 0; i < re_N; i++) { inData->gids[i] = re_gids[i]; }
    
		delete [] re_X;
		delete [] re_gids;
    
		MPI_Barrier(comm);
		MTdistributeToLeaves( inData, rootNpoints, dupFactor, 
							searchNode->kid, range, outData, leaf);
    
    
	}	// else
	
	return;
  
}



void MTdistributeToLeaves( pMetricData inData, long rootNpoints, 
						double dupFactor,
                        pMetricNode searchNode, 
                        pMetricData *outData, pMetricNode *leaf)
{
	int size, rank;
	MPI_Comm comm = searchNode->comm;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);
	
	int numof_kids = 2;
	int dim = inData->dim;
	int numof_query_points = inData->X.size() / dim;
	assert( inData->radii.size() == numof_query_points );

	int worldsize, worldrank;
	int globalnumof_query_points;
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
	MPI_Allreduce(&numof_query_points, &globalnumof_query_points, 1, MPI_INT, MPI_SUM, comm);

	if(searchNode->options.debug_verbose) {
		if(worldrank == 0) cout<<"\nlevel "<<searchNode->level<<" distribute to leaf\n";
	}

	//Check for excessive point duplication
	int globalqppproc = rootNpoints / worldsize;
	int myqppproc = globalnumof_query_points / size;
	double currDuplication = (double)myqppproc / (double)globalqppproc; 
	
	if(NULL == searchNode->kid || currDuplication > dupFactor) {
		*outData = new MetricData();
		(*outData)->Copy(inData);
		*leaf = searchNode;
		return;
    
	}
	else {		// not a leaf node
		int numof_clusters = 2;
		double * projValue = new double [numof_query_points];
		//#pragma omp parallel for
		for(int i = 0; i < numof_query_points; i++) {
			projValue[i] = 0.0;
			for(int j = 0; j < dim; j++) {
				projValue[i] += searchNode->proj[j]*inData->X[i*dim+j];
			}
		}
		
		vector<long> * members_in_kid = new vector<long> [numof_kids];
		for(int i = 0; i < numof_kids; i++)
			members_in_kid[i].reserve(numof_query_points/numof_kids);
    
		// queries that satisfy the condition d(c_i, q_j) < r_i + range
		double medValue = searchNode->median;
		for(int i = 0; i < numof_query_points; i++) {
			
			if(searchNode->options.debug_verbose) {
				cout<<"worldrank: "<<worldrank
					<<" - gid: "<<inData->gids[i]
					<<" - pV: "<<projValue[i]
					<<" - med: "<<medValue
					<<" - diff: "<<abs(projValue[i]-medValue)
					<<" - range: "<<sqrt(inData->radii[i])
					<<endl;
			}


			if(projValue[i] < medValue) {
				members_in_kid[ searchNode->cluster_to_kid_membership[0] ].push_back(i);
				if( (medValue - projValue[i]) < sqrt(inData->radii[i]) ) {
					members_in_kid[ searchNode->cluster_to_kid_membership[1] ].push_back(i);
				}
			}
			else {
				members_in_kid[ searchNode->cluster_to_kid_membership[1] ].push_back(i);
				if( (projValue[i] - medValue) < sqrt(inData->radii[i]) ) {
					members_in_kid[ searchNode->cluster_to_kid_membership[0] ].push_back(i);
				}
			}
		}

		delete [] projValue;

		// remove duplicate points
		for(int i = 0; i < numof_kids; i++) {
			sort(members_in_kid[i].begin(), members_in_kid[i].end());
			vector<long>::iterator it;
			it = unique(members_in_kid[i].begin(), members_in_kid[i].end());
			members_in_kid[i].resize(it - members_in_kid[i].begin());
		}
    
		int new_numof_query_points = 0;
		for(int i = 0; i < numof_kids; i++) {
			new_numof_query_points += members_in_kid[i].size();
		}
    
		double *new_queries = new double [new_numof_query_points * dim];
		long *new_query_gids = new long [new_numof_query_points];
		double *new_query_radii = new double [new_numof_query_points];
		int *point_to_kid_membership = new int [new_numof_query_points];
		int p = 0;
		for(int i = 0; i < numof_kids; i++) {
			for(int j = 0; j < members_in_kid[i].size(); j++) {
				point_to_kid_membership[p] = i;
				int local_ind = members_in_kid[i].at(j);
				new_query_gids[p] = inData->gids[ local_ind ];
        			new_query_radii[p] = inData->radii[ local_ind ];
				//#pragma omp parallel for
				for(int k = 0; k < dim; k++)
					new_queries [p*dim+k] = inData->X[local_ind*dim+k];
				p++;
			}
		}
		for(int i = 0; i < numof_kids; i++)
			members_in_kid[i].clear();
		delete [] members_in_kid;
		inData->X.clear();
		inData->gids.clear();
		inData->radii.clear();
		
		int *send_count = new int [size];
		memset(send_count, 0, size*sizeof(int));
		group2rankDistribute( new_numof_query_points, 
							  &(searchNode->rank_colors[0]), 
							  size, point_to_kid_membership,
							  send_count);
    
    
		double *re_X;
		long *re_gids;
		double *re_radii;
		long re_N;
    
		knn::repartition::repartition( new_query_gids, new_queries, 
									   new_query_radii,
									   (long)new_numof_query_points, 
									   send_count, dim, 
									   &re_gids, &re_X, 
									   &re_radii, &re_N, comm);
		delete [] new_queries;
		delete [] new_query_gids;
		delete [] new_query_radii;
		delete [] point_to_kid_membership;
		delete [] send_count;
		
		inData->X.resize(dim*re_N);
		inData->gids.resize(re_N);
		inData->radii.resize(re_N);
		#pragma omp parallel for
		for(int i = 0; i < re_N*dim; i++) { inData->X[i] = re_X[i]; }
		#pragma omp parallel for
		for(int i = 0; i < re_N; i++) { 
			inData->gids[i] = re_gids[i]; 
      		inData->radii[i] = re_radii[i]; 
		}
    
		delete [] re_X;
		delete [] re_gids;
		delete [] re_radii;
    
		MPI_Barrier(comm);
		MTdistributeToLeaves( inData, rootNpoints, dupFactor,
								searchNode->kid, outData, leaf); 
	}	// else
	
	return;
}




void MTdistributeToNearestLeaf( pMetricData inData, 
							  pMetricNode searchNode, 
							  pMetricData *outData, pMetricNode *leaf)
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
	
	if(NULL == searchNode->kid) {	
		*outData = new MetricData();
		(*outData)->Copy(inData);
		*leaf = searchNode;
		 return;
	}
	else {		// not a leaf node
    	int numof_clusters = 2;
		double * projValue = new double [numof_query_points];
		//#pragma omp parallel for
		for(int i = 0; i < numof_query_points; i++) {
			projValue[i] = 0.0;
			for(int j = 0; j < dim; j++) {
				projValue[i]+= searchNode->proj[j]*inData->X[i*dim+j];
			}
		}

		vector<long> * members_in_kid = new vector<long> [numof_kids];
		for(int i = 0; i < numof_kids; i++)
			members_in_kid[i].reserve(numof_query_points/numof_kids);
		
	   	
		//Find the cluster center with the minimum distance for each query point
		double medValue = searchNode->median;
		for(int i = 0; i < numof_query_points; i++) {
			if(projValue[i] < medValue) {
				members_in_kid[ searchNode->cluster_to_kid_membership[0] ].push_back(i);
			}
			else {
				members_in_kid[ searchNode->cluster_to_kid_membership[1] ].push_back(i);
			}
		}

		delete [] projValue;
	
		int new_numof_query_points = 0;
		for(int i = 0; i < numof_kids; i++) {
			new_numof_query_points += members_in_kid[i].size();
		}
    
		double *new_queries = new double [new_numof_query_points * dim];
		long *new_query_gids = new long [new_numof_query_points];
		int *point_to_kid_membership = new int [new_numof_query_points];
		int p = 0;
		for(int i = 0; i < numof_kids; i++) {
			for(int j = 0; j < members_in_kid[i].size(); j++) {
				point_to_kid_membership[p] = i;
				int local_ind = members_in_kid[i].at(j);
				new_query_gids[p] = inData->gids[ local_ind ];
				//#pragma omp parallel for
				for(int k = 0; k < dim; k++)
					new_queries [p*dim+k] = inData->X[local_ind*dim+k];
				p++;
			}
		}
		for(int i = 0; i < numof_kids; i++)
			members_in_kid[i].clear();
		delete [] members_in_kid;
		inData->X.clear();
		inData->gids.clear();
	
		int *send_count = new int [size];
		memset(send_count, 0, size*sizeof(int));
		group2rankDistribute(new_numof_query_points, &(searchNode->rank_colors[0]), 
                         size, point_to_kid_membership, 
                         send_count);

		double *re_X;
		long *re_gids;
		long re_N;
		knn::repartition::repartition( new_query_gids, new_queries, 
                                  (long)new_numof_query_points, send_count, dim, 
                                  &re_gids, &re_X, &re_N, comm);
		delete [] new_queries;
		delete [] new_query_gids;
		delete [] point_to_kid_membership;
		delete [] send_count;
	
		inData->X.resize(dim*re_N);
		inData->gids.resize(re_N);
		#pragma omp parallel for
		for(int i = 0; i < re_N*dim; i++) { inData->X[i] = re_X[i]; }
		//memcpy(&(inData->X[0]), re_X, re_N*dim*sizeof(double));
		#pragma omp parallel for
		for(int i = 0; i < re_N; i++) { inData->gids[i] = re_gids[i]; }
		//memcpy(&(inData->gids[0]), re_gids, re_N*sizeof(long));
    
		delete [] re_X;
		delete [] re_gids;
    
		MPI_Barrier(comm);
   		MTdistributeToNearestLeaf( inData, searchNode->kid, outData, leaf);
	}	// else
	return;
}



