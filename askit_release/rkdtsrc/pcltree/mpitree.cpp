#include "mpitree.h"

#include <mpi.h>
#include <omp.h>
#include <cmath>

#include "verbose.h"
#include "repartition.h"
#include "CmdLine.h"
#include "generator.h"
#include "clustering.h"
#include "direct_knn.h"
#include "mpitree.h"
#include <ompUtils.h>

using namespace std;
using namespace knn;
using namespace knn::repartition;
typedef std::map<int, std::vector<int> > RedistributeReturnType;  // see repartition.h


void MTNode::Insert(  pMTNode in_parent, int maxp, int maxLevel, int minCommSize, MPI_Comm inComm, pMTData inData, int seedType)
{

	double start_t = 0.0;

	int worldsize, worldrank;
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);

	// input checks
	int numof_kids = 2;
	//int cluster_factor = 1; //Do not overcluster
	int cluster_factor = options.cluster_factor; //Do not overcluster
	assert( maxp > 1 );
	assert( maxLevel >= 0 );  //&& maxLevel <= options.max_max_treelevel);

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
		data = new MTData;
		data->Copy(inData);
		// compute centroid and radius. 
		// When a node is a leaf, pruning will be done by its centroid. 
		// Otherwise, use its children clusters

		double *new_C;
		new_C =  centroids(&(X[0]), N, Nglobal, dim, comm); 
		for(int k=0;k<dim;k++) {C.push_back(new_C[k]);} delete [] new_C;

		// compute radius for cluster
		double *R_tmp;
		MPI_Barrier(comm);
		Calculate_Cluster_Radius( &C[0], &X[0], dim, &N, 1, &R_tmp, comm); 
		R.push_back( R_tmp[0]);  delete [] R_tmp;


		MPI_Barrier(MPI_COMM_WORLD);
		int globalsize, globalrank;
		MPI_Comm_size(MPI_COMM_WORLD, &globalsize);
		MPI_Comm_rank(MPI_COMM_WORLD, &globalrank);
		int npoints = data->X.size()/dim;

		return;
	}// end of base case 


	// CLUSTERING
	// 1. calculate cluster centers and point memberships
	int numof_clusters = cluster_factor*numof_kids;
	vector<int> point_to_cluster_membership(N);
	vector<int> local_numof_points_per_cluster(numof_clusters);
	vector<int> global_numof_points_per_cluster(numof_clusters);
	C.resize(numof_clusters*dim);
	int kmeans_max_it=5; //Only two clusters, so don't need a lot of iterations.
				//Ostrovsky seeding will find a "good" paritioning.
	k_clusters( &X[0], dim, N, numof_clusters,  // calculate cluster centers using k-means
				kmeans_max_it, seedType,
				&point_to_cluster_membership[0],
				&C[0],   // this->C, cluster centers
				&global_numof_points_per_cluster[0],
				&local_numof_points_per_cluster[0], 
				comm);
	//k_clusters_balanced( &X[0], dim, N, numof_clusters,  // calculate cluster centers using k-means
	//			kmeans_max_it, seedType, 10,
	//			&point_to_cluster_membership[0],
	//			&C[0],   // this->C, cluster centers
	//			&global_numof_points_per_cluster[0],
	//			&local_numof_points_per_cluster[0], 
	//			comm);


	if(options.debug_verbose) {
		cout<<"rank: "<<worldrank<<" level: "<<level
			<<" centers: ";
		for(int i = 0; i < numof_clusters; i++) {
			for(int j = 0; j < dim; j++)
				cout<<C[i*dim+j]<<" ";
			cout<<"; ";
		}
		cout<<endl;
		cout<<"rank: "<<worldrank
			<<" membership: ";
		for(int i = 0; i < X.size()/dim; i++)
			cout<<point_to_cluster_membership[i]<<" ";
		cout<<endl;
	}

	// 2. calculate radii of clusters
	{
		double *re_points = new double [N * dim];
		rearrange_data(&point_to_cluster_membership[0], &X[0], N, dim, re_points);
		double *R_tmp(NULL);
		Calculate_Cluster_Radius( &C[0], re_points, dim,
			&local_numof_points_per_cluster[0], numof_clusters, &R_tmp, comm);
		R.resize(numof_clusters);
		for(int i=0; i<numof_clusters; i++) { 
			R[i]=R_tmp[i];
		}
		delete [] R_tmp;
		delete [] re_points;
	}

	// 3. assign clusters and points to the kids of the node 
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
	
	start_t = omp_get_wtime();
	knn::repartition::repartition( ra_gids, ra_X, long(N), &send_count[0], 
			dim, &new_gids, &new_X, &new_N, comm);
	Repartition_Tree_Build_T_ += omp_get_wtime() - start_t;

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
	start_t = omp_get_wtime();
	int new_rank  = rank % numof_kids; 
	MPI_Comm new_comm = MPI_COMM_NULL;
	if( MPI_SUCCESS != MPI_Comm_split( comm, my_rank_color, rank, &new_comm) )
		assert(NULL);
	Comm_Split_T_ += omp_get_wtime() - start_t;

	if(options.timing_verbose) {
		int newCommSize;
		MPI_Comm_size(new_comm, &newCommSize);
		char ptrOutputName[256] = {0};
		sprintf(ptrOutputName, "proc%05d_dim%03d_rank%05d_r%02d.info", 
				worldsize, dim, worldrank, options.timing_verbose);
		ofstream fout(ptrOutputName, ios::app|ios::out);
		if(level == 0) fout<<"rank, level, id, #points, comm_size, #points_per_cluster, cluster_2_children_membership, cluster_radius"<<endl;
		fout<<worldrank<<" "<<level<<" "<<chid<<" "<<new_N<<" "<<newCommSize<<" ";
		for(int ii = 0; ii < numof_clusters; ii++)
			fout<<global_numof_points_per_cluster[ii]<<" ";
		for(int ii = 0; ii < numof_clusters; ii++)
			fout<<cluster_to_kid_membership[ii]<<" ";
		for(int ii = 0; ii < numof_clusters; ii++)
			fout<<R[ii]<<" ";
		fout<<endl;
		fout.flush();
		fout.close();
	}

	
	//8. Create new node and insert new data
	kid = new MTNode(its_child_id);
	kid->options.pruning_verbose = options.pruning_verbose;
	kid->options.timing_verbose = options.timing_verbose;
	kid->options.cluster_factor = options.cluster_factor;
	kid->Insert( this, maxp, maxLevel, minCommSize, new_comm, inData, seedType);


};







void distributeToLeaves( pMTData inData, long rootNpoints, double dupFactor,
            pMTNode searchNode, 
            double range, 
            pMTData *outData, pMTNode *leaf)
{
	int size, rank;
	MPI_Comm comm = searchNode->comm;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);
	int numof_kids = 2;
	
	double start_t = 0.0;

	int dim = inData->dim;
	int numof_query_points = inData->X.size() / dim;
	int worldsize, worldrank;
	int globalnumof_query_points;
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
	MPI_Allreduce(&numof_query_points, &globalnumof_query_points, 1, MPI_INT, MPI_SUM, comm);

	//Check for excessive point duplication
	int globalqppproc = rootNpoints / worldsize;
	int myqppproc = globalnumof_query_points / size;
	double currDuplication = (double)myqppproc / (double)globalqppproc; 
	
	if(NULL == searchNode->kid || currDuplication > dupFactor ) {
		*outData = new MTData();
		(*outData)->Copy(inData);
		*leaf = searchNode;
		return;
    
	}
	else {		// not a leaf node
    		int numof_clusters = searchNode->C.size() / dim;
		double * dist = new double [numof_clusters * numof_query_points];
		knn::compute_distances( &(searchNode->C[0]), &(inData->X[0]), 
                           numof_clusters, numof_query_points, dim, 
                           dist );
		
		vector<long> * members_in_kid = new vector<long> [numof_kids];
		for(int i = 0; i < numof_kids; i++)
			members_in_kid[i].reserve(numof_query_points/numof_kids);
    
		// queries that satisfy the condition d(c_i, q_j) < r_i + range
		for(int j = 0; j < numof_clusters; j++) {
			double threshold = searchNode->R[j] + range;
			threshold = threshold * threshold;
			for(int i = 0; i < numof_query_points; i++) {
				if(dist[i*numof_clusters+j] <= threshold ) {
					members_in_kid[searchNode->cluster_to_kid_membership[j]].push_back(i); // store the local id
				}
			}
		}  // for(int j = 0; j < numof_clusters; j++)
		delete [] dist;
    
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
				#pragma omp parallel for
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

		start_t = omp_get_wtime();
		knn::repartition::repartition( new_query_gids, new_queries, 
                                  (long)new_numof_query_points, send_count, dim, 
                                  &re_gids, &re_X, &re_N, comm);
		Repartition_Query_T_ += omp_get_wtime() - start_t;

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
		distributeToLeaves( inData, rootNpoints, dupFactor, 
				searchNode->kid, range, outData, leaf);
    
    
	}	// else
	
	return;
  
}




void distributeToLeaves( pMTData inData, long rootNpoints, double dupFactor,
                        pMTNode searchNode, 
                        pMTData *outData, pMTNode *leaf)
{
	int size, rank;
	MPI_Comm comm = searchNode->comm;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);
	
	double start_t = 0.0;

	int numof_kids = 2;
	int dim = inData->dim;
	int numof_query_points = inData->X.size() / dim;
	assert( inData->radii.size() == numof_query_points );

	int worldsize, worldrank;
	int globalnumof_query_points;
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
	MPI_Allreduce(&numof_query_points, &globalnumof_query_points, 1, MPI_INT, MPI_SUM, comm);

	//Check for excessive point duplication
	int globalqppproc = rootNpoints / worldsize;
	int myqppproc = globalnumof_query_points / size;
	double currDuplication = (double)myqppproc / (double)globalqppproc; 
	
	if(NULL == searchNode->kid || currDuplication > dupFactor) {
		*outData = new MTData();
		(*outData)->Copy(inData);
		*leaf = searchNode;
		return;
    
	}
	else {		// not a leaf node
		int numof_clusters = searchNode->C.size() / dim;
		double * dist = new double [numof_clusters * numof_query_points];
		knn::compute_distances( &(searchNode->C[0]), &(inData->X[0]), 
                           numof_clusters, numof_query_points, dim, 
                           dist );
		
		vector<long> * members_in_kid = new vector<long> [numof_kids];
		for(int i = 0; i < numof_kids; i++)
			members_in_kid[i].reserve(numof_query_points/numof_kids);
    
		// queries that satisfy the condition d(c_i, q_j) < r_i + range
		for(int j = 0; j < numof_clusters; j++) {
			for(int i = 0; i < numof_query_points; i++) {
				double threshold = searchNode->R[j] + sqrt(inData->radii[i]); //radii contains squared distances
				threshold = threshold*threshold;
				threshold += threshold*1.0e-6;
				if(dist[i*numof_clusters+j] <= threshold ) {
					members_in_kid[searchNode->cluster_to_kid_membership[j]].push_back(i); // store the local id
				}
			}
		}  // for(int j = 0; j < numof_clusters; j++)
		delete [] dist;
    
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
				#pragma omp parallel for
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
		group2rankDistribute(new_numof_query_points, &(searchNode->rank_colors[0]), 
                         size, point_to_kid_membership,
                         send_count);
    
    
    
		double *re_X;
		long *re_gids;
		double *re_radii;
		long re_N;
    
		start_t = omp_get_wtime();
		knn::repartition::repartition( new_query_gids, new_queries, new_query_radii,
                                  (long)new_numof_query_points, send_count, dim, 
                                  &re_gids, &re_X, &re_radii, &re_N, comm);
		Repartition_Query_T_ += omp_get_wtime() - start_t;

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
		distributeToLeaves( inData, rootNpoints, dupFactor,searchNode->kid, outData, leaf); 
	}	// else
	
	return;
}




void distributeToNearestLeaf( pMTData inData, 
            pMTNode searchNode, 
            pMTData *outData, pMTNode *leaf)
{
	int size, rank;
	MPI_Comm comm = searchNode->comm;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	int worldsize, worldrank;
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);

	double start_t = 0.0;

	int numof_kids = 2;
	int dim = inData->dim;
	int numof_query_points = inData->X.size() / dim;
	
	if(NULL == searchNode->kid) {	
		*outData = new MTData();
		(*outData)->Copy(inData);
		*leaf = searchNode;
		 return;
	}
	else {		// not a leaf node
    		int numof_clusters = searchNode->C.size() / dim;
		double * dist = new double [numof_clusters * numof_query_points];
		knn::compute_distances( &(searchNode->C[0]), &(inData->X[0]), 
                           numof_clusters, numof_query_points, dim, 
                           dist );
	
		vector<long> * members_in_kid = new vector<long> [numof_kids];
		for(int i = 0; i < numof_kids; i++)
			members_in_kid[i].reserve(numof_query_points/numof_kids);

	   	//Find the cluster center with the minimum distance for each query point
		for(int i = 0; i < numof_query_points; i++) {
			double min_dist = DBL_MAX;
			int min_c = -1;
			for(int j = 0; j < numof_clusters; j++) {
				if( dist[i*numof_clusters+j] < min_dist) {
					min_dist = dist[i*numof_clusters+j];
					min_c = j;
				}
			}
			members_in_kid[searchNode->cluster_to_kid_membership[min_c]].push_back(i);
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
				#pragma omp parallel for
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
		start_t = omp_get_wtime();
		knn::repartition::repartition( new_query_gids, new_queries, 
                                  (long)new_numof_query_points, send_count, dim, 
                                  &re_gids, &re_X, &re_N, comm);
		Repartition_Query_T_ += omp_get_wtime() - start_t;

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
    		distributeToNearestLeaf( inData, searchNode->kid, outData, leaf);
	}	// else
	return;
}
