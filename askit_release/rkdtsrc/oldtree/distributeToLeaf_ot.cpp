#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <stdio.h>
#include <ompUtils.h>
#include <blas.h>
#include "repartition.h"
#include "CmdLine.h"
#include "generator.h"
#include "clustering.h"
#include "direct_knn.h"
#include "mpitree.h"
#include "binTree.h"
#include "oldTree.h"
#include "distributeToLeaf_ot.h"
#include "rotation.h"
#include "verbose.h"

using namespace std;
using namespace knn;
using namespace knn::repartition;

// this function is used for "mtree" and exact search. It need a search range "range"
// points could go to both kids
void oldtree::distributeToLeaves(	pbinData inData, long rootNpoints, 
									double dupFactor, poldNode searchNode, 
									double range, 
									pbinData *outData, poldNode *leaf)
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
	MPI_CALL(MPI_Allreduce( &numof_query_points, &globalnumof_query_points, 1, MPI_INT, MPI_SUM, comm));

	//Check for excessive point duplication
	int globalqppproc = rootNpoints / worldsize;
	int myqppproc = globalnumof_query_points / size;
	double currDuplication = (double)myqppproc / (double)globalqppproc; 
	
	if(NULL == searchNode->kid || currDuplication > dupFactor ) {
		*outData = new binData();
		(*outData)->Copy(inData);
		*leaf = searchNode;
		return;
    
	}
	else {		// not a leaf node
		vector<long> * members_in_kid = new vector<long> [numof_kids];
		for(int i = 0; i < numof_kids; i++)
			members_in_kid[i].reserve(numof_query_points/numof_kids);
		double medValue = searchNode->median;
		
		vector<double> projValue;
		projValue.resize(numof_query_points);
		if( 0 == strcmp(searchNode->options.splitter.c_str(), "rsmt") 
				&& -1 == searchNode->coord_mv ) {
			int ONE = 1;
			#pragma omp parallel for
			for(int i = 0; i < numof_query_points; i++) {
				//for(int j = 0; j < dim; j++)
				//	projValue[i]+= searchNode->proj[j]*inData->X[i*dim+j];
				projValue[i] = ddot(&dim, &(searchNode->proj[0]), &ONE, &(inData->X[i*dim]), &ONE);
			}
		}
		else {				// "rkdt"
			int mvid = searchNode->coord_mv;
			#pragma omp parallel for
			for(int i = 0; i < numof_query_points; i++) {
				projValue[i] = inData->X[i*dim+mvid];
			}
		}

		//#pragma omp parallel for
		//for(int i = 0; i < numof_query_points; i++) {
		//	for(int j = 0; j < dim; j++) 
		//		projValue[i]+= searchNode->proj[j]*inData->X[i*dim+j];
		//}

		for(int i = 0; i < numof_query_points; i++) {
			if(projValue[i] < medValue) {
				//members_in_kid[ searchNode->cluster_to_kid_membership[0] ].push_back(i);
				members_in_kid[ 0 ].push_back(i);
				if( (medValue - projValue[i]) < range ) {
					//members_in_kid[ searchNode->cluster_to_kid_membership[1] ].push_back(i);
					members_in_kid[ 1 ].push_back(i);
				}
			}
			else {
				//members_in_kid[ searchNode->cluster_to_kid_membership[1] ].push_back(i);
				members_in_kid[ 1 ].push_back(i);
				if( (projValue[i] - medValue) < range ) {
					//members_in_kid[ searchNode->cluster_to_kid_membership[0] ].push_back(i);
					members_in_kid[ 0 ].push_back(i);
				}
			}
		} // end for
		
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
		oldtree::distributeToLeaves( inData, rootNpoints, dupFactor, 
							searchNode->kid, range, outData, leaf);
    
    
	}	// else
	
	return;
  
}


// this function is used for "mtree" and exact search, each point has its own search range
// points can go to both kids
void oldtree::distributeToLeaves( pbinData inData, long rootNpoints, 
						double dupFactor,
                        poldNode searchNode, 
                        pbinData *outData, poldNode *leaf)
{
	int size, rank;
	MPI_Comm comm = searchNode->comm;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);
	
	int numof_kids = 2;
	int dim = inData->dim;
	int numof_query_points = inData->numof_points;
	assert( inData->radii.size() == numof_query_points );

	int worldsize, worldrank;
	int globalnumof_query_points;
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
	MPI_CALL(MPI_Allreduce(&numof_query_points, &globalnumof_query_points, 1, MPI_INT, MPI_SUM, comm));

	double stage_t = omp_get_wtime();

	//Check for excessive point duplication
	int globalqppproc = rootNpoints / worldsize;
	int myqppproc = globalnumof_query_points / size;
	double currDuplication = (double)myqppproc / (double)globalqppproc; 
	
	if(NULL == searchNode->kid || currDuplication > dupFactor) {
		*outData = new binData();
		(*outData)->Copy(inData);
		*leaf = searchNode;
		return;
	}
	else {		// not a leaf node
		vector<long> * members_in_kid = new vector<long> [numof_kids];
		for(int i = 0; i < numof_kids; i++)
			members_in_kid[i].reserve(numof_query_points/numof_kids);
		double medValue = searchNode->median;
		
		vector<double> projValue;
		projValue.resize(numof_query_points);
		if( 0 == strcmp(searchNode->options.splitter.c_str(), "rsmt") 
				&& -1 == searchNode->coord_mv ) {
			int ONE = 1;
			#pragma omp parallel for
			for(int i = 0; i < numof_query_points; i++) {
				projValue[i] = ddot(&dim, &(searchNode->proj[0]), &ONE, &(inData->X[i*dim]), &ONE);
			}
		}
		else {				// "rkdt"
			int mvid = searchNode->coord_mv;
			#pragma omp parallel for
			for(int i = 0; i < numof_query_points; i++) {
				projValue[i] = inData->X[i*dim+mvid];
			}
		}
	
		#if STAGE_OUTPUT_VERBOSE
			MPI_Barrier(comm);
			if(worldrank == 0 && searchNode->level < 3)
				cout<<"    >> Dist2Leaf: level "<<searchNode->level<<": project points done! -> "<<omp_get_wtime()-stage_t<<endl;
			stage_t = omp_get_wtime();
		#endif

		for(int i = 0; i < numof_query_points; i++) {
			if(projValue[i] < medValue) {
				//members_in_kid[ searchNode->cluster_to_kid_membership[0] ].push_back(i);
				members_in_kid[ 0 ].push_back(i);
				if( (medValue - projValue[i]) < sqrt(inData->radii[i]) ) {
					//members_in_kid[ searchNode->cluster_to_kid_membership[1] ].push_back(i);
					members_in_kid[ 1 ].push_back(i);
				}
			}
			else {
				//members_in_kid[ searchNode->cluster_to_kid_membership[1] ].push_back(i);
				members_in_kid[ 1 ].push_back(i);
				if( (projValue[i] - medValue) < sqrt(inData->radii[i]) ) {
					//members_in_kid[ searchNode->cluster_to_kid_membership[0] ].push_back(i);
					members_in_kid[ 0 ].push_back(i);
				}
			}	
		} // end for
	
		#if PCL_DEBUG_VERBOSE
			for(int i = 0; i < numof_query_points; i++) {
				cout<<"("<<inData->gids[i]<<" "
					<<projValue[i] - medValue<<" "
					<<sqrt(inData->radii[i])<<")  ";
			}
			cout<<endl;
		#endif


		#if STAGE_OUTPUT_VERBOSE
			MPI_Barrier(comm);
			if(worldrank == 0 && searchNode->level < 3)
				cout<<"    >> Dist2Leaf: level "<<searchNode->level<<": assign membership done! -> "<<omp_get_wtime()-stage_t<<endl;
			stage_t = omp_get_wtime();
		#endif

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
		
		#if STAGE_OUTPUT_VERBOSE
			MPI_Barrier(comm);
			if(worldrank == 0 && searchNode->level < 3)
				cout<<"    >> Dist2Leaf: level "<<searchNode->level<<": remove duplicate done! -> "<<omp_get_wtime()-stage_t<<endl;
			stage_t = omp_get_wtime();
		#endif

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
		inData->numof_points = re_N;
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
   	
		#if STAGE_OUTPUT_VERBOSE
			MPI_Barrier(comm);
			if(worldrank == 0 && searchNode->level < 3)
				cout<<"    >> Dist2Leaf: level "<<searchNode->level<<": repartition done! -> "<<omp_get_wtime()-stage_t<<endl;
			stage_t = omp_get_wtime();
		#endif

		MPI_Barrier(comm);
		oldtree::distributeToLeaves( inData, rootNpoints, dupFactor,
								searchNode->kid, outData, leaf); 
	}	// else
	
	return;
}



// this one is used for both "mtree" or "rkdt"
// for "rsmt", we project points onto direction "searchNode->proj"
// for "rkdt, flag_r == 1", we also should project points on direction "searchNode->proj"
// for "rkdt, flag_r == 2", we only rotate points on the root, other level direct use coord_mv
// for "rkdt, flag_r == 0", we use direct coord_mv on all level
void oldtree::distributeToNearestLeaf( pbinData inData, poldNode searchNode, 
									   pbinData *outData, poldNode *leaf)
{
	double start_t, end_t;
	double stage_t = omp_get_wtime();
		
	int size, rank;
	MPI_Comm comm = searchNode->comm;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);
	int worldsize, worldrank;
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);

	int numof_kids = 2;
	int dim = inData->dim;
	int numof_query_points = inData->numof_points;
	
	if(searchNode->level == 0 && searchNode->options.flag_r == 1
			&& 0 == strcmp(searchNode->options.splitter.c_str(), "rkdt") )	{ // root
		double * tmpX = new double [numof_query_points*dim];
		memcpy( tmpX, &(inData->X[0]), sizeof(double)*numof_query_points*dim );
		rotatePoints( tmpX, numof_query_points, dim, searchNode->rw, &(inData->X[0]) );
		delete [] tmpX;
	}

	#if STAGE_OUTPUT_VERBOSE
		MPI_Barrier(comm);
		if(worldrank == 0 && searchNode->level < 3)
			cout<<"    >> Dist2NearLeaf: level "<<searchNode->level<<": rotate points done! -> "<<omp_get_wtime()-stage_t<<endl;
		stage_t = omp_get_wtime();
	#endif


	if(NULL == searchNode->kid) {	// if leaf
		*outData = new binData();
		(*outData)->Copy(inData);
		*leaf = searchNode;
		 return;
	}
	else {		// not a leaf node
		// 1. project points along direction
		vector<double> projValue;
		projValue.resize(numof_query_points);
		if( 0 == strcmp(searchNode->options.splitter.c_str(), "rkdt") 
				&& searchNode->options.flag_r != 2) {
			int mvid = searchNode->coord_mv;
			#pragma omp parallel for
			for(int i = 0; i < numof_query_points; i++) 
				projValue[i] = inData->X[i*dim+mvid];
		}
		else {		// "rsmt" or "rkdt, flag_r = 2"
			int ONE = 1;
			#pragma omp parallel for
			for(int i = 0; i < numof_query_points; i++) 
				projValue[i] = ddot(&dim, &(searchNode->proj[0]), &ONE, &(inData->X[i*dim]), &ONE);
		}
	
		#if STAGE_OUTPUT_VERBOSE
			MPI_Barrier(comm);
			if(worldrank == 0 && searchNode->level < 3)
				cout<<"    >> Dist2NearLeaf: level "<<searchNode->level<<": project points done! -> "<<omp_get_wtime()-stage_t<<endl;
			stage_t = omp_get_wtime();
		#endif

		// 2. ponit_to_kid_membership
		int * point_to_kid_membership = new int [numof_query_points];
		double medValue = searchNode->median;
		int nsame = 0;
		#pragma omp parallel for
		for(int i = 0; i < numof_query_points; i++) {
			double diff = fabs( (projValue[i] - medValue)/medValue );
			if(diff < 1.0e-6) {
				nsame++;
				point_to_kid_membership[i] = -1;
			}
			else {
				if(projValue[i] < medValue)	{
					point_to_kid_membership[i] = 0;
				}
				else {
					point_to_kid_membership[i] = 1;
				}
			}
		}
		int pmove = 0;
		if(nsame > 0) {
			for(int i = 0; i < numof_query_points; i++) {
				if(point_to_kid_membership[i] == -1) {
					if(pmove < nsame / 2) {
						point_to_kid_membership[i] = 0;
					}
					else {
						point_to_kid_membership[i] = 1;
					}
					pmove++;
				}   // end if (-1)
			}
		}

	
		#if STAGE_OUTPUT_VERBOSE
			MPI_Barrier(comm);
			if(worldrank == 0 && searchNode->level < 3) 
				cout<<"    >> Dist2NearLeaf: level "<<searchNode->level<<": point to kid membership done! -> "<<omp_get_wtime()-stage_t<<endl;
			stage_t = omp_get_wtime();
		#endif

		// 3. local_rearrange points acc. to p2k_mem
		pre_all2all( &(inData->gids[0]), point_to_kid_membership, &(inData->X[0]), (long)numof_query_points, dim);
	
		#if STAGE_OUTPUT_VERBOSE
			MPI_Barrier(comm);
			if(worldrank == 0 && searchNode->level < 3)
				cout<<"    >> Dist2NearLeaf: level "<<searchNode->level<<": local rearrange done! -> "<<omp_get_wtime()-stage_t<<endl;
			stage_t = omp_get_wtime();
		#endif
	
		// 4. point_to_rank_membership
		int *send_count = new int [size];
		memset(send_count, 0, size*sizeof(int));
		group2rankDistribute(numof_query_points, &(searchNode->rank_colors[0]), 
                         size, point_to_kid_membership, 
                         send_count);
		
		#if STAGE_OUTPUT_VERBOSE
			MPI_Barrier(comm);
			if(worldrank == 0 && searchNode->level < 3)
				cout<<"    >> Dist2NearLeaf: level "<<searchNode->level<<": group2rank done! -> "<<omp_get_wtime()-stage_t<<endl;
			stage_t = omp_get_wtime();
		#endif
	
		// 5. repartition points
		double *new_X;
		long *new_gids;
		long new_N;

		#if COMM_TIMING_VERBOSE
			MPI_Barrier(comm);
			start_t = omp_get_wtime();
		#endif
		knn::repartition::repartition( &(inData->gids[0]), &(inData->X[0]), (long)numof_query_points, 
										send_count, dim, 
										&new_gids, &new_X, &new_N, comm);
		#if COMM_TIMING_VERBOSE
			Repartition_Query_T_ += omp_get_wtime() - start_t;
		#endif

		delete [] send_count;
		delete [] point_to_kid_membership;
		
		inData->X.resize(dim*new_N);
		inData->gids.resize(new_N);
		inData->numof_points = new_N;
		#pragma omp parallel for
		for(int i = 0; i < new_N; i++) {
			for(int j = 0; j < dim; j++)
				inData->X[i*dim+j] = new_X[i*dim+j];
			inData->gids[i] = new_gids[i];
		}

		delete [] new_X;
		delete [] new_gids;
	

		#if STAGE_OUTPUT_VERBOSE
			MPI_Barrier(comm);
			if(worldrank == 0 && searchNode->level < 3)
				cout<<"    >> Dist2NearLeaf: level "<<searchNode->level<<": repartition done! -> "<<omp_get_wtime()-stage_t<<endl;
			stage_t = omp_get_wtime();
		#endif
	
		MPI_Barrier(comm);
		oldtree::distributeToNearestLeaf( inData, searchNode->kid, outData, leaf);
	}	// else
	return;
}




