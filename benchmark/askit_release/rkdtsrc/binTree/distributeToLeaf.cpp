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
#include "distributeToLeaf.h"
#include "rotation.h"
#include "verbose.h"
//#include "random123wrapper.h"

using namespace std;
using namespace knn;
using namespace knn::repartition;

// this function is used for "mtree" and exact search. It need a search range "range"
// points could go to both kids
void bintree::distributeToLeaves(	pbinData inData, long rootNpoints, 
									double dupFactor, pbinNode searchNode, 
									double range, 
									pbinData *outData, pbinNode *leaf)
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
		bintree::distributeToLeaves( inData, rootNpoints, dupFactor, 
							searchNode->kid, range, outData, leaf);
    
    
	}	// else

	return;
  
}


// this function is used for "mtree" and exact search, each point has its own search range
// points can go to both kids
void bintree::distributeToLeaves( pbinData inData, long rootNpoints, 
						double dupFactor,
                        pbinNode searchNode, 
                        pbinData *outData, pbinNode *leaf)
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
				//if( (medValue - projValue[i]) < sqrt(inData->radii[i]) ) {
				double tmp = medValue - projValue[i];
				if( tmp*tmp <= inData->radii[i] ) {
					//members_in_kid[ searchNode->cluster_to_kid_membership[1] ].push_back(i);
					members_in_kid[ 1 ].push_back(i);
				}
			}
			else if( (projValue[i] - medValue) < 1.0e-6*medValue ) {
				members_in_kid[0].push_back(i);
				members_in_kid[1].push_back(i);
			}
			else {
				//members_in_kid[ searchNode->cluster_to_kid_membership[1] ].push_back(i);
				members_in_kid[ 1 ].push_back(i);
				//if( (projValue[i] - medValue) < sqrt(inData->radii[i]) ) {
				double tmp = projValue[i] - medValue;
				if( tmp*tmp <= inData->radii[i] ) {
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
   		
		#if PCL_DEBUG_VERBOSE
			for(int i = 0; i < numof_kids; i++) {
				cout<<"("<<worldrank<<") (kid "<<i<<") ";
				for(int j = 0; j < members_in_kid[i].size(); j++)
					cout<<inData->gids[ members_in_kid[i][j] ]<<" ";
				cout<<endl;
			}
			cout<<endl;
			MPI_Barrier(comm);
		#endif


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
   		
		#if STAGE_OUTPUT_VERBOSE
			MPI_Barrier(comm);
			if(worldrank == 0 && searchNode->level < 3)
				cout<<"    >> Dist2Leaf: level "<<searchNode->level<<": group2rank done! -> "<<omp_get_wtime()-stage_t<<endl;
			stage_t = omp_get_wtime();
		#endif


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
				cout<<"    >> Dist2Leaf: level "<<searchNode->level<<": repartition done! -> "
					<<omp_get_wtime()-stage_t
					<<" newN: "<<re_N
					<<endl;
			stage_t = omp_get_wtime();
		#endif

		MPI_Barrier(comm);
		bintree::distributeToLeaves( inData, rootNpoints, dupFactor,
								searchNode->kid, outData, leaf); 
	}	// else
	
	return;
}



// this one is used for both "mtree" or "rkdt"
// for "rsmt", we project points onto direction "searchNode->proj"
// for "rkdt, flag_r == 1", we also should project points on direction "searchNode->proj"
// for "rkdt, flag_r == 2", we only rotate points on the root, other level direct use coord_mv
// for "rkdt, flag_r == 0", we use direct coord_mv on all level
void bintree::distributeToNearestLeaf( pbinData inData, pbinNode searchNode, 
									   pbinData *outData, pbinNode *leaf)
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

	#if STAGE_OUTPUT_VERBOSE
		MPI_Barrier(comm);
		//MPI_Barrier(MPI_COMM_WORLD);
		if(worldrank == 0)
			cout<<"    >> Dist2NearLeaf: level "<<searchNode->level<<": enter func ! -> "<<omp_get_wtime()-stage_t<<endl;
		stage_t = omp_get_wtime();
	#endif
	
	if(size > 1) {
		knn::repartition::loadBalance_arbitraryN(inData->X, inData->gids, inData->numof_points,
											inData->dim, inData->numof_points, comm);
	}

	#if STAGE_OUTPUT_VERBOSE
		MPI_Barrier(comm);
		//MPI_Barrier(MPI_COMM_WORLD);
		//int max_nlq, min_nlq;
		//MPI_Reduce(&inData->numof_points, &max_nlq, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
		//MPI_Reduce(&inData->numof_points, &min_nlq, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
		if(worldrank == 0) {
			cout<<"    >> Dist2NearLeaf: level "<<searchNode->level
				<<": load balance done ! -> "<<omp_get_wtime()-stage_t
				//<<"  min="<<min_nlq<<"  max="<<max_nlq
				<<endl;
		}
		stage_t = omp_get_wtime();
	#endif
	
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
		//MPI_Barrier(MPI_COMM_WORLD);
		MPI_Barrier(comm);
		if(worldrank == 0)
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
			//MPI_Barrier(MPI_COMM_WORLD);
			if(worldrank == 0)
				cout<<"    >> Dist2NearLeaf: level "<<searchNode->level<<": project points done! -> "<<omp_get_wtime()-stage_t<<endl;
			stage_t = omp_get_wtime();
		#endif

		// 2. ponit_to_kid_membership
		int * point_to_kid_membership = new int [numof_query_points];
		double medValue = searchNode->median;
		int nsame = 0;
		#pragma omp parallel for reduction(+:nsame)
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
				}	// end if (-1)
			}
		}
	
		#if STAGE_OUTPUT_VERBOSE
			MPI_Barrier(comm);
			//MPI_Barrier(MPI_COMM_WORLD);
			if(worldrank == 0) 
				cout<<"    >> Dist2NearLeaf: level "<<searchNode->level<<": point to kid membership done! -> "<<omp_get_wtime()-stage_t<<endl;
			stage_t = omp_get_wtime();
		#endif

		// 3. local_rearrange points acc. to p2k_mem
		pre_all2all( &(inData->gids[0]), point_to_kid_membership, &(inData->X[0]), (long)numof_query_points, dim);
	
		#if STAGE_OUTPUT_VERBOSE
			MPI_Barrier(comm);
			//MPI_Barrier(MPI_COMM_WORLD);
			if(worldrank == 0)
				cout<<"    >> Dist2NearLeaf: level "<<searchNode->level<<": local rearrange done! -> "<<omp_get_wtime()-stage_t<<endl;
			stage_t = omp_get_wtime();
		#endif
	
		#if COMM_TIMING_VERBOSE
			MPI_Barrier(comm);
			start_t = omp_get_wtime();
		#endif

		int new_N = tree_repartition_arbitraryN(inData->gids, inData->X, inData->numof_points,
											point_to_kid_membership, &(searchNode->rank_colors[0]), dim, comm);
		inData->numof_points = new_N;

		#if COMM_TIMING_VERBOSE
			Repartition_Query_T_ += omp_get_wtime() - start_t;
		#endif

		delete [] point_to_kid_membership;
		

		#if STAGE_OUTPUT_VERBOSE
			MPI_Barrier(comm);
			//MPI_Barrier(MPI_COMM_WORLD);
			int max_nreq, min_nreq;
			//MPI_Reduce(&inData->numof_points, &max_nreq, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
			//MPI_Reduce(&inData->numof_points, &min_nreq, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
			if(worldrank == 0)
				cout<<"    >> Dist2NearLeaf: level "<<searchNode->level
					<<": repartition done! -> "<<omp_get_wtime()-stage_t
					//<<"   min="<<min_nreq<<"  max="<<max_nreq
					<<endl;
			stage_t = omp_get_wtime();
		#endif
	
		MPI_Barrier(comm);
		bintree::distributeToNearestLeaf( inData, searchNode->kid, outData, leaf);
	}	// else
	return;
}


// use median splitting
void bintree::GoToNearestLeafbyMedian( pbinData inData, pbinNode searchNode,
									   pbinData *outData, pbinNode *leaf)
{
	int size, rank;
	MPI_Comm comm = searchNode->comm;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);
	int worldsize, worldrank;
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);

    double start_t = 0.0;

    if(size > 1) {
		knn::repartition::loadBalance_arbitraryN(inData->X, inData->gids, inData->numof_points,
									             inData->dim, inData->numof_points, comm);
	}

	int numof_kids = 2;
	int dim = inData->dim;
	int numof_query_points = inData->numof_points;

    if(NULL == searchNode->kid) {	// if leaf
		*outData = new binData();
		(*outData)->Copy(inData);
		*leaf = searchNode;
		 return;
	}
    else {	// not a leaf node
		// 1. project points along direction
		vector<double> projValue;
		projValue.resize(numof_query_points);
		int ONE = 1;
		#pragma omp parallel for
		for(int i = 0; i < numof_query_points; i++)
			projValue[i] = ddot(&dim, &(searchNode->proj[0]), &ONE, &(inData->X[i*dim]), &ONE);

		// 2. ponit_to_kid_membership
		int * point_to_kid_membership = new int [numof_query_points];
		double medValue = searchNode->median;
		int nsame = 0;
		#pragma omp parallel for reduction(+:nsame)
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
				}	// end if (-1)
			}
		}

		// 3. local_rearrange points acc. to p2k_mem
		pre_all2all( &(inData->gids[0]), point_to_kid_membership, &(inData->X[0]), (long)numof_query_points, dim);

		#if COMM_TIMING_VERBOSE
			MPI_Barrier(comm);
			start_t = omp_get_wtime();
		#endif

		int new_N = tree_repartition_arbitraryN(inData->gids, inData->X, inData->numof_points,
											point_to_kid_membership, &(searchNode->rank_colors[0]), dim, comm);
		inData->numof_points = new_N;

		#if COMM_TIMING_VERBOSE
			Repartition_Query_T_ += omp_get_wtime() - start_t;
		#endif

		delete [] point_to_kid_membership;

		MPI_Barrier(comm);
		bintree::GoToNearestLeafbyMedian(inData, searchNode->kid, outData, leaf);
	}	// else
	return;
}



// --- funcs used to distribute to random sampled node

void bintree::randperm(int m, int N, vector<long>& arr)
{

	if(m > N) {
		cerr<<" m must <= N"<<endl;
		return;
	}

	arr.resize(m);
	for(int i = 0; i <arr.size(); i++) {
		double tmp = floor( (double)N*(double)rand()/(double)RAND_MAX );
		arr[i] = (long)tmp;
	}
	sort(arr.begin(), arr.end());
	vector<long>::iterator it1 = unique(arr.begin(), arr.end());
	arr.resize(it1 - arr.begin());

	int pp = m;
	while(arr.size() < m) {
		pp++;
		double tmp = floor( (double)N*(double)rand()/(double)RAND_MAX );
		arr.push_back((long)tmp);
		sort(arr.begin(), arr.end());
		vector<long>::iterator it2 = unique(arr.begin(), arr.end());
		arr.resize(it2 - arr.begin());
	}
}


void bintree::uniformSample(double *points, int numof_points, int dim,
				   int numof_sample_points,
				   double *sample_points, long *sample_ids,
				   MPI_Comm comm)
{
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	// 1. number points in this comm
	long tmp_numof_points = numof_points;
	long offset;
	MPI_Scan(&tmp_numof_points, &offset, 1, MPI_LONG, MPI_SUM, comm);
	offset -= tmp_numof_points;
	vector<long> IDs(numof_points);
	#pragma omp parallel for
	for(int i = 0; i < numof_points; i++)
		IDs[i] = offset + (long)i;

	// 2. sample id
	int glb_numof_points;
	MPI_CALL(MPI_Allreduce( &numof_points, &glb_numof_points, 1, MPI_INT, MPI_SUM, comm));
	vector<long> sampleIDs(numof_sample_points);
	randperm(numof_sample_points, glb_numof_points, sampleIDs);
	sort(sampleIDs.begin(), sampleIDs.end());
	MPI_Bcast(&(sampleIDs[0]), numof_sample_points, MPI_LONG, 0, comm);
	for(int i = 0; i < numof_sample_points; i++) sample_ids[i] = sampleIDs[i];

	// 3. sample points locally
	vector< pair<long, int> > found;
	found.reserve(numof_sample_points);
	for(int i = 0; i < numof_sample_points; i++) {
		for(int j = 0; j < numof_points; j++) {
			if(IDs[j] == sampleIDs[i])
				found.push_back( make_pair<long, int>(IDs[j], j) );
		}
	}
	int numof_found = found.size();
	double *localSamples = new double [numof_sample_points*dim];
	for(int i = 0; i < numof_found; i++) {
		for(int j = 0; j < dim; j++) {
			localSamples[i*dim+j] = points[found[i].second*dim+j];
		}
	}

	// 4. distribute sampled points to other process
	int sendcount = numof_found*dim;
	int *rcvcounts = new int [size];
	int *rcvdisp = new int [size];
	MPI_CALL(MPI_Allgather(&sendcount, 1, MPI_INT, rcvcounts, 1, MPI_INT, comm));
	omp_par::scan(rcvcounts, rcvdisp, size);
	assert( (rcvdisp[size-1]+rcvcounts[size-1])/dim == numof_sample_points);
	MPI_CALL(MPI_Allgatherv(	localSamples, sendcount, MPI_DOUBLE, sample_points,
					rcvcounts, rcvdisp, MPI_DOUBLE, comm ));

	delete [] localSamples;
	delete [] rcvcounts;
	delete [] rcvdisp;

}



void bintree::sampleTwoKids(pbinNode inNode, int numof_sample_points, 
		             double * sample0, double *sample1)
{
	int worldrank, worldsize;
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
	if(NULL == inNode->kid) {    // if leaf
		cerr<<"worldrank "<<worldrank<<" is already on a leaf, stop!"<<endl;
		return;
	}


	int rank, size;
	MPI_Comm comm = inNode->comm;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	// 1. get the ref points stored in the current rank
	pbinNode current = inNode;
	while ( current->kid )			// go to leaf
		current = current->kid;
	int dim = current->data->dim;
	int numof_points = current->data->X.size() / dim;

	// 2. sample points from the input node's kid
	vector<long> dummy_ids(numof_sample_points);
	bintree::uniformSample( &(current->data->X[0]), numof_points, dim,
							numof_sample_points, 
							sample0, &(dummy_ids[0]),
							inNode->kid->comm);


	// 3. rank 0 has sample points from kid 0, collect points form kid 1
	//    then broadcast to other rank
	//	ex:	in fact, rank 0 and rank size-1 should belongs to different node, need to verify
	vector<int>::iterator it = find(inNode->rank_colors.begin(), inNode->rank_colors.end(), 1);
	int rank_sendfrom = it - inNode->rank_colors.begin();
	MPI_Status recvstatus;
	if(rank == rank_sendfrom) 
		MPI_Send(sample0, numof_sample_points*dim, MPI_DOUBLE, 0, TAG_R, comm);
	if(rank == 0)
		MPI_Recv(sample1, numof_sample_points*dim, MPI_DOUBLE, rank_sendfrom, TAG_R, comm, &recvstatus);
	MPI_Barrier(comm);
	MPI_Bcast(sample0, numof_sample_points*dim, MPI_DOUBLE, 0, comm);
	MPI_Bcast(sample1, numof_sample_points*dim, MPI_DOUBLE, 0, comm);

	if(inNode->options.debug_verbose == 6) {
		cout<<"rank "<<rank<<" sample 0: ";
		for(int i = 0; i < numof_sample_points; i++) {
			for(int j = 0; j < dim; j++)
				cout<<sample0[i*dim+j]<<" ";
			cout<<";";
		}
		cout<<endl;
		cout<<"rank "<<rank<<" sample 1: ";
		for(int i = 0; i < numof_sample_points; i++) {
			for(int j = 0; j < dim; j++)
				cout<<sample1[i*dim+j]<<" ";
			cout<<";";
		}
		cout<<endl;
	}
	
}



// distribute to the nearest leaf according to sampled points
void bintree::distributeToSampledLeaf( pbinData inData, 
								pbinNode searchNode, 
								pbinData *outData, 
								pbinNode *leaf)
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
		*outData = new binData();
		(*outData)->Copy(inData);
		*leaf = searchNode;
		 return;
	}
	else {		// not a leaf node
		int numof_sample_points = ceil( sqrt((double)dim)*log10((double)searchNode->Nglobal)/log10(2.0) );

		vector<long> * members_in_kid = new vector<long> [numof_kids];
		for(int i = 0; i < numof_kids; i++)
			members_in_kid[i].reserve(numof_query_points/numof_kids);
		
		// 1. sample some points from two kids seperately;
		vector<double> sample0(numof_sample_points*dim);
		vector<double> sample1(numof_sample_points*dim);
		bintree::sampleTwoKids(searchNode, numof_sample_points, &(sample0[0]), &(sample1[0]));
		
		// 2. compute the hausdorff distance
		pair<double, long> *haus0 = new pair<double, long> [numof_query_points];
		pair<double, long> *haus1 = new pair<double, long> [numof_query_points];
		knn::directKQueryLowMem(&(sample0[0]), &(inData->X[0]), numof_sample_points, numof_query_points, 1, dim, haus0);
		knn::directKQueryLowMem(&(sample1[0]), &(inData->X[0]), numof_sample_points, numof_query_points, 1, dim, haus1);

		//vector<double> dist0;
		//vector<double> dist1;
		//dist0.resize(numof_query_points*numof_sample_points);
		//dist1.resize(numof_query_points*numof_sample_points);
		//knn::compute_distances(&(sample0[0]), &(inData->X[0]), numof_sample_points,
		//							numof_query_points, dim, &(dist0[0]));
		//knn::compute_distances(&(sample1[0]), &(inData->X[0]), numof_sample_points,
		//							numof_query_points, dim, &(dist1[0]));

		// 2. assign each query points to the cluster has the minimum hausdorff distance
		for(int i = 0; i < numof_query_points; i++) {
			//double haus0 = *min_element(dist0.begin()+i*numof_sample_points, dist0.begin()+(i+1)*numof_sample_points);
			//double haus1 = *min_element(dist1.begin()+i*numof_sample_points, dist1.begin()+(i+1)*numof_sample_points);

			if(haus0[i].first < haus1[i].first) {
				members_in_kid[ 0 ].push_back(i);
			}
			else {
				members_in_kid[ 1 ].push_back(i);
			}
		}

		delete [] haus0;
		delete [] haus1;

		// 3. perpare new query data
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
	
		// 4. repartition data
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
		bintree::distributeToSampledLeaf( inData, searchNode->kid, outData, leaf);
	}	// else
	return;
}


