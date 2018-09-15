#include <cmath>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#include "repartition.h"
#include "gatherTree_ot.h"
#include "stTree.h"
#include "rotation.h"

using namespace knn;
using namespace knn::repartition;

// only gather median and coord
void oldtree::gatherTree(poldNode root, vector< pair<int, double> > & treearr)
{
	int rank, size, worldrank, worldsize;
	MPI_Comm_size(root->comm, &worldsize);
	MPI_Comm_rank(root->comm, &worldrank);
	
	int depth = int( ceil( log((double)worldsize) / log(2.0) ) );
	int residual = worldsize - (int)pow(2.0, (double)(depth-1));
	
	double *sendbuf = new double [2];
	int sendcnt;
	int *recvcnts = new int [worldsize];
	int *displs = new int [worldsize];
	int maxnodes = (int)pow(2.0, (double)(depth)) - 1;
	double *recvbuf = new double [maxnodes*2];
	
	// 1. collect the whole tree on root (rank 0)
	int offset = 0;
	poldNode curr_node = root;
	int flag = 1;
	while(flag) {
		MPI_Comm_rank(curr_node->comm, &rank);
		if(rank == 0) {
			sendcnt = 2;
			if(curr_node->kid != NULL) {
				sendbuf[0] = (double)curr_node->coord_mv;
				sendbuf[1] = curr_node->median;
			}
			else {
				sendbuf[0] = -1.0;
				sendbuf[1] = -1.0;
			}
		}
		else {
			sendcnt = 0;
		}


		offset = (int)pow(2.0, (double)(curr_node->level)) - 1;

		MPI_Gather(&sendcnt, 1, MPI_INT, recvcnts, 1, MPI_INT, 0, root->comm);

		displs[0] = 0;
		for(int i = 1; i < worldsize; i++)
			displs[i] = displs[i-1] + recvcnts[i-1];
		MPI_Gatherv(sendbuf, sendcnt, MPI_DOUBLE, recvbuf+offset*2, recvcnts, displs, MPI_DOUBLE, 0, root->comm);

		curr_node = curr_node->kid;
		MPI_Barrier(root->comm);
		
		if(curr_node == NULL) flag = 0;
		else if(curr_node->level == depth) flag = 0;
		else flag = 1;

	}	// end while
	
	// 2. bcast the whole tree to all processes
	MPI_Bcast(recvbuf, maxnodes*2, MPI_DOUBLE, 0, root->comm);

	// 3. decode the tree array
	treearr.resize(maxnodes);
	for(int i = 0; i < treearr.size(); i++) {
		treearr[i].first = (int)recvbuf[2*i+0];
		treearr[i].second = (double)recvbuf[2*i+1];
	}
	
	delete [] sendbuf;
	delete [] recvbuf;
	delete [] recvcnts;
	delete [] displs;

}


void oldtree::plantTree(vector< pair<int, double> >& ArrTree,
			   int depth,
			   pstNode inParent,
			   pstNode inNode)
{
	// 0. initialization
	if(inParent != NULL) {
		inNode->level = inParent->level + 1;
		inNode->parent = inParent;
	}

	// 1. plant the tree
	if(inNode->level == depth) {	// base case
		return;
	}
	else {
		// 1.1 copy median and coord_mv
		int pid = 0;
		if(inNode->parent != NULL) pid = inNode->parent->lnid;
		int offset = (int)pow(2.0, (double)(inNode->level)) - 1 + 2 * pid;
		int idx = offset + inNode->lnid % 2;
		inNode->coord_mv = ArrTree[idx].first;
		inNode->median = ArrTree[idx].second;

		// 1.2 create new kid node
		inNode->leftNode = new stNode(2*inNode->lnid+0);
		inNode->rightNode = new stNode(2*inNode->lnid+1);
		oldtree::plantTree(ArrTree, depth, inNode, inNode->leftNode);
		oldtree::plantTree(ArrTree, depth, inNode, inNode->rightNode);
	}
}


void oldtree::destroyTree(pstNode inNode)
{
	if(inNode != NULL) {
		oldtree::destroyTree(inNode->leftNode);
		oldtree::destroyTree(inNode->rightNode);
		delete inNode;
		inNode = NULL;
	}
}


void oldtree::printTree(pstNode inNode)
{
	if(NULL == inNode->leftNode && NULL == inNode->rightNode) {
		return;
	}
	else {
		cout<<"l: "<<inNode->level<<" - id: "<<inNode->lnid
			<<" - coord: "<<inNode->coord_mv
			<<" - median: "<<inNode->median
			<<endl;
		oldtree::printTree(inNode->leftNode);
		oldtree::printTree(inNode->rightNode);
	}
}


// count should be initialized as 0 before input this function
void oldtree::visitTree(double *points, int numof_points, int dim,
			   vector<int> & member_ids,
			   pstNode inNode, int depth,
			   int * point_to_visual_leaf_membership,
			   int * count)
{
	int worldrank;
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);

	if(inNode->parent == NULL) {
		int nc = (int)pow(2.0, (double)depth);
		for(int i = 0; i < nc; i++)
			count[i] = 0;
	}

	int numof_members = member_ids.size();

	vector<int> left_membership;
	vector<int> right_membership;

	if(inNode->coord_mv == -1) {
		left_membership.resize(numof_members);
		for(int i = 0; i < numof_members; i++)
			left_membership[i] = member_ids[i];
		count[inNode->lnid*2+1] = -1;
	}
	else {
		vector<int> equal_membership;
		left_membership.reserve(numof_members/2);
		right_membership.reserve(numof_members/2);
		for(int i = 0; i < numof_members; i++) {
			double px = points[ member_ids[i]*dim + inNode->coord_mv ];
			double diff = fabs((px - inNode->median)/inNode->median);
			if(diff < 1.0e-6) {
				equal_membership.push_back( member_ids[i] );
			}
			else if (px < inNode->median) {
				left_membership.push_back( member_ids[i] );
			}
			else {
				right_membership.push_back( member_ids[i] );
			}
		}
		int cur = equal_membership.size() / 2;
		for(int i = 0; i < cur; i++) 
			left_membership.push_back( equal_membership[i] );
		for(int i = cur; i < equal_membership.size(); i++)
			right_membership.push_back( equal_membership[i] );
	}
	
	if(inNode->level == depth-1) {
		for(int i = 0; i < left_membership.size(); i++) {
			point_to_visual_leaf_membership[ left_membership[i] ] = inNode->lnid*2+0;
			count[inNode->lnid*2+0]++;
		}
		for(int i = 0; i < right_membership.size(); i++) {
			point_to_visual_leaf_membership[ right_membership[i] ] = inNode->lnid*2+1;
			count[inNode->lnid*2+1]++;
		}
		return;
	}

	vector<int>().swap(member_ids);

	oldtree::visitTree(points, numof_points, dim, left_membership, inNode->leftNode, depth, 
			  point_to_visual_leaf_membership, count);
	oldtree::visitTree(points, numof_points, dim, right_membership, inNode->rightNode, depth, 
			  point_to_visual_leaf_membership, count);

}



void oldtree::repartitionQueryData(double *points, long * ids, int numof_points, int dim, poldNode inNode,
						  double **new_X, long ** new_gids, long &new_N)
{
	int worldsize, worldrank; 
	MPI_Comm_size(inNode->comm, &worldsize);
	MPI_Comm_rank(inNode->comm, &worldrank);

	// 0. ONLY ROTATE POINTS AT ROOT LEVEL
	if(inNode->level == 0 && inNode->options.flag_r == 1
			&& 0 == strcmp(inNode->options.splitter.c_str(), "rkdt") ) {
		double * tmpX = new double [numof_points*dim];
		memcpy(tmpX, points, sizeof(double)*numof_points*dim);
		rotatePoints(tmpX, numof_points, dim, inNode->rw, points);
		delete [] tmpX;
	}

	// 1. collect the tree
    vector< pair<int, double> > ArrTree;
	oldtree::gatherTree(inNode, ArrTree);

	// 2. rebuild the whole tree
	int depth = int( ceil( log((double)(ArrTree.size()+1)) / log(2.0) ) );
    pstNode root = new stNode();
	oldtree::plantTree(ArrTree, depth, NULL, root);
				
	// 3. travse the tree
	int * point_to_visual_leaf_membership = new int [numof_points];
	vector<int> member_ids(numof_points);
	for(int i = 0; i < numof_points; i++)
		member_ids[i] = i;
	int tmp_nc = (int)pow(2.0, (double)depth);
	int * count = new int [tmp_nc];
	oldtree::visitTree(points, numof_points, dim, member_ids, root, depth,
			  point_to_visual_leaf_membership, count);
	oldtree::destroyTree(root);

	// 4. local rearrage data accd. to membership
	pre_all2all(ids, point_to_visual_leaf_membership, points, (long)numof_points, dim);

	// 5. cal send_count
	int * send_count = new int [worldsize];
	int p = 0;
	for(int i = 0; i < tmp_nc; i++) {
		if(count[i] != -1) {
			send_count[p] = count[i];
			p++;
		}
	}

	// 5. repartition points
	#if COMM_TIMING_VERBOSE
		MPI_Barrier(inNode->comm);
		start_t = omp_get_wtime();
	#endif

		knn::repartition::repartition( ids, points, long(numof_points), send_count,
								   dim, new_gids, new_X, &new_N, inNode->comm);

	#if COMM_TIMING_VERBOSE
		Repartition_Query_T_ += omp_get_wtime() - start_t;
	#endif
	
	delete [] point_to_visual_leaf_membership;
	delete [] count;
	delete [] send_count;


}








