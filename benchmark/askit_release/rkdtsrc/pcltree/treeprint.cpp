#include <mpi.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include "mpitree.h"
#include "treeprint.h"


using namespace std;
//using namespace knn;

void treePrint(pMTNode in_node)
{
	MPI_Comm in_comm = in_node->comm;
	int nproc, myrank;
	MPI_Comm_rank(in_comm, &myrank);
	MPI_Comm_size(in_comm, &nproc);

	int tmp_p_id = 0;

	if(NULL == in_node->kid) {	// if leaf
		int tmp_p_cid = -1;
		if(NULL != in_node->parent) tmp_p_cid = in_node->parent->chid;
		cout<<"level "<<in_node->level
			<<" cid "<<in_node->chid
			<<" my_parent_chid "<<tmp_p_cid
			<<" my_rank "<<myrank<<" p ";
		for(int i = 0; i < (*(in_node->data)).gids.size(); i++)
			cout<<(*(in_node->data)).gids[i]<<" ";
		cout<<endl;
		return;
	}
	else {	// if not leaf
		cout<<"not a leaf"<<endl;
		if( 0 == myrank ) {
			if(NULL != in_node->parent) tmp_p_id = in_node->parent->chid;
			cout<<"level "<<in_node->level
				<<" cid "<<in_node->chid
				<<" my_parent_chid "<<tmp_p_id
				<<" my_rank "<<myrank<<" c ";
			for(int i = 0; i < in_node->C.size(); i++)
				cout<<in_node->C[i]<<" ";
			for(int i = 0; i < in_node->cluster_to_kid_membership.size(); i++)
				cout<<in_node->cluster_to_kid_membership[i]<<" ";
			cout<<endl;
		}

		treePrint(in_node->kid);

	}

}


void treeSave(ofstream &outfile, pMTNode in_node)
{
	int nproc, myrank;
	MPI_Comm in_comm = in_node->comm;
	MPI_Comm_rank(in_comm, &myrank);
	MPI_Comm_size(in_comm, &nproc);

	int p_id = -1;
	if(NULL == in_node->kid) {	// if leaf
		if(NULL != in_node->parent) p_id = in_node->parent->chid;
		outfile<<"level "<<in_node->level
			<<" cid "<<in_node->chid
			<<" pid "<<p_id<<" p ";
		for(int i = 0; i < (*(in_node->data)).gids.size(); i++)
			outfile<<(*(in_node->data)).gids[i]<<" ";
		outfile<<endl;
		return;
	}
	else {	// if not leaf
//		if( 0 == myrank ) {
			if(NULL != in_node->parent) p_id = in_node->parent->chid;
			outfile<<"level "<<in_node->level
				<<" cid "<<in_node->chid
				<<" pid "<<p_id<<" c ";
			for(int i = 0; i < in_node->C.size(); i++)
				outfile<<in_node->C[i]<<" ";
			for(int i = 0; i < in_node->cluster_to_kid_membership.size(); i++)
				outfile<<in_node->cluster_to_kid_membership[i]<<" ";
			outfile<<endl;
//		}

		treeSave(outfile, in_node->kid);
	}
}

void treeSaveRadii(ofstream &outfile, pMTNode in_node)
{
	int nproc, myrank;
	MPI_Comm in_comm = in_node->comm;
	MPI_Comm_rank(in_comm, &myrank);
	MPI_Comm_size(in_comm, &nproc);

	int p_id = -1;
	if(NULL == in_node->kid) {	// if leaf
		if(NULL != in_node->parent) p_id = in_node->parent->chid;
		outfile<<"level "<<in_node->level
			<<" cid "<<in_node->chid
			<<" pid "<<p_id<<" p ";
		for(int i = 0; i < (*(in_node->data)).gids.size(); i++)
			outfile<<(*(in_node->data)).gids[i]<<" ";
		outfile<<endl;
		return;
	}
	else {	// if not leaf
//		if( 0 == myrank ) {
			if(NULL != in_node->parent) p_id = in_node->parent->chid;
			outfile<<"level "<<in_node->level
				<<" cid "<<in_node->chid
				<<" pid "<<p_id<<" c ";
			for(int i = 0; i < in_node->C.size(); i++)
				outfile<<in_node->C[i]<<" ";
			for(int i = 0; i < in_node->R.size(); i++)
				outfile<<in_node->R[i]<<" ";
			for(int i = 0; i < in_node->cluster_to_kid_membership.size(); i++)
				outfile<<in_node->cluster_to_kid_membership[i]<<" ";
			outfile<<endl;
//		}

		treeSaveRadii(outfile, in_node->kid);
	}
}
