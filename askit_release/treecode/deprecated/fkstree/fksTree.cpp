#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>

#include "ompUtils.h"
#include "binTree.h"
#include "binQuery.h"
#include "blas.h"
#include "repartition.h"
#include "generator.h"
#include "clustering.h"
#include "direct_knn.h"
#include "rotation.h"

#include "fksTree.h"

using namespace std;
using namespace knn;
using namespace knn::repartition;

#define _DEBUG_ true

void print(vector<long> &queryIDs, vector< pair<double, long> > *knns, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
	MPI_Barrier(comm);

    int k = knns->size() / queryIDs.size();
    for(int r = 0; r < size; r++) {
        if(r == rank) {
            cout<<"queryIDs.size = "<<queryIDs.size()<<", knns.size = "<<knns->size()<<endl;
	        for(int i = 0; i < queryIDs.size(); i++) {
		        cout<<"(rank "<<rank<<") "<<queryIDs[i]<<": ";
		        for(int j = 0; j < k; j++)
			        cout<<(*knns)[i*k+j].second<<" ";
		        cout<<endl;
	        }
            cout.flush();
        }
	    MPI_Barrier(comm);
    }
    if(rank == 0) cout<<endl;
}

// ======== fks_mpiNode member functions =======

void fks_mpiNode::Insert(fks_mpiNode *in_parent, int maxp, int maxLevel, int minCommSize, MPI_Comm inComm, binData *inData)
{
	int worldsize, worldrank;
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);

	// input checks
	int numof_kids = 2;
	assert( maxp > 1 );
	assert( maxLevel >= 0 && maxLevel <= options.max_max_treelevel);

	comm = inComm;
	int size, rank;
    MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank); // needed to print information
	int dim = inData->dim;
	MPI_Allreduce(&(inData->numof_points), &Nglobal, 1, MPI_INT, MPI_SUM, comm);

	vector<double> &X = inData->X;
	vector<long> &gids= inData->gids;

	// Initializations
	int its_child_id = 0;
	int n_over_p = Nglobal / size;

	knn::repartition::loadBalance_arbitraryN(inData->X, inData->gids,
        inData->numof_points, inData->dim, inData->numof_points, comm);

    if(in_parent!=NULL)  {
		level = in_parent->level + 1;
		fks_parent = in_parent;
		its_child_id = chid;
	}

	// BASE CASE TO TERMINATE RECURSION
	if (size <= minCommSize || level == maxLevel || Nglobal <= maxp) {
		data = new binData;
		data->Copy(inData);

		MPI_Barrier(MPI_COMM_WORLD);
		return;
	}   // end of base case

	int numof_clusters = 2;
	vector<int> point_to_cluster_membership(inData->numof_points);
	vector<int> local_numof_points_per_cluster(numof_clusters);
	vector<int> global_numof_points_per_cluster(numof_clusters);

	coord_mv = -1;
	proj.resize(dim);
	mtreeSplitter(&X[0], inData->numof_points, dim, &(proj[0]), median,
				    &(point_to_cluster_membership[0]),
				    &(local_numof_points_per_cluster[0]),
				    &(global_numof_points_per_cluster[0]),
				    comm);

	int my_rank_color;
	rank_colors.resize(size);
	if(size % 2 == 0) {  // Even split
		my_rank_color = (rank < size/2) ? 0 : 1;
		for(int i = 0; i < size; i++)
			rank_colors[i] = (i < size/2) ? 0 : 1;
	}
    else {
		my_rank_color = (rank <= size/2) ? 0 : 1;
        for(int i = 0; i < size; i++)
			rank_colors[i] = (i <= size/2) ? 0 : 1;
	}

	pre_all2all(&(gids[0]), &(point_to_cluster_membership[0]), &(X[0]), (long)(inData->numof_points), dim);

	int newN = tree_repartition_arbitraryN(inData->gids, inData->X,
            inData->numof_points, &(point_to_cluster_membership[0]),
            &(rank_colors[0]), dim, comm);
	inData->numof_points = newN;

	//6. create new communicator
	MPI_Comm new_comm = MPI_COMM_NULL;
	if( MPI_SUCCESS != MPI_Comm_split( comm, my_rank_color, rank, &new_comm) )
		assert(NULL);
	assert(new_comm != MPI_COMM_NULL);

	//7. Create new node and insert new data
	fks_kid = new fks_mpiNode(its_child_id);
    if(my_rank_color == 0) {    // left child
        fks_kid->node_morton = ( node_morton & (~(1 << (level+1))) );
    }
    else {
        fks_kid->node_morton = ( node_morton | (1 << (level+1)) );
    }
	fks_kid->options.hypertree = options.hypertree;
	fks_kid->options.flag_r = options.flag_r;
	fks_kid->options.flag_c = options.flag_c;
	fks_kid->options.pruning_verbose = options.pruning_verbose;
	fks_kid->options.timing_verbose = options.timing_verbose;
	fks_kid->options.splitter = options.splitter;
	fks_kid->options.debug_verbose = options.debug_verbose;
	fks_kid->Insert(this, maxp, maxLevel, minCommSize, new_comm, inData);
};


fks_mpiNode::~fks_mpiNode()
{
	if(this->fks_parent != NULL) {  // if this is not root
		MPI_Barrier(comm);
		MPI_Comm_free(&comm);
	}

	if(NULL == this->fks_kid && NULL != this->data) {   // if leaf
		delete this->data;
        this->data = NULL;
        delete this->skeletons;
        this->skeletons = NULL;
	}
	else {
        if(this->skeletons != NULL) {
            delete this->skeletons;
            this->skeletons = NULL;
        }
		delete this->fks_kid;
	}
}





// ======== fksTree member functions =========

fksTree::~fksTree()
{
    if(root_mpi != NULL) {
        delete root_mpi;
        root_mpi = NULL;
    }
    if(root_omp != NULL) {
        delete root_omp;
        root_omp = NULL;
    }
    if(inLeafData != NULL) {
        delete inLeafData;
        inLeafData = NULL;
    }
    if(exclKNNofLeaf != NULL) {
        delete exclKNNofLeaf;
        exclKNNofLeaf = NULL;
    }
}

void fksTree::build(fksData *inData, void *ctx)
{
    fksCtx *user = (fksCtx*) ctx;

    int rank, size;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    int dim = inData->dim;

    // .1 build the tree
    root_mpi = new fks_mpiNode();
    root_mpi->options.hypertree = 1;        // point-wise data exchange, alltoall if set to 0
    root_mpi->options.splitter = "rsmt";    // use metric tree splitting
    root_mpi->options.flag_r = 0;           // do not rotate
    root_mpi->options.flag_c = 0;           // do not use in fksTree
    root_mpi->Insert( NULL, user->fks_mppn, user->fks_maxLevel, user->minCommSize, comm, inData );

    if(_DEBUG_) {
        cout<<"rank "<<rank<<": build the tree done "<<endl;
    }


    // .2 collect the charges for all inLeafData points
    fks_mpiNode *curr = root_mpi;
    while(curr->fks_kid != NULL) {
        curr = curr->fks_kid;
    }
    vector< pair<double, long> > allchargeid;
    allchargeid.resize(curr->data->numof_points);
    #pragma omp parallel for
    for(int i = 0; i < curr->data->numof_points; i++) {
        allchargeid[i].first = 0.0;     // unuseful for charges
        allchargeid[i].second = curr->data->gids[i];
    }
    double *chargesForLeafNode = new double [allchargeid.size()];
    bintree::getNeighborCoords(inData->numof_points, &(inData->charges[0]), 1,
                               allchargeid.size(), 1, allchargeid, chargesForLeafNode, comm);

    if(_DEBUG_) {
        for(int k = 0; k < size; k++) {
            if(k == rank) {
                cout<<"rank "<<rank<<": collect charges done "<<endl;
	            for(int i = 0; i < curr->data->numof_points; i++) {
		            cout<<"(rank "<<rank<<") "<<curr->data->gids[i]<<": ("
                        <<chargesForLeafNode[i]<<") ";
		            for(int j = 0; j < dim; j++)
			            cout<<curr->data->X[i*dim+j]<<" ";
		            cout<<endl;
	            }
                cout.flush();
            }
	        MPI_Barrier(comm);
        }
        if(rank == 0) cout<<endl;
    }


    // .3 find knn for points in each leaf
    // inLeafData and exclKNNofLeaf will be constructed in this function
    find_knn_for_leaves(ctx);
    inLeafData->charges.resize( inLeafData->numof_points );
    #pragma omp parallel for
    for(int i = 0; i < inLeafData->numof_points; i++)
        inLeafData->charges[i] = chargesForLeafNode[i];
    delete [] chargesForLeafNode;

    if(_DEBUG_) {
        cout<<"rank "<<rank<<": find knn for each leaf done "<<endl;
        print(inLeafData, comm);
    }

    // .4 collect charges for data in exclKNNofLeaf
    allchargeid.resize( exclKNNofLeaf->numof_points );
    #pragma omp parallel for
    for(int i = 0; i < exclKNNofLeaf->numof_points; i++) {
        allchargeid[i].first = 0.0;     // unuseful for charges
        allchargeid[i].second = exclKNNofLeaf->gids[i];
    }
    exclKNNofLeaf->charges.resize( exclKNNofLeaf->numof_points );
    bintree::getNeighborCoords(inLeafData->numof_points, &(inLeafData->charges[0]), 1,
                               exclKNNofLeaf->numof_points, 1, allchargeid,
                               &(exclKNNofLeaf->charges[0]), comm);

    if(_DEBUG_) {
        cout<<"rank "<<rank<<": collect charges for excl knn done "<<endl;
        print(exclKNNofLeaf, comm);
    }

    // .4 morton id
    int worldsize;
    MPI_Comm_size(root_mpi->comm, &worldsize);
    if(worldsize == 1) {    // if only one processor
        // .4.1 morton id
        inLeafData->mortons.resize(inLeafData->numof_points);
        for(int i = 0; i < inLeafData->numof_points; i++)
            inLeafData->mortons[i] = 0;
        exclKNNofLeaf->mortons.resize(inLeafData->numof_points);
        for(int i = 0; i < exclKNNofLeaf->numof_points; i++)
            exclKNNofLeaf->mortons[i] = 0;
    }
    else {
        // .4.1 gather the whole distributed tree, and plant it
        int depth = int( ceil( log((double)worldsize) / log(2.0) ) );
        vector<fksTreeInfo> treeArr;
        gatherDistributedTree(root_mpi, dim, treeArr);

        if(_DEBUG_) {
            for(int r = 0; r < size; r++) {
                if(r == rank) {
                    cout<<"(rank "<<rank<<") gather whole distributed tree done"
                        <<", treeArr.size = "<<treeArr.size()<<endl;
                    for(int i = 0; i < treeArr.size(); i++) {
                        for(int j = 0; j < treeArr[i].proj.size(); j++)
                            cout<<treeArr[i].proj[j]<<" ";
                        cout<<" median = "<<treeArr[i].median<<endl;
                    }
                }
                MPI_Barrier(comm);
            }
        }

        fks_ompNode *dtree_root = new fks_ompNode();
        plantTree(treeArr, depth, NULL, dtree_root);

        if(_DEBUG_) {
            for(int r = 0; r < size; r++) {
                if(r == rank) {
                    cout<<"rank "<<rank<<": plant distributed tree done "<<endl;
                    printTree(dtree_root);
                }
                cout<<endl;
                MPI_Barrier(comm);
            }
        }

        // .4.2 find morton id for each point
        inLeafData->mortons.resize(inLeafData->numof_points);
        for(int i = 0; i < inLeafData->numof_points; i++)
            inLeafData->mortons[i] = mortonID( &(inLeafData->X[i*dim]), dim, 0, 0, dtree_root);
        exclKNNofLeaf->mortons.resize(inLeafData->numof_points);
        for(int i = 0; i < exclKNNofLeaf->numof_points; i++)
            exclKNNofLeaf->mortons[i] = mortonID( &(exclKNNofLeaf->X[i*dim]), dim, 0, 0, dtree_root);
        delete dtree_root;
    }

    // .5 build the local subtree
    vector<long> *active_set = new vector<long>();
    active_set->resize(inLeafData->numof_points);
    for(int i = 0; i < inLeafData->numof_points; i++)
        (*active_set)[i] = (long)i;
    root_omp = new fks_ompNode();
    root_omp->node_morton = curr->node_morton;
    root_omp->insert(NULL, inLeafData, active_set, curr->level, user->fks_mppn, user->fks_maxLevel-curr->level);

    if(_DEBUG_) {
        cout<<"rank "<<rank<<": build local tree done "<<endl;
        printTree(root_omp);
    }

    // .6 get morton id for exclKNN data
    double *tmp_mid_tosend = new double [inLeafData->numof_points];
    double *tmp_mid_torecv = new double [exclKNNofLeaf->numof_points];
    #pragma omp parallel for
    for(int i = 0; i < inLeafData->numof_points; i++)
        tmp_mid_tosend[i] = (double)inLeafData->mortons[i];
    bintree::getNeighborCoords(inLeafData->numof_points, tmp_mid_tosend, 1,
                               exclKNNofLeaf->numof_points, 1, allchargeid,
                               tmp_mid_torecv, comm);
    #pragma omp parallel for
    for(int i = 0; i < exclKNNofLeaf->numof_points; i++)
        exclKNNofLeaf->mortons[i] = (long)tmp_mid_torecv[i];
    delete [] tmp_mid_tosend;
    delete [] tmp_mid_torecv;

    if(_DEBUG_) {
        cout<<"rank "<<rank<<": morton id totally done "<<endl;
        cout<<"(rank "<<rank<<") full inLeafData: "<<endl;
        print(inLeafData, comm);
        cout<<"(rank "<<rank<<") full exclKNNofLeaf: "<<endl;
        print(exclKNNofLeaf, comm);
    }

}

void fksTree::find_knn_for_leaves(void *ctx)
{
    fksCtx *user = (fksCtx*) ctx;

    int rank, size;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    // .1 first refer inLeafData to the data in leaf node
    fks_mpiNode *curr = root_mpi;
    while(curr->fks_kid != NULL) {
        curr = curr->fks_kid;
    }
    inLeafData = new fksData();
    inLeafData->Copy(curr->data);
    int dim = curr->data->dim;

    // .2 reassign global id for knn, 0 from rank 0, and ...
    long offset, np;
    np = inLeafData->numof_points;
	MPI_Scan( &np, &offset, 1, MPI_LONG, MPI_SUM, comm );
	offset -= np;
    #pragma omp parallel for
    for(int i = 0; i < inLeafData->numof_points; i++)
        inLeafData->gids[i] = offset + (long)i;

    //print2(inLeafData, comm);

    // .3 find knn for inLeafData
	treeParams params;
	params.hypertree = 1;
	params.splitter = 1;
	params.max_points_per_node = user->rkdt_mppn;
	params.max_tree_level = user->rkdt_maxLevel;
	params.min_comm_size_per_node = user->minCommSize;
	vector<long> queryIDs_rkdt;
	knnInfoForLeaf = new vector< pair<double, long> >();
	bintree::knnTreeSearch_RandomRotation_a2a(inLeafData, user->k, user->rkdt_niters, params, 1, 0,
							                    queryIDs_rkdt, knnInfoForLeaf);
    //print(inLeafData, comm);
    print(queryIDs_rkdt, knnInfoForLeaf, comm);

    // .4 copy the leaf points back to inLeafData
    inLeafData->Copy(curr->data);
    #pragma omp parallel for
    for(int i = 0; i < inLeafData->numof_points; i++)
        inLeafData->gids[i] = offset + (long)i;
    // - now we can remove the data in the metric tree leaf node to release some memory
    delete curr->data;
    curr->data = NULL;
    // - construct leafmap
    for(int i = 0; i < inLeafData->numof_points; i++)
        leafmap.insert(make_pair<long, int>(inLeafData->gids[i], i));

    print(inLeafData, comm);

    // .5 find the exclusive nn point ids
    vector<long> all_knn_gids(inLeafData->numof_points*user->k);
    #pragma omp parallel for
    for(int i = 0; i < knnInfoForLeaf->size(); i++)
        all_knn_gids[i] = (*knnInfoForLeaf)[i].second;
    vector<long> excl_nn_gids;
    find_excl_nn_ids(all_knn_gids, excl_nn_gids);

    // .6 collect coordinates
    vector< pair<double, long> > excl_knn(excl_nn_gids.size());
    #pragma omp parallel for
    for(int i = 0; i < excl_nn_gids.size(); i++) {
        excl_knn[i].first = 0.0;
        excl_knn[i].second = excl_nn_gids[i];
    }
    exclKNNofLeaf = new fksData();
    exclKNNofLeaf->dim = dim;
    exclKNNofLeaf->numof_points = excl_nn_gids.size();
    exclKNNofLeaf->gids.resize(exclKNNofLeaf->numof_points);
    #pragma omp parallel for
    for(int i = 0; i < exclKNNofLeaf->numof_points; i++)
        exclKNNofLeaf->gids[i] = excl_nn_gids[i];
    // - construct exclNNmap
    for(int i = 0; i < exclKNNofLeaf->numof_points; i++)
        exclNNmap.insert( make_pair<long, int>(excl_nn_gids[i], i) );
    // - coordinates
    exclKNNofLeaf->X.resize(excl_nn_gids.size()*dim);
    bintree::getNeighborCoords(inLeafData->numof_points, &(inLeafData->X[0]), dim,
                               excl_nn_gids.size(), 1, excl_knn, &(exclKNNofLeaf->X[0]), comm);
}


bool fksTree::getLeafexclNN(fks_ompNode *ompLeaf, fksData *exclNN)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    int k = knnInfoForLeaf->size() / inLeafData->numof_points;
    int nlocal = ompLeaf->leaf_point_local_id.size();

    if(nlocal == 0)
    {
        cout<<"rank "<<rank<<" input node might not be a leaf, local leaf id is empty"<<endl;
        return false;
    }

    // -1. get nn for all leaf points
    vector<long> all_knn_ids(nlocal*k);
    #pragma omp parallel for
    for(int i = 0; i < nlocal; i++) {
        int idx = ompLeaf->leaf_point_local_id[i];
        for(int j = 0; j < k; j++)
            all_knn_ids[i*k+j] = (*knnInfoForLeaf)[idx*k+j].second;
    }
    omp_par::merge_sort(all_knn_ids.begin(), all_knn_ids.end());
    vector<long>::iterator it = unique(all_knn_ids.begin(), all_knn_ids.end());
    all_knn_ids.resize(it-all_knn_ids.begin());

    // -2. exclude those in leaf
    vector<long> leaf_ids(nlocal);
    #pragma omp parallel for
    for(int i = 0; i < nlocal; i++)
        leaf_ids[i] = inLeafData->gids[ ompLeaf->leaf_point_local_id[i] ];
    omp_par::merge_sort(leaf_ids.begin(), leaf_ids.end());

    vector<long> excl_ids(all_knn_ids.size()+1);
    it = set_difference(all_knn_ids.begin(), all_knn_ids.end(),
                        leaf_ids.begin(), leaf_ids.end(), excl_ids.begin());
    excl_ids.resize(it-excl_ids.begin());

    // -3. copy data
    int nexcl = excl_ids.size();
    int dim = inLeafData->dim;
    exclNN->dim = dim;
    exclNN->numof_points = excl_ids.size();
    exclNN->X.resize( nexcl*dim );
    exclNN->gids.resize( nexcl );
    exclNN->mortons.resize( nexcl );
    exclNN->charges.resize( nexcl );
    #pragma omp parallel for
    for(int i = 0; i < nexcl; i++) {
        map<long, int>::iterator it = exclNNmap.find( excl_ids[i] );
        int idx = it->second;
        exclNN->gids[i] = exclKNNofLeaf->gids[idx];
        exclNN->mortons[i] = exclKNNofLeaf->mortons[idx];
        exclNN->charges[i] = exclKNNofLeaf->charges[idx];
        memcpy( &(exclNN->X[i*dim]), &(exclKNNofLeaf->X[idx*dim]), sizeof(double)*dim );
    }
    return true;

}


bool fksTree::getLeafData(fks_ompNode *ompLeaf, fksData *leaf)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    int nlocal = ompLeaf->leaf_point_local_id.size();

    if(nlocal == 0)
    {
        cout<<"rank "<<rank<<" input node might not be a leaf, local leaf id is empty"<<endl;
        return false;
    }

    int dim = inLeafData->dim;
    leaf->dim = dim;
    leaf->numof_points = nlocal;
    leaf->X.resize( nlocal*dim );
    leaf->gids.resize( nlocal );
    leaf->mortons.resize( nlocal );
    leaf->charges.resize( nlocal );
    #pragma omp parallel for
    for(int i = 0; i < nlocal; i++) {
        int idx = ompLeaf->leaf_point_local_id[i];
        leaf->gids[i] = inLeafData->gids[idx];
        leaf->mortons[i] = inLeafData->mortons[idx];
        leaf->charges[i] = inLeafData->charges[idx];
        memcpy( &(leaf->X[i*dim]), &(inLeafData->X[idx*dim]), sizeof(double)*dim );
    }
    return true;
}


void fksTree::find_excl_nn_ids(vector<long> &all_knn_gids, vector<long> &excl_knn_gids)
{
    long all_knn_n = all_knn_gids.size();
    // .1 remove duplicates within nns
    omp_par::merge_sort(all_knn_gids.begin(), all_knn_gids.end());
    vector<long>::iterator it = unique(all_knn_gids.begin(), all_knn_gids.end());
    all_knn_gids.resize(it-all_knn_gids.begin());
    // .2 remove duplicates from leaf node
    vector<long> all_leaf_gids(inLeafData->numof_points);
    #pragma omp parallel for
    for(int i = 0; i < inLeafData->numof_points; i++)
        all_leaf_gids[i] = inLeafData->gids[i];
    omp_par::merge_sort(all_leaf_gids.begin(), all_leaf_gids.end());

    excl_knn_gids.resize(all_knn_gids.size()+1);
    it = set_difference(all_knn_gids.begin(), all_knn_gids.end(),
                        all_leaf_gids.begin(), all_leaf_gids.end(),
                        excl_knn_gids.begin());
    excl_knn_gids.resize(it-excl_knn_gids.begin());
}


// only gather median and projection direction
// currently only works for p = 2^l
// if treearr.size = 0, means there is only one root_mpi node in this tree
// each treearr[i] is dim + 1 array, proj + median
void fksTree::gatherDistributedTree(fks_mpiNode *root_mpi, int dim, vector<fksTreeInfo> &treeArr)
{
	int rank, size, worldrank, worldsize;
	MPI_Comm_size(root_mpi->comm, &worldsize);
	MPI_Comm_rank(root_mpi->comm, &worldrank);

	int depth = int( ceil( log((double)worldsize) / log(2.0) ) );

    if(_DEBUG_) {
        cout<<"\trank "<<worldrank<<": enter gatherDistributedTree() "<<endl;
    }

    int sendcnt;
	int *recvcnts = new int [worldsize];
	int numof_internal_nodes = (int)pow(2.0, (double)(depth)) - 1;
	double *sendbuf = new double [dim+1];
	double *recvbuf = new double [numof_internal_nodes*(dim+1)];
	int *displs = new int [worldsize];

    // 1. collect the whole tree on root_mpi (rank 0)
	int offset = 0;
	fks_mpiNode *curr_node = root_mpi;
	int flag = 1;
    while(flag) {
		MPI_Comm_rank(curr_node->comm, &rank);
        if(rank == 0) {
			sendcnt = dim+1;
            if(curr_node->kid != NULL) {
                memcpy(sendbuf, &(curr_node->proj[0]), sizeof(double)*dim);
                sendbuf[dim] = curr_node->median;
			}
            else {
                memset(sendbuf, 0, sizeof(double)*(dim+1));
			}
		}
        else {
			sendcnt = 0;
		}

		offset = (int)pow(2.0, (double)(curr_node->level)) - 1;
		MPI_Gather(&sendcnt, 1, MPI_INT, recvcnts, 1, MPI_INT, 0, root_mpi->comm);

		displs[0] = 0;
		for(int i = 1; i < worldsize; i++)
			displs[i] = displs[i-1] + recvcnts[i-1];
		MPI_Gatherv(sendbuf, sendcnt, MPI_DOUBLE, recvbuf+offset*(dim+1),
                        recvcnts, displs, MPI_DOUBLE, 0, root_mpi->comm);

		curr_node = curr_node->fks_kid;
		MPI_Barrier(root_mpi->comm);

		if(curr_node == NULL) flag = 0;
		else if(curr_node->level == depth) flag = 0;
		else flag = 1;

	}	// end while

    if(_DEBUG_) {
        cout<<"\trank "<<worldrank<<": gatherDistributedTree() collect from root "<<endl;
    }

	// 2. bcast the whole tree to all processes
	MPI_Bcast(recvbuf, numof_internal_nodes*(dim+1), MPI_DOUBLE, 0, root_mpi->comm);

    if(_DEBUG_) {
        cout<<"\trank "<<worldrank<<": gatherDistributedTree() bcast from root "<<endl;
    }

	// 3. decode the tree array
	treeArr.resize(numof_internal_nodes);
    for(int i = 0; i < treeArr.size(); i++) {
        treeArr[i].proj.resize(dim);
        memcpy( &(treeArr[i].proj[0]), recvbuf+i*(dim+1), sizeof(double)*dim );
        treeArr[i].median = recvbuf[i*(dim+1)+dim];
	}

    if(_DEBUG_) {
        cout<<"\trank "<<worldrank<<": gatherDistributedTree() decode tree array "<<endl;
    }

	delete [] sendbuf;
	delete [] recvbuf;
	delete [] recvcnts;
	delete [] displs;

    if(_DEBUG_) {
        cout<<"\trank "<<worldrank<<": gatherDistributedTree() exit "<<endl;
    }

}


void fksTree::plantTree(vector<fksTreeInfo> &ArrTree, int depth,
                            fks_ompNode *inParent, fks_ompNode *inNode)
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
        int dim = ArrTree[idx].proj.size();
        inNode->proj.resize(dim);
        memcpy( &(inNode->proj[0]), &(ArrTree[idx].proj[0]), sizeof(double)*dim );
        inNode->median = ArrTree[idx].median;

		// 1.2 create new kid node
		inNode->leftNode = new fks_ompNode(2*inNode->lnid+0);
		inNode->rightNode = new fks_ompNode(2*inNode->lnid+1);
		plantTree(ArrTree, depth, inNode, inNode->leftNode);
		plantTree(ArrTree, depth, inNode, inNode->rightNode);
	}
}


void fksTree::destroyTree(fks_ompNode *inNode)
{
	if(inNode != NULL) {
		destroyTree(inNode->leftNode);
		destroyTree(inNode->rightNode);
		delete inNode;
		inNode = NULL;
	}
}


void fksTree::printTree(fks_ompNode *inNode)
{
	if(NULL == inNode->leftNode && NULL == inNode->rightNode) {
        if(inNode->leaf_point_local_id.size() > 0) {
		    cout<<"level: "<<inNode->level
                <<" - node array id: "<<inNode->lnid
                <<" - node morton id: "<<inNode->node_morton
                <<" @leaf ";
            for(int i = 0; i < inNode->leaf_point_local_id.size(); i++)
                cout<<inNode->leaf_point_local_id[i]<<" ";
            cout<<endl;
        }
		return;
	}
    else {
		cout<<"level: "<<inNode->level
            <<" - node array id: "<<inNode->lnid
            <<" - node morton id: "<<inNode->node_morton
			<<" - median: "<<inNode->median<<" - proj: ";
        for(int i = 0; i < inNode->proj.size(); i++)
            cout<<inNode->proj[i]<<" ";
        cout<<endl;
		printTree(inNode->leftNode);
		printTree(inNode->rightNode);
	}
}


// count should be initialized as 0 before input this function
void fksTree::traverseTree(double *points, int numof_points, int dim, vector<int> &member_ids,
			            fks_ompNode *inNode, int depth,
			            int *point_to_visual_leaf_membership, int *count)
{
	int worldrank;
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);

	if(inNode->parent == NULL) {
		int nc = (int)pow(2.0, (double)depth);
		for(int i = 0; i < nc; i++)
			count[i] = 0;
	}

    if(depth == 0) {
        for(int i = 0; i < numof_points; i++)
            point_to_visual_leaf_membership[i] = 0;
        return;
    }

	int numof_members = member_ids.size();
	vector<int> left_membership;
	vector<int> right_membership;

	vector<int> equal_membership;
	left_membership.reserve(numof_members/2);
	right_membership.reserve(numof_members/2);
    int ONE = 1;
    for(int i = 0; i < numof_members; i++) {
        double px = ddot( &dim, &(inNode->proj[0]), &ONE, &(points[member_ids[i]*dim]), &ONE);
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

	traverseTree(points, numof_points, dim, left_membership, inNode->leftNode, depth,
			  point_to_visual_leaf_membership, count);
	traverseTree(points, numof_points, dim, right_membership, inNode->rightNode, depth,
			  point_to_visual_leaf_membership, count);
}


long fksTree::mortonID(double *point, int dim, long inmid, long offset, fks_ompNode *inNode)
{
    if(NULL == inNode->leftNode && NULL == inNode->rightNode) {     // leaf node
        return inmid;
    }
    else {
        int ONE = 1;
        double py = ddot( &dim, &(inNode->proj[0]), &ONE, point, &ONE );
        if(py < inNode->median) {   // if left child
            int mymid = ( inmid & (~(1 << (inNode->level+1+offset))) );  // set inmid bit 'level' as 0
            return mortonID(point, dim, mymid, offset, inNode->leftNode);
        }
        else {      // if right child
            int mymid = ( inmid | (1 << (inNode->level+1+offset)) ) ;  // set inmid bit 'level' as 1
            return mortonID(point, dim, mymid, offset, inNode->rightNode);
        }
    }
}


