#include <mpi.h>

#include <vector>
#include <cassert>
#include <cmath>
#include <utility>
#include <CmdLine.h>
#include <ompUtils.h>
#include <ctime>
#include <omp.h>
#include <float.h>
#include <queue>

#include "fks_ompNode.h"
#include "generator.h"
#include "fksTree.h"
// #include "fileio.h"
#include "parallelIO.h"

using namespace Torch;
using namespace std;
using namespace askit;

void dummySkeleton(fks_ompNode *root, int dim)
{
    // level order traversal of the tree
    fks_ompNode *curr = root;
    queue<fks_ompNode *> myqueue;
    myqueue.push(curr);
    while( !myqueue.empty() ) {
        // dequeue the front node
        curr = myqueue.front();
        myqueue.pop();
        if(curr->skeletons == NULL)
            curr->skeletons = new fksData();
        int ns = curr->global_node_id;
        curr->skeletons->numof_points = ns;
        curr->skeletons->dim = dim;
        curr->skeletons->charges.resize(ns);
        curr->skeletons->X.resize(ns*dim);
        for(int i = 0; i < ns; i++) {
            curr->skeletons->charges[i] = (double)rand()/RAND_MAX;
            for(int j = 0; j < dim; j++) {
                curr->skeletons->X[i*dim+j] = (double)rand()/RAND_MAX;
            }
        }

        // enqueue left child
        if(curr->leftNode != NULL)
            myqueue.push(curr->leftNode);

        // enqueue right child
        if(curr->rightNode != NULL)
            myqueue.push(curr->rightNode);
    }

    delete root->skeletons;
    root->skeletons = NULL;

}


int main(int argc, char **argv) {

	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

    CmdLine cmd;
	const char *phelp = "Help";
    cmd.addInfo(phelp);

    char *ptrInputFile = NULL;
    cmd.addSCmdOption("-data", &ptrInputFile, "./data.bin", "input file storing points");

    char *ptrKNNFile = NULL;
    cmd.addSCmdOption("-knn", &ptrKNNFile, "./knn.bin", "input file storing knn results");

    bool isBinary;
    cmd.addBCmdOption("-binary", &isBinary, false, "use binary file");

    int numof_ref_points;
	cmd.addICmdOption("-rn", &numof_ref_points, 1000, "number of points per process (1000).");
	int dim;
	cmd.addICmdOption("-d", &dim, 4, "Dimension of the space (4).");

    int k;
	cmd.addICmdOption("-k", &k, 2, "Number of nearest neighbors to find (2).");
	int min_comm_size_per_node;
	cmd.addICmdOption("-mcsptn", &min_comm_size_per_node, 1, "min comm size per tree node (1)");
    int fks_maxLevel;
	cmd.addICmdOption("-fks_mtl", &fks_maxLevel, 10, "maximum kernel summation tree depth [fks] (default = 10)");
	int fks_mppn;
	cmd.addICmdOption("-fks_mppn", &fks_mppn, 1000, "maximum number of points per kernel summation tree node [fks] (1000)");

	int max_excl_knn;
	cmd.addICmdOption("-maxexcl", &max_excl_knn, 2, "max exclusive nn of this node to save (*2*)");

    bool debug;
    cmd.addBCmdOption("-debug", &debug, false, "debug output");


    cmd.read(argc, argv);

    srand((unsigned)time(NULL)*rank);

    long dummy_numof_ref_points = numof_ref_points;
    long glb_numof_ref_points;
    MPI_Allreduce(&dummy_numof_ref_points, &glb_numof_ref_points, 1, MPI_LONG, MPI_SUM, comm);

    // .1 read data
    fksData *refData = new fksData();
    if(isBinary) {
        knn::mpi_binread(ptrInputFile, glb_numof_ref_points, dim, numof_ref_points, refData->X, comm);
    }
    else {
        knn::mpi_dlmread(ptrInputFile, glb_numof_ref_points, dim, refData->X, comm, false);
        numof_ref_points = glb_numof_ref_points;
    }
    refData->dim = dim;
    refData->numof_points = numof_ref_points;
    if(rank == 0) {
        cout<<"read from "<<ptrInputFile
            <<", numof_ref_points = "<<numof_ref_points<<", dim = "<<dim
            <<endl;
    }

    MPI_Allreduce(&dummy_numof_ref_points, &glb_numof_ref_points, 1, MPI_LONG, MPI_SUM, comm);

    long refid_offset;
	MPI_Scan( &dummy_numof_ref_points, &refid_offset, 1, MPI_LONG, MPI_SUM, comm );
	refid_offset -= dummy_numof_ref_points;
    refData->gids.resize(numof_ref_points);
    refData->charges.resize(numof_ref_points);
    #pragma omp parallel for
    for(int i = 0; i < numof_ref_points; i++) {
        refData->gids[i] = refid_offset + (long)i;
        refData->charges[i] = (double)rand()/(double)RAND_MAX;
    }

    if(debug || true) {
        if(rank == 0) cout<<"read data done: "<<endl;
        MPI_Barrier(comm);
        print_data(refData, comm);
    }

    // .2 read knn
    vector< pair<double, long> > *kNN_rkdt = new vector< pair<double, long> >;
    if(isBinary) {
        knn::binread_knn(ptrKNNFile, refData->gids, k, kNN_rkdt);
    }
    else {
        knn::dlmread_knn(ptrKNNFile, refData->gids, k, kNN_rkdt);
    }

    if(debug) {
        if(rank == 0) cout<<"read knn done: "<<endl;
        MPI_Barrier(comm);
        print_knn(refData->gids, kNN_rkdt, comm);
    }

    // .2 build tree
    fksCtx *ctx = new fksCtx();
    ctx->k = k;
    ctx->minCommSize = min_comm_size_per_node;
    ctx->fks_mppn = fks_mppn;
    ctx->fks_maxLevel = fks_maxLevel;
    ctx->check_knn_accuracy = false;

    fksTree *tree = new fksTree;
    tree->build(refData, ctx);

    MPI_Barrier(comm);
    if(rank == 0) cout<<"build done"<<endl;

    if(debug || true) {
        if(rank == 0) cout<<"after build: inProcData:"<<endl;
        MPI_Barrier(comm);
        askit::print_data(tree->inProcData, tree->inProcMap, comm);

        if(rank == 0) cout<<"after build: omp tree:"<<endl;
        MPI_Barrier(comm);
        askit::print_tree(tree->root_omp, comm);
    }

    // test shuffle_back func
    int divd = glb_numof_ref_points / size;
    int rem = glb_numof_ref_points % size;
    int shuffled_numof_points = rank < rem ? (divd+1) : divd;
    cout<<"rank "<<rank<<": glb_numof_ref_points = "<<glb_numof_ref_points
        <<", divd = "<<divd
        <<", shuffled_numof_points ="<<shuffled_numof_points
        <<endl;
    double *shuffle_charges = new double [shuffled_numof_points];
    tree->shuffle_back(tree->inProcData->numof_points, &(tree->inProcData->charges[0]), &(tree->inProcData->gids[0]), shuffled_numof_points, shuffle_charges, tree->comm);

    MPI_Barrier(comm);
    if(rank == 0) cout<<"shuffle_back done"<<endl;

    if(debug || true) {
        if(rank == 0) cout<<"shuffled back charges:"<<endl;
        MPI_Barrier(comm);
        for(int r = 0; r < size; r++) {
            if(r == rank) {
                cout<<"rank "<<rank<<": ";
                for(int i = 0; i < shuffled_numof_points; i++)
                    cout<<shuffle_charges[i]<<" ";
                cout<<endl;
                cout.flush();
            }
            MPI_Barrier(comm);
        }
        cout<<endl;
    }
    delete [] shuffle_charges;


    tree->knn(refData, kNN_rkdt);

    MPI_Barrier(comm);
    if(rank == 0) cout<<"knn done"<<endl;

    if(debug) {
        if(rank == 0) cout<<"after exchange knn: inProcKNN"<<endl;
        vector<long> tmp_gids(tree->numof_points_of_dist_leaf);
        memcpy(tmp_gids.data(), tree->inProcData->gids.data(), sizeof(long)*tree->numof_points_of_dist_leaf);
        MPI_Barrier(comm);
        askit::print_knn(tmp_gids, tree->inProcKNN, comm);

        if(rank == 0) cout<<"after exchange knn: inProcData: "<<endl;
        MPI_Barrier(comm);
        askit::print_data(tree->inProcData, tree->inProcMap, comm);
    }


    set< askit::triple<long, long, int> > set_leaves;
    set< askit::triple<long, long, int> > set_skeletons;
    tree->LET(1, set_leaves, set_skeletons);

    MPI_Barrier(comm);
    if(rank == 0) cout<<"let done"<<endl;

    if(debug) {
        if(rank == 0) cout<<"let node: "<<endl;
        MPI_Barrier(comm);
        askit::print_tree(tree->root_let, comm);
    }


    tree->exchange_let(set_leaves, set_skeletons);

    MPI_Barrier(comm);
    if(rank == 0) cout<<"exchange let done"<<endl;

    if(debug) {
        if(rank == 0) cout<<"after exchange let data, let: "<<endl;
        MPI_Barrier(comm);
        askit::print_tree(tree->root_let, comm);

        if(rank == 0) cout<<"after exchange let data, inProcData: "<<endl;
        MPI_Barrier(comm);
        askit::print_data(tree->inProcData, tree->inProcMap, comm);
    }











    /*
    if(size > 1) {

        tree->LET(ctx);

        for(int r = 0; r < size; r++) {
            if(r == rank) {
                cout<<"rank "<<rank<<" root_let: "<<endl;
                tree->printLET_preorder(tree->root_let);
                cout<<endl;
            }
            MPI_Barrier(comm);
        }


        cout<<"rank "<<rank<<" let done "<<endl;

        if(debug) {
            MPI_Barrier(comm);
            cout<<"(rank "<<rank<<") ===== build LET done ===== "<<endl;
        }

        int dtree_depth = (int)log2(size) + 1;
        vector< pair<int, int> > skeleton_arr;
        vector< pair<int, int> > leaf_arr;
        tree->initial_LET_Comm(0, tree->inLeafData->numof_points-1, dtree_depth, skeleton_arr, leaf_arr);

        cout<<"rank "<<rank<<" initial let comm done "<<endl;

        if(debug) {
            MPI_Barrier(comm);
            cout<<"(rank "<<rank<<") ===== intital LET comm done ===== "<<endl;
            for(int r = 0; r < size; r++) {
                if(r == rank) {
                    cout<<"(rank "<<rank<<") skeleton_arr: ";
                    for(int i = 0; i < skeleton_arr.size(); i++)
                        cout<<"("<<skeleton_arr[i].second<<", r"<<skeleton_arr[i].first<<")  ";
                    cout<<endl;
                    cout<<"(rank "<<rank<<") leaf_arr: ";
                    for(int i = 0; i < leaf_arr.size(); i++)
                        cout<<"("<<leaf_arr[i].second<<", r"<<leaf_arr[i].first<<")  ";
                    cout<<endl;
                }
                cout.flush();
                MPI_Barrier(comm);
            }
        }

        // inorder to debug skeleton, padding some dummy skeleton data
        dummySkeleton(tree->root_omp, dim);

        if(debug) {
            MPI_Barrier(comm);
            for(int r = 0; r < size; r++) {
                if(r == rank) {
                    cout<<"(rank "<<rank<<") ===== intital LET comm done ===== "<<endl;
                    tree->printLET_preorder(tree->root_let);
                    cout<<endl;
                }
                cout.flush();
                MPI_Barrier(comm);
            }
        }

        tree->finish_LET_Comm(skeleton_arr, leaf_arr);

        cout<<"rank "<<rank<<" finish let comm done "<<endl;

        if(debug) {
            MPI_Barrier(comm);
            for(int r = 0; r < size; r++) {
                if(r == rank) {
                    cout<<"(rank "<<rank<<") ===== finish LET comm done ===== "<<endl;
                    tree->printLET_preorder(tree->root_let);
                    cout<<endl;
                }
                cout.flush();
                MPI_Barrier(comm);
            }
        }

    }
    */

    // .3 check getLeafData and getLeafexclNN
    /*
    for(int r = 0; r < size; r++) {
        if(r == rank) {
            cout<<"rank "<<rank<<" root_omp: "<<endl;
            //tree->printLET_preorder(tree->root_omp);
            tree->printTree(tree->root_omp);
            cout<<endl;
        }
        MPI_Barrier(comm);
    }*/

    /*
    fks_ompNode *curr = tree->root_omp;
    while(curr->rightNode != NULL)
        curr = curr->rightNode;
    fksData *leafData = new fksData();
    tree->getLeafData(curr, leafData);
    cout<<"rank "<<rank<<" print leaf data: "<<endl;
    //print(leafData, comm);
    delete leafData;

    // getLeafexclNN()
    fksData *exclKNN = new fksData();
    tree->getLeafexclNN(curr, exclKNN);
    cout<<"rank "<<rank<<" print excl knn for leaf data: "<<endl;
    //print(exclKNN, comm);
    delete exclKNN;
    */

    /*
    MPI_Barrier(comm);
    if(rank == 0) cout<<endl<<"exclKNNofLeaf: "<<endl;
    print(tree->exclKNNofLeaf, comm);

    fks_mpiNode *curr_mpi = tree->root_mpi;
    while(curr_mpi->fks_kid != NULL) {
        curr_mpi = curr_mpi->fks_kid;
    }
    //curr_mpi = curr_mpi->fks_parent;
    //tree->mergeNNList(curr_mpi);


    while(curr_mpi != NULL) {
        tree->mergeNNList(curr_mpi, max_excl_knn);
        curr_mpi = curr_mpi->fks_parent;
    }

    curr_mpi = tree->root_mpi;
    // go to leaf
    while(curr_mpi->fks_kid != NULL)
        curr_mpi = curr_mpi->fks_kid;

    // sample leaf
    fksData *samples = new fksData();
    tree->uniformSampleSibling(curr_mpi->fks_parent, 1,  samples);
    print(samples, comm);

    delete samples;
    */

    cout<<"rank "<<rank<<" done!"<<endl;

    delete tree;

    delete refData;

	MPI_Finalize();
	return 0;
}



