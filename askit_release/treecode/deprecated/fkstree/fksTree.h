#ifndef __FKSTREE_H__
#define __FKSTREE_H__

#include <cstdlib>
#include <vector>
#include <mpi.h>

#include "fks_ompNode.h"
#include "binTree.h"


typedef struct {
    int k;                  // 'k' nearest neighbors
    int rkdt_mppn;          // rkdt: (to find the knn) maximum points per leaf node
    int rkdt_maxLevel;      // rkdt: (to find the knn) maximum level of the tree
    int rkdt_niters;        // rkdt: (to find the knn) iterations used to find knn in rkdt
    int fks_mppn;           // fks_tree: (to do the free kernel summation) maximum points per leaf node
    int fks_maxLevel;       // fks_tree: (to do the free kernel summation) maximum level of the tree
    int minCommSize;        // minimum communicator size for both tree, usually set to be 1
} fksCtx;


typedef struct {
    vector<double> proj;
    double median;
} fksTreeInfo;


class fks_mpiNode : public binNode
{
public:
    // - it also has the following member
    // vector<double> proj;
    // double median;
    // binNode *kid;    // not used in fks_mpiNode
    // binNode *parent;     // not used in fks_mpiNode
    // int level;
    // binData *data;   // points in leaf node
    // MPI_Comm comm
    //

    fks_mpiNode *fks_kid;
    fks_mpiNode *fks_parent;
    fksData *skeletons;
    long node_morton;

    fks_mpiNode() : fks_kid(NULL), fks_parent(NULL), skeletons(NULL), node_morton(0) {;}
    fks_mpiNode(int ci): binNode(ci), fks_kid(NULL), fks_parent(NULL), skeletons(NULL), node_morton(0) {;}
    ~fks_mpiNode();

    void Insert(fks_mpiNode *in_parent, int maxp, int maxLevel, int minCommSize, MPI_Comm inComm, binData *inData);


};


class fksTree {
public:
    fks_mpiNode *root_mpi;
    fks_ompNode *root_omp;
    fksData *inLeafData;
    fksData *exclKNNofLeaf;
    vector< pair<double, long> > *knnInfoForLeaf;
    MPI_Comm comm;

    fksTree() : root_mpi(NULL), inLeafData(NULL), exclKNNofLeaf(NULL), knnInfoForLeaf(NULL), comm(MPI_COMM_WORLD) { }
    ~fksTree();

    void build(fksData *inData, void *ctx);
    bool getLeafData(fks_ompNode *ompLeaf, fksData *leaf);
    bool getLeafexclNN(fks_ompNode *ompLeaf, fksData *exclNN);

    // made this public for getting local ids from global ids in treecode driver -- Bill
    map<long, int> leafmap;


private:

    // better use hashmap
    map<long, int> exclNNmap;

    void find_knn_for_leaves(void *ctx);
    void find_excl_nn_ids(vector<long> &all_knn_gids, vector<long> &excl_knn_gids);

    // ----- functions used to collect the distributed tree
    void gatherDistributedTree(fks_mpiNode *root_mpi, int dim, vector<fksTreeInfo> &treeArr);

    void plantTree(vector<fksTreeInfo> &ArrTree, int depth,
                            fks_ompNode *inParent, fks_ompNode *inNode);

    void destroyTree(fks_ompNode *inNode);

    void printTree(fks_ompNode *inNode);

    void traverseTree(double *points, int numof_points, int dim, vector<int> &member_ids,
			            fks_ompNode *inNode, int depth,
			            int *point_to_visual_leaf_membership, int *count);

    long mortonID(double *point, int dim, long inmid, long offset, fks_ompNode *inNode);



};


void print(fksData *data, MPI_Comm comm);

void print(binData *data, MPI_Comm comm);

void print(vector<long> &queryIDs, vector< pair<double, long> > *knns);


#endif
