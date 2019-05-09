#ifndef _FKS_OMPNODE_H__
#define _FKS_OMPNODE_H__

#include <mpi.h>

#include <mkl_lapacke.h>
#include <cassert>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <utility>
#include <util.h>

#include "binTree.h"
#include "direct_knn.h"

using namespace std;

namespace askit {

class fksData : public binData
{
public:
  
  // initialize the failed flag to always be false
  fksData() : cant_prune(false) {}
  
    vector<long> mortons;
    vector<double> charges;
    // fksData also contains the following data
    // -
    // vector<double> X;
    // vector<long> gids;
    // int dim;
    // int numof_points;

    // The matrix computed by the ID and used to obtain effective charges
    vector<double> proj;
    // The permutation needed to apply proj
    vector<lapack_int> skeleton_perm;

    // Added these for other kernel summation algorithms

    // The centroid of the bounding ball
    vector<double> centroid;
    // The radius of the bounding ball
    double radius;
    // The sum of the charges of all the points owned by this node
    double node_charge;

    // For MC sampling, stores the local ids of all points owned by this node
    std::vector<int> local_ids;

    // Entries in the series coefficients
    // These are constructed in the same order as the multi-indices to handle
    // the sparse grid
    std::vector<double> coefficients;
    
    // Flag for adaptive level restiction 
    bool cant_prune;

};

class fks_ompNode {
public:
	int lnid;			    // local node id, on every level from left to right, 0, 1, ..., 2^level-1
	int level;			    // level of the node.  Root is level 0.
	vector<double> proj;	// projection direction
	double median;			// median of projected values
	fks_ompNode *parent;	// Pointer to parent
	fks_ompNode *leftNode;	// the left child
	fks_ompNode *rightNode; // the right child
    vector<int> leaf_point_local_id;    // if leaf, local id. *deprecated*, now only the shared memoery tree has valid leaf_point_local_id, the other nodes in let do not use it anymore. I will use leaf_point_gids
    vector<long> leaf_point_gids;     // if leaf, global ids of points

    // the total number of points owned by all descendants of this node
    // IMPORTANT: this isn't currently passed around in the LET, so don't use 
    // it after the upward pass
    long num_points_owned;

    fksData *skeletons;
    long node_morton;

    // for LET tree
    bool isLocalRoot;
    bool isDistributedNode;
    long global_node_id;     // from root (0), in an order of level order traversal, left to right
    int myrank;             // if it is subtree, which rank it resides
    bool skeleton_request;         // 0: do not need, 1: need
    bool leaf_request;

	// default constructor
    // Changed default value of myrank to 0 for correctness in single MPI rank case -- Bill
	fks_ompNode() : level(0),lnid(0),median(0),parent(NULL),leftNode(NULL),rightNode(NULL),skeletons(NULL),node_morton(0), isDistributedNode(true), isLocalRoot(false), global_node_id(0), myrank(0), skeleton_request(0), leaf_request(0) {;}
    // Constructor.  Must be used for non-root nodes.
	fks_ompNode(int id) : level(0),lnid(id),median(0),parent(NULL),leftNode(NULL),rightNode(NULL),skeletons(NULL),node_morton(0), isDistributedNode(true), isLocalRoot(false),global_node_id(0), myrank(0), skeleton_request(0), leaf_request(0) {;}

    ~fks_ompNode();

    void insert(fks_ompNode *in_parent, fksData *inData, vector<long> *&active_set, int mppn, int maxlevel);

    void insert(fks_ompNode *in_parent, fksData *inData, vector<long> *&active_set, long morton_offset, int mppn, int maxlevel);

    void getLeafData(fksData *inData, fksData *outData);

private:
    void mtreeProjection(double *points, int numof_points, int dim,
                                double *proj, double *pv);
    void furthestPoint(double *points, int numof_points, int dim,
                                        double *query, double *furP);
    void mean(double *points, int numof_points, int dim, double *mu);
    void assignMembership(const vector<double>& px, double median,
	                      vector<int>& leftkid_membership,
                          vector<int>& rightkid_membership);
    double select(vector<double> &arr, int ks);
    double getMedian(vector<double> &arr);

    long setBitZero(long input, long bitpos);
    long setBitOne(long input, long bitpos);

};

} // namespace

#endif




