#ifndef _FKS_OMPNODE_H__
#define _FKS_OMPNODE_H__

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

class fksData : public binData
{
public:
    vector<long> mortons;
    vector<double> charges;
    // fksData also contains the following data
    // -
    // vector<double> X;
    // vector<long> gids;
    // int dim;
    // int numof_points;
};

void print(fksData *data, MPI_Comm comm);


class fks_ompNode {
public:
	int lnid;			    // local node id, on every level from left to right, 0, 1, ..., 2^level-1
	int level;			    // level of the node.  Root is level 0.
	vector<double> proj;	// projection direction
	double median;			// median of projected values
	fks_ompNode *parent;	// Pointer to parent
	fks_ompNode *leftNode;	// the left child
	fks_ompNode *rightNode; // the right child
    vector<int> leaf_point_local_id;    // if leaf node, will store the local id of all points

    fksData *skeletons;
    long node_morton;

	// default constructor
	fks_ompNode() : level(0),lnid(0),median(0),parent(NULL),leftNode(NULL),rightNode(NULL),skeletons(NULL),node_morton(0) {;}
    // Constructor.  Must be used for non-root nodes.
	fks_ompNode(int id) : level(0),lnid(id),median(0),parent(NULL),leftNode(NULL),rightNode(NULL),skeletons(NULL),node_morton(0) {;}

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


#endif




