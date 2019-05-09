#ifndef _SRKDT_H__
#define _SRKDT_H__

#include <cassert>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <utility>
#include <util.h>

#include "binTree.h"

using namespace std;


/* ************************************************** */
class snode;
class srkdt;
typedef snode* psnode;
typedef srkdt* psrkdt;
/* ************************************************** */


class snode {
public:
	int lnid;				// local node id, on every level from left to right, 0, 1, 2, ..., 2^level-1
	int level;				// level of the node.  Root is level 0.
	vector<double> proj;	// projection direction
	double median;			// median of projected values
	psnode parent;			// Pointer to parent
	psnode leftNode;		// the next level
	psnode rightNode;

	//-------------  Methods
	// default constructor
	snode() : level(0),lnid(0),median(0),parent(NULL),leftNode(NULL),rightNode(NULL) {;}
    // Constructor.  Must be used for non-root nodes.
	snode(int id) : level(0),lnid(id),median(0),parent(NULL),leftNode(NULL),rightNode(NULL) {;}
};

class srkdt {
    private:
        binData *ptrData;   // only point to the input data pointer

	public:
		psnode root;
        vector<double> rw;
		vector< vector<long> * > leafRefIDArr;
		int numof_ref_points_in_tree;		// total number of ref points in this tree
		int depth;

		srkdt() : root(NULL),depth(0) {;}
		~srkdt();

		void build(pbinData inData, int minp, int maxlev);
        void queryGreedy(pbinData queryData, int k, vector< pair<double, long> > &results);
		//void queryGreedy_a2a(int k, vector< pair<double, long> > &results);
        int getNumofLeaves() const { return leafRefIDArr.size(); }
        int getDepth() const { return depth; }
        int getNumofPointsInTree() const { return numof_ref_points_in_tree; }

        void queryGreedyandMerge(pbinData queryData, int k, vector< pair<double, long> > &results);


        void queryGreedy_a2a(int k, vector< pair<double, long> > &results);
        void queryGreedyandMerge_a2a(int k, vector< pair<double, long> > &results);


        void merge_one_knn(binData *refData, vector< pair<double, long> > &A, int id, int k,
                                pair<double, long> *B, pair<double, long> *tmp);

        void merge_one(pair<double, long> *A, pair<double, long> *B, pair<double, long> *tmp, int k);



    private:
	    void destroy_tree(psnode node);
        void insert(psnode in_parent, psnode inNode, pbinData inData, int minp, int maxlev, vector<long> *&pids);

        double select(vector<double> &arr, int ks);
		void assignMembership(const vector<double>& px, double median,
							  vector<int> &leftkid_membership, vector<int>& rightkid_membership);
		void copyData(pbinData inData, vector<long>& membership, pbinData outData);
		int visitGreedy(double *point, int dim, psnode node);
};

#endif




