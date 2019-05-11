#ifndef _PARTREE_H__
#define _PARTREE_H__

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
class parNode; 
//class binData;
class parTree;
typedef parNode* pparNode;
//typedef binData* pbinData;
typedef parTree* pparTree;
/* ************************************************** */


class parNode {
public:
	int lnid;				// local node id, on every level from left to right, 0, 1, 2, ..., 2^level-1
	int level;				// level of the node.  Root is level 0.
	vector<double> rw;		// workspace for fast rotation
	vector<double> proj;	// projection direction
	//double *rw;
	//int wsize;
	//double *proj;

	
	int coord_mv;			// coorid with maximum variance
	double median;			// median of projected values
	int commsize;
	int glbN;			// global number of points in this node
	pparNode parent;		// Pointer to parent
	pparNode leftNode;		// the next level
	pparNode rightNode;
	
	//-------------  Methods
	// default constructor
	//parNode() : level(0),lnid(0),median(0),rw(NULL),proj(NULL),parent(NULL),leftNode(NULL),rightNode(NULL),commsize(-1) {;}
	parNode() : level(0),lnid(0),median(0),parent(NULL),leftNode(NULL),rightNode(NULL),commsize(-1),glbN(0) {;}
    // Constructor.  Must be used for non-root nodes.
	//parNode(int id) : level(0),lnid(id),median(0),rw(NULL),proj(NULL),parent(NULL),leftNode(NULL),rightNode(NULL),commsize(-1) {;}
	parNode(int id) : level(0),lnid(id),median(0),parent(NULL),leftNode(NULL),rightNode(NULL),commsize(-1),glbN(0) {;}
	// Constructor,  Must be used for non-root nodes
	//parNode(int id, int size) : level(0),lnid(id),median(0),rw(NULL),proj(NULL),parent(NULL),leftNode(NULL),rightNode(NULL),commsize(size) {;}
	parNode(int id, int size) : level(0),lnid(id),median(0),parent(NULL),leftNode(NULL),rightNode(NULL),commsize(size),glbN(0) {;}
	parNode(int id, int size, int gN) : level(0),lnid(id),median(0),parent(NULL),leftNode(NULL),rightNode(NULL),commsize(size),glbN(gN) {;}
};


class parTree {
	public:
		pparNode root;
		vector<pbinData> leafRefArr;
		vector<int> leafArrFlag;			// if -1, it is NULL in leafRefArr
		vector<int> rcoordArr;				// store some random coordinates used for split points
		int numof_ref_points_in_tree;		// total number of ref points in this tree
		int depth;
		
		struct Options{
			string splitter;		// "rsmt" or "rkdt"
			int debug_verbose;		// print out debug info
			int timing_verbose;		// print out timming info.
			int flag_r;				// do not rotation (0), rotate only at root (1), rotate on all levels (2)
			int flag_c;				// random choose a dim (0), coord with the maximum variance (1)
			// method
			Options() : splitter("rsmt"),debug_verbose(false),timing_verbose(false),flag_r(0),flag_c(0) {;}
			void Copy(Options o) {
				splitter = o.splitter;
				debug_verbose = o.debug_verbose;
				timing_verbose = o.timing_verbose;
				flag_r = o.flag_r;
				flag_c = o.flag_c;
			}
		};
		
		Options options;  // This nodes Options object.


		parTree() : root(NULL),depth(0) {;}
		~parTree();
		
		void build(pbinData inData, int minp, int maxlev, MPI_Comm comm);
		void destroy_tree(pparNode node);
		
		void exchangeRefData(pbinData outData, MPI_Comm comm);
		int visitGreedy(double *point, int dim, pparNode node);
		void distributeQueryData(pbinData queryData, MPI_Comm comm);
		
		void insert(pparNode in_parent, pparNode inNode, pbinData inData, int minp, int maxlev);
		void parvar(double *points, int numof_points, int dim, double *mean, double *var); 
		void glb_mean( double *points, int numof_points, int glb_numof_points, int dim,
					   double *glb_mu, MPI_Comm comm);
		void maxvarProjection(double *points, int numof_points, int glb_numof_points, int dim,
							  int &mvind, double *pv, MPI_Comm comm);
		double distSelect(vector<double> &arr, int ks, MPI_Comm comm);
		void assignMembership(double *px, int numof_points, int glb_numof_points, double median,
							  int *point_to_kid_membership, int *local_numof_points_per_kid,
							  int *glb_numof_points_per_kid, MPI_Comm comm);
		void copyData(pbinData inData, vector<int>& membership, pbinData outData);
		
		
		double select(vector<double> &arr, int ks);
		static void mean(double *points, int numof_points, int dim, double *mu);
};

#endif




