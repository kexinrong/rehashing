#ifndef _STTREE_H__
#define _STTREE_H__

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
class stNode; 
//class binData;
class stTree;
typedef stNode* pstNode;
//typedef binData* pbinData;
typedef stTree* pstTree;
/* ************************************************** */

//  Auxiliary data structure to hold point coords (X), their dimension (dim) and global ids.
/*struct binData {
	int dim;				// Dimensionality of points.
	vector<double> X;		// Data point coordinates.
	vector<long> gids;		// global ids of points.
	vector<long> lids;		// local ids of points
	// - Methods
	void Copy(pbinData data){
			X.resize( data->X.size() );
			gids.resize( data->gids.size() );
			lids.resize( data->lids.size() );
			dim = data->dim;
			int npoints = data->gids.size();
			#pragma omp parallel if(npoints > 1000)
			{
				#pragma omp parallel for schedule(dynamic,256)
				for(int i=0; i< npoints*data->dim; i++) X[i] = data->X[i];
				#pragma omp parallel for schedule(dynamic,256)
				for(int i=0; i<npoints; i++) gids[i] = data->gids[i];
			}
			if(data->lids.size() > 0) {
				#pragma omp parallel if(npoints > 1000)
				{
					#pragma omp parallel for schedule(dynamic,256)
					for(int i=0; i<npoints; i++) lids[i] = data->lids[i];
				}
			}
		}
};*/


class stNode {
public:
	int lnid;				// local node id, on every level from left to right, 0, 1, 2, ..., 2^level-1
	int level;				// level of the node.  Root is level 0.
	//vector<double> matR;	// rotation matrix on this level
	vector<double> rw;		// workspace for fast rotation
	vector<double> proj;	// projection direction
	int coord_mv;			// coorid with maximum variance
	double median;			// median of projected values
	pstNode parent;			// Pointer to parent
	pstNode leftNode;		// the next level
	pstNode rightNode;

	//-------------  Methods
	// default constructor
	stNode() : level(0),lnid(0),median(0),parent(NULL),leftNode(NULL),rightNode(NULL) {;}
    // Constructor.  Must be used for non-root nodes.
	stNode(int id) : level(0),lnid(id),median(0),parent(NULL),leftNode(NULL),rightNode(NULL) {;}
};


class stTree {
	public:
		pstNode root;
		vector<pbinData> leafRefArr;
		int numof_ref_points_in_tree;		// total number of ref points in this tree
		int depth;

		struct Options{
			string splitter;		// "rsmt" or "rkdt"
			int debug_verbose;		// print out debug info
			int timing_verbose;		// print out timming info.
			int flag_r;				// do not rotation (0), rotate only at root (1), rotate on all levels (2)
			// method
			Options() : splitter("rsmt"),debug_verbose(false),timing_verbose(false),flag_r(0) {;}
			void Copy(Options o) {
				splitter = o.splitter;
				debug_verbose = o.debug_verbose;
				timing_verbose = o.timing_verbose;
				flag_r = o.flag_r;
			}
		};

		Options options;  // This nodes Options object.


		stTree() : root(NULL),depth(0) {;}
		~stTree();
		
		void build(pbinData inData, int minp, int maxlev);
		void recoverData(pbinData outData);

		void queryGreedy(pbinData queryData, int k, vector< pair<double, long> > &results);
		void queryGreedy_a2a(int k, vector< pair<double, long> > &results);
		void querySampling(pbinData queryData, int k, vector< pair<double, long> > &results);
	
		void destroy_tree(pstNode node);
		void insert(pstNode in_parent, pstNode inNode, pbinData inData, int minp, int maxlev);
		
		static void mean(double *points, int numof_points, int dim, double *mu);
		void parvar(double *points, int numof_points, int dim, double *mean, double *var); 
		void maxvarProjection(double *points, int numof_points, int dim, int &mvind, double *pv);
	
		void furthestPoint(double *points, int numof_points, int dim, double *query, double *furP);
		void mtreeProjection(double * points, int numof_points, int dim, double * proj, double *pv);

		//void assignMembership(const vector<double>& px, 
		//					 double &median, vector<int>& leftkid_membership, vector<int>& rightkid_membership);

		double select(vector<double> &arr, int ks);
		
		void assignMembership(const vector<double>& px, double median, 
							  vector<int> &leftkid_membership, vector<int>& rightkid_membership);
		
		void copyData(pbinData inData, vector<int>& membership, pbinData outData);
		
		int visitGreedy(double *point, int dim, pstNode node);
		void randperm(int m, int N, vector<int>& arr);
		void sampleNode(pstNode node, vector<double> &samples);
		void visitSampling(pbinData data, pstNode node, int *membership);
};

#endif




