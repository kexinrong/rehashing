#ifndef _OLDTREE_H_
#define _OLDTREE_H_

#include <mpi.h>
#include <cassert>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <utility>
#include <util.h>
#include <cstring>
#include <omp.h>

#include "binTree.h"

using namespace std;



/* ************************************************** */
class oldNode; 
typedef oldNode* poldNode;
/* ************************************************** */



/* ************************************************** */
/** 
 * This is the main data structure for the PCL-tree.  Each MPI process stores a doubly-linked
 * list of oldNode objects, representing a path from the root to that process's leaf node.  
 */
class oldNode {
public:
	pbinData data; 
	int level;  
	MPI_Comm comm;  
	
	poldNode parent;	 
	poldNode kid;   

    int Nglobal;  // The total number of points stored in or beneath this tree node, across all member processes.

	//vector<double> matR;		// rotation matrix on this level
	vector<double> rw;			// workspace for fast rotation
	vector<double> proj; 
	int coord_mv;				// which coord to use 
	double median;				// median of coord_mv
	
	//vector<int> cluster_to_kid_membership; 
	vector<int> rank_colors;  // Length is equal to size of comm. Stores the child that each MPI rank belongs to.
	int chid;	// This node's child id.
	
	struct Options{
		string splitter;        // splitter type: 0 mtree 1 maxVar
		int flag_r;				// do not rotate (0), rotate on root level (1) or rotate on every level (2) 
		int flag_c;				// choose coord randomly (0), or with max variance (1)
		int hypertree;			// repartition points using hypertree(1) or oldtree (0)
		int debug_verbose;		// print extensive information and/or compare results against direct search.
		int timing_verbose;		// print timing.
    	int pruning_verbose;	// Print pruning statistics in tree query operations.
		int flops_verbose;		// now it's useful, to be deleted later
		static const int max_max_treelevel=50;  // Maximum allowable value for maxlev.
		Options() : splitter("rkdt"),hypertree(1),flag_r(1),flag_c(0),flops_verbose(false),debug_verbose(false),timing_verbose(false),pruning_verbose(false) {;}
	};
	Options options;  ///< This nodes Options object.


	//-------------  Methods
	oldNode() : data(NULL),level(0),chid(0),comm(MPI_COMM_WORLD),parent(NULL),kid(NULL) {;}
	oldNode(int ci) : data(NULL),level(0),chid(ci),comm(MPI_COMM_WORLD),parent(NULL),kid(NULL) {;}
	
	~oldNode();
	//void destroy_node(poldNode inNode);

	void Insert(poldNode inParent, int maxp, int maxlev, int minCommSize, MPI_Comm comm, pbinData inData);
	void Insert_hypertree(poldNode inParent, int maxp, int maxlev, int minCommSize, MPI_Comm comm, pbinData inData);
	void Insert_oldtree(poldNode inParent, int maxp, int maxlev, int minCommSize, MPI_Comm comm, pbinData inData);
	
	void parvar(double *points, int numof_points, int dim, double *mean, double *var);
	void maxVarSplitter( double *points, int numof_points, int dim, int flag_c,
						// output
						int &coord_mv, double &medV,
						int* point_to_hyperplane_membership,
						int *local_numof_points_per_hyperplane,
						int *global_numof_points_per_hyperplane,
						MPI_Comm comm);


	void getProjection(double * points, int numof_points, int dim, double *proj, MPI_Comm comm);
	void furthestPoint(double *points, int numof_points, int dim, double* query, double* furP, MPI_Comm comm);
	void mtreeSplitter( double *points, int numof_points, int dim,
						// output
						double *proj, double &medianValue,
						int* point_to_hyperplane_membership,
						int *local_numof_points_per_hyperplane,
						int *global_numof_points_per_hyperplane,
						MPI_Comm comm);

	double distributeSelect(vector<double> &arr, int ks, MPI_Comm comm);

};


#endif




