#ifndef _BINARYTREE_H_
#define _BINARYTREE_H_

#include <mpi.h>
#include <cassert>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <utility>
#include <cstring>
#include <omp.h>
#include "util.h"

using namespace std;

/* ************************************************** */
class binNode;
class binData;
typedef binNode* pbinNode;
typedef binData* pbinData;
/* ************************************************** */

struct treeParams{
	int splitter;  
	int hypertree;
	int debug_verbose; 
	int timing_verbose;
    int pruning_verbose; 
	int max_points_per_node;
	int max_tree_level;
	int min_comm_size_per_node;
	int flops_verbose;
	int eval_verbose;
	int traverse_type;

	treeParams():splitter(0),hypertree(1),debug_verbose(0),timing_verbose(0),pruning_verbose(0),max_points_per_node(1000),max_tree_level(20),min_comm_size_per_node(1),flops_verbose(0),eval_verbose(0),traverse_type(0){}

};


/**
  Auxiliary data structure to hold point coords (X), their dimension (dim) and global ids.
 */
class binData {
public:
	vector<double> X;    ///< Data point coordinates.
	int dim;             ///< Dimensionality of points.
	int numof_points;
	vector<long> gids;   ///< global ids of points.
	vector<long> lids;	/// < local ids of points, used by shared memory tree
	vector<double> radii; ///< Search radii of points (only meaningful if this binData object is a query point set).

    binData() : dim(0),numof_points(0) {;}

    //-------------  Methods
    virtual void Copy(pbinData data){
		X.resize( data->numof_points * data->dim );
		gids.resize( data->numof_points );
		dim = data->dim;
		numof_points = data->numof_points;
		int npoints = numof_points;
		#pragma omp parallel if(npoints > 2000)
		{
			int omp_num_points, last_omp_num_points;
			int t = omp_get_thread_num();
			int numt = omp_get_num_threads();
			omp_num_points = npoints / numt;
			last_omp_num_points = npoints - (omp_num_points * (numt-1));
			//This thread's number of points
			int threadpoints = (t == numt-1) ? last_omp_num_points : omp_num_points;
			memcpy( (void*)&(X[t*omp_num_points*dim]), (void*)&(data->X[t*omp_num_points*dim]),
                                 threadpoints*dim*sizeof(double) );
			memcpy( (void*)&(gids[t*omp_num_points]), (void*)&(data->gids[t*omp_num_points]),
                                 threadpoints*sizeof(long) );

		}
		if(data->radii.size()>0){ 
			radii.resize(data->numof_points); 
			#pragma omp parallel if(npoints > 2000)
			{
				#pragma omp for //schedule(dynamic,256)
				for(int i=0; i<npoints; i++) radii[i] = data->radii[i];
			}
		}
		if(data->lids.size()>0){ 
			lids.resize(data->numof_points); 
			#pragma omp parallel if(npoints > 2000)
			{
				#pragma omp for //schedule(dynamic,256)
				for(int i=0; i<npoints; i++) lids[i] = data->lids[i];
			}
		}
	}
};


/* ************************************************** */
/**
 * This is the main data structure for the PCL-tree.  Each MPI process stores a doubly-linked
 * list of binNode objects, representing a path from the root to that process's leaf node.
 */
class binNode {
public:
	pbinData data;
	int level;
	MPI_Comm comm;

	pbinNode parent;
	pbinNode kid;

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
	binNode() : data(NULL),level(0),chid(0),comm(MPI_COMM_WORLD),parent(NULL),kid(NULL) {;}
	binNode(int ci) : data(NULL),level(0),chid(ci),comm(MPI_COMM_WORLD),parent(NULL),kid(NULL) {;}

	virtual ~binNode();
	//void destroy_node(pbinNode inNode);

	void Insert(pbinNode inParent, int maxp, int maxlev, int minCommSize, MPI_Comm comm, pbinData inData);
	//void Insert_hypertree(pbinNode inParent, int maxp, int maxlev, int minCommSize, MPI_Comm comm, pbinData inData);
	//void Insert_oldtree(pbinNode inParent, int maxp, int maxlev, int minCommSize, MPI_Comm comm, pbinData inData);
    void InsertInMemory(pbinNode in_parent, int maxp, int maxLevel, int minCommSize, MPI_Comm inComm, pbinData inData, binData *datapool, vector<int> &gid2lid);

	void parvar(double *points, int numof_points, int dim, double *mean, double *var);
	void maxVarSplitter( double *points, int numof_points, int dim, int flag_c,
			// output
			int &coord_mv, double &medV,
			int* point_to_hyperplane_membership,
			int *local_numof_points_per_hyperplane,
			int *global_numof_points_per_hyperplane,
			MPI_Comm comm);


    void medianSplitter(// input
			const vector<double> &px,
			// output
			double &medV,
			int* point_to_hyperplane_membership,
			int* local_numof_points_per_hyperplane,
			int* global_numof_points_per_hyperplane,
			MPI_Comm comm );

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




