#ifndef _METRICTREE_H_
#define _METRICTREE_H_

#include <mpi.h>
#include <cassert>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <utility>
#include <util.h>

using namespace std;

/* ************************************************** */
class MetricNode; 
class MetricData;
typedef MetricNode* pMetricNode;
typedef MetricData* pMetricData;
/* ************************************************** */

/** 
  Auxiliary data structure to hold point coords (X), their dimension (dim) and global ids.
 */
struct MetricData {
		vector<double> X;    ///< Data point coordinates.
		int dim;             ///< Dimensionality of points.
		vector<long> gids;   ///< global ids of points.
		vector<double> radii; ///< Search radii of points (only meaningful if this MetricData object is a query point set).
	//-------------  Methods
	void Copy(pMetricData data){
			X.resize( data->X.size() );
			gids.resize( data->gids.size() );
			dim = data->dim;
			if(data->radii.size()>0){ 
				radii.resize(data->radii.size()); 
				for(int i=0; i<data->radii.size(); i++) radii[i] = data->radii[i];
			}
			for(int i=0; i<data->X.size(); i++) X[i] = data->X[i];
			for(int i=0; i<data->gids.size(); i++) gids[i] = data->gids[i];
	}	
};


/* ************************************************** */
/** 
 * This is the main data structure for the PCL-tree.  Each MPI process stores a doubly-linked
 * list of MetricNode objects, representing a path from the root to that process's leaf node.  
 */
class MetricNode {
public:
	pMetricData data; ///< struct that stores points and global ids.
	int level;    ///< level of the node.  Root is level 0.
	MPI_Comm comm;  ///< communicator of the node.
	
	pMetricNode parent;	  ///< Pointer to parent
	pMetricNode kid;     ///< the next level

    int Nglobal;  ///< The total number of points stored in or beneath this tree node, across all member processes.

	vector<double> proj; ///< cluster centers for kids (or centroid of locally stored points if this is a leaf).
	double median; ///< cluster radii for every kid (or of locally stored points if this is a leaf).
	vector<int> cluster_to_kid_membership; ///< Element i stores the child id (0 or 1 for binary tree) that
					       ///< cluster i (with center C[i*dim]) is assigned to.

	vector<int> rank_colors;  ///< Length is equal to size of comm. Stores the child that each MPI rank belongs to.
	int chid;	///< This node's child id.


	//-------------  Methods
	/// Default constructor, comm is initialized to MPI_COMM_WORLD.
	MetricNode() : data(NULL),level(0),chid(0),comm(MPI_COMM_WORLD),parent(NULL),kid(NULL) {;}
        /**
         * Constructor.  Must be used for non-root nodes.
         * \param ci The id of the child (0 or 1 for binary tree nodes).
         */
	MetricNode(int ci) : data(NULL),level(0),chid(ci),comm(MPI_COMM_WORLD),parent(NULL),kid(NULL) {;}
	
	/**
	 * Inserts points into a tree node, recursively splitting as necessary.
	 * The algorithm works as follows:
	 * 1. We create clusters of points using kmeans
	 * 2. We assign each cluster to a child.
	 * 3. We store the cluster information for each child so that we can perform efficient queries
	 * 4. We allocate processes to childen and create the corresponding communicators.
	 * 5. All processors exchange coordinates and point IDs.
	 * 6. Recurse on our locally stored child node.
	 *
	 * \param inParent The parent of the node (NULL if this is the root).
	 * \param maxp Max points per node; if size of input is less, we don't split.
	 * \param maxlev Maximum allowable level.
	 * \param comm Communicator for the node.
	 * \param pMetricData Holds ids and point coordinates: IT IS MODIFIED.
	 */
	void Insert(pMetricNode inParent, int maxp, int maxlev, int minCommSize, MPI_Comm comm, pMetricData inData, int seedType);

 public:
	/**
	 * Options for MetricNode tree class.
	 */
	struct Options{
		int debug_verbose;   ///< Print extensive information and/or compare results against direct search.
		int timing_verbose;  ///< Currently unused.
    	int pruning_verbose; ///< Print pruning statistics in tree query operations.
		static const int max_max_treelevel=24;  ///< Maximum allowable value for maxlev.
	Options() : debug_verbose(false),timing_verbose(false),pruning_verbose(false) {;}
	};
	Options options;  ///< This nodes Options object.
};

void getProjection(double * points, 
		           int numof_points,
				   int dim,
				   double *proj,
				   MPI_Comm comm);

void furthestPoint(double *points,
				   int numof_points,
				   int dim,
				   double* query,
				   double* furP,
				   MPI_Comm comm);

void medianPartition(double *points, 
					 int numof_points,
					 int dim,
					 double *projDirection,
					 int* point_to_hyperplane_membership,
					 double &medianValue,
					 int *global_numof_points_per_hyperplane,
					 int *local_numof_points_per_hyperplane,
					 MPI_Comm comm);

double distributeSelect(vector<double> &arr, int ks, MPI_Comm comm);


void MTdistributeToLeaves(pMetricData inData, long rootNpoints,
						double dupFactor, pMetricNode searchNode, 
						double range,
						pMetricData *outData, pMetricNode *leaf);

void MTdistributeToLeaves(pMetricData inData, long rootNpoints,
						double dupFactor, pMetricNode searchNode, 
						pMetricData *outData, pMetricNode *leaf);

void MTdistributeToNearestLeaf( pMetricData inData,
						   pMetricNode searchNode, 
						   pMetricData *outData, 
						   pMetricNode *leaf);

void MTqueryR( pMetricData inData, long rootNpoints, double dupFactor,
				pMetricNode searchNode, double range,
				vector< pair<double, long> > *&neighbors);


void MTqueryR(  pMetricData inData, long rootNpoints, double dupFactor,
				pMetricNode searchNode,
				vector< pair<double, long> > *&neighbors,
				int *nvectors);


void MTqueryRK( pMetricData inData, long rootNpoints, double dupFactor, 
				int k, pMetricNode searchNode,
				vector< pair<double, long> > *&neighbors,int *nvectors);

void MTqueryKSelectRs(pMetricData redistQuery,
		              pMetricData homeData, pMetricNode searchNode,
					  int global_numof_query_points,
					  int k, double **R);

void MTqueryK( pMetricData inData, long rootNpoints, double dupFactor,
				pMetricNode searchNode, int k, 
				vector<long> *queryIDs,
				vector< pair<double, long> > *kNN);


#endif




