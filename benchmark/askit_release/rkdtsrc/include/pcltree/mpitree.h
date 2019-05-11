#ifndef _MPITREE_H_
#define _MPITREE_H_

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
class MTNode; 
class MTData;
typedef MTNode *pMTNode;
typedef MTData *pMTData;

// error checking for MPI CALLS
#define iM(fun) { int ierr=(fun); assert(ierr==MPI_SUCCESS); } 
#define PRINTMPT /*{ cout<< "MPT: "; } */
#define PRINTSELF( fun ) /* { PRINTMPT; cout<<"["<<rank<<"]"<<" "; (fun); } */
#define PRINTWORLD( fun) /* { if(root.options.verbose){ if(!rank){ PRINTMPT; PRINTSELF(run);} } } */

#define DBG_PRINTMPT /* {cout<< "MPT: ";} */
#define DBG_PRINTSELF( fun ) /*{ if(root.options.verbose){ PRINTMPT; cout<<"["<<rank<<"]"<<" "; (fun); } } */
#define DGB_PRINTWORLD( fun) /*{ if(root.options.verbose){ if(!rank){ PRINTMPT; PRINTSELF(run);} } } */


/* ************************************************** */

/**
	Auxiliary data structure to hold point coords (X), their dimension (dim) an global ids.
 */
struct MTData {
	vector<double> X;    ///< Data point coordinates.
	int dim;             ///< Dimensionality of points.
	vector<long> gids;   ///< global ids of points.
	vector<double> radii; ///< Search radii of points (only meaningful if this MTData object is a query point set).
	//-------------  Methods
	void Copy(pMTData data){
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
 * list of MTNode objects, representing a path from the root to that process's leaf node.  
 */
class MTNode {
public:
	pMTData data; ///< struct that stores points and global ids.
	int level;    ///< level of the node.  Root is level 0.
	MPI_Comm comm;  ///< communicator of the node.
	vector <char>path;    ///<path can be used as a unique identifier.
	
	pMTNode parent;	  ///< Pointer to parent
	pMTNode kid;     ///< the next level

        int Nglobal;  ///< The total number of points stored in or beneath this tree node, across all member processes.

	vector<double> C; ///< cluster centers for kids (or centroid of locally stored points if this is a leaf).
	vector<double> R; ///< cluster radii for every kid (or of locally stored points if this is a leaf).
	vector<int> cluster_to_kid_membership; ///< Element i stores the child id (0 or 1 for binary tree) that
					       ///< cluster i (with center C[i*dim]) is assigned to.

	vector<int> rank_colors;  ///< Length is equal to size of comm. Stores the child that each MPI rank belongs to.
	int chid;	///< This node's child id.


	//-------------  Methods
	/// Default constructor, comm is initialized to MPI_COMM_WORLD.
	MTNode() : data(NULL),level(0),chid(0),comm(MPI_COMM_WORLD),parent(NULL),kid(NULL) {;}
        /**
         * Constructor.  Must be used for non-root nodes.
         * \param ci The id of the child (0 or 1 for binary tree nodes).
         */
	MTNode(int ci) : data(NULL),level(0),chid(ci),comm(MPI_COMM_WORLD),parent(NULL),kid(NULL) {;}
	
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
	 * \param pMTData Holds ids and point coordinates: IT IS MODIFIED.
	 */
	void Insert(pMTNode inParent, int maxp, int maxlev, int minCommSize, MPI_Comm comm, pMTData inData, int seedType);

 public:
	/**
	 * Options for MTNode tree class.
	 */
	struct Options{
		int debug_verbose;   ///< Print extensive information and/or compare results against direct search.
		int timing_verbose;  ///< Currently unused.
    	int pruning_verbose; ///< Print pruning statistics in tree query operations.
		static const int max_max_treelevel=24;  ///< Maximum allowable value for maxlev.
		int cluster_factor;
		Options() : debug_verbose(false),timing_verbose(false),pruning_verbose(false),cluster_factor(1) {;}
	};
	Options options;  ///< This nodes Options object.
};



/**
 * Given a communicator, a workload in each process, and a set of group assignments
 * for each point; determine group to which each process is assigned, and assign each
 * point to a process owned by its group.  For internal use.
 */
int groupDistribute(
	int *item_group, ///< store the group id for each item
	int numof_items,    ///<  number of items
	int numof_groups,   ///< number of groups 
	MPI_Comm comm,   ///< mpi communicator
	int &my_group_id,  ///< [out] the group the I belong to
	int *proc_id ///< [out] the processor id for each data item
);
	

/**
 * Partitions size MPI processes into 2 groups, so that the work/process in each group
 * is roughly the same. For internal use.
 * \param work_per_group The number of points stored in each group.
 * \param size Total number of processes available (size of MPI communicator).
 * \param group_size [out] Number of processes to assign to each group.
 */
void work_partition( vector<int> work_per_group, int size, vector<int> &group_size);


/**
 * For tree traversals, determine how many query points are to be assigned to each process of
 * each child node.
 * \param numof_querypoints The number of query points to distribute.
 * \param rank_colors The mapping of MPI ranks in the current tree node's communicator to child nodes (length=size).
 * \param size The size of the current node's MPI communicator.
 * \param point_to_kid_membership The mapping of query points too child nodes (length=numof_querypoints).
 * \param send_count [out] The number of query points to transmit to each process in current node's communicator.
 */
void group2rankDistribute(int numof_querypoints,
			int *rank_colors,
			int size,
			int * point_to_kid_membership,
			int *send_count);



/**
 * Performs an exact radius-query on the tree, returning all reference points within range of each query point.
 * \param queryData [in,out] The set of query point coordinates and IDs; modified to contain the query points
 * corresponding to the neighbors returned by this process.
 * \param rootNpoints The total number of points stored in the tree.
 * \param dupFactor Determines the maximum number of query points that can be handled by a given process (mulitplied by numGlobalQueries/size).
 * \param searchNode The root node for the search.
 * \param range The search radius.
 * \param neighbors [out] The r-near neighbors found by the search.
 */
void queryR(pMTData queryData, long rootNpoints, double dupFactor, 
		pMTNode searchNode, double range,
		vector< pair<double, long> > *&neighbors);


/**
 * Performs an exact radius-query on the tree with individual search radii for each query point.
 * \param queryData [in,out] The set of query point coordinates and IDs; modified to contain the query points
 * corresponding to the neighbors returned by this process. Must have its radii array populated.
 * \param rootNpoints The total number of points stored in the tree.
 * \param dupFactor Determines the maximum number of query points that can be handled by a given process (mulitplied by numGlobalQueries/size).
 * \param searchNode The root node for the search.
 * \param neighbors [out] The r-near neighbors found by the search.
 * \param nvectors [out] The length of the neighbors array (number of neighbor vectors).
 */
void queryR( pMTData queryData, long rootNpoints, double dupFactor,
            pMTNode searchNode, 
            vector< pair<double, long> > *&neighbors, int *nvectors);


/**
 * Performs an exact radius-query on the tree with individual search radii for each query point, keeping only the k nearest
 * neighbors returned for each query point. This is primarily for internal use and is used by the exact queryK function.
 * \param queryData [in,out] The set of query point coordinates and IDs; modified to contain the query points
 * corresponding to the neighbors returned by this process. Must have its radii array populated.
 * \param rootNpoints The total number of points stored in the tree.
 * \param dupFactor Determines the maximum number of query points that can be handled by a given process (mulitplied by numGlobalQueries/size).
 * \param k The number of near neighbors to keep.
 * \param searchNode The root node for the search.
 * \param neighbors [out] The r-near neighbors found by the search.
 * \param nvectors [out] The length of the neighbors array (number of neighbor vectors).
 */
void queryRK( pMTData queryData, long rootNpoints, double dupFactor, int k,
            pMTNode searchNode,
            vector< pair<double, long> > *&neighbors, int *nvectors);


/**
 * Performs an approximate radius-query on the tree (using cyclic distributed LSH), returning all reference points within range of each query point.
 * \param queryData [in,out] The set of query point coordinates and IDs; modified to contain the query points
 * corresponding to the neighbors returned by this process.
 * \param rootNpoints The total number of points stored in the tree.
 * \param dupFactor Determines the maximum number of query points that can be handled by a given process (mulitplied by numGlobalQueries/size).
 * \param searchNode The root node for the search.
 * \param range The search radius.
 * \param neighbors [out] The r-near neighbors found by the search.
 */
void queryR_Approx(pMTData queryData, long rootNpoints, double dupFactor, 
		pMTNode searchNode, double range,
		vector<long> *&neighbors);



/**
 * Performs an approximate radius-query on the tree (using partitioned LSH), returning at most k reference points within range of each query point, in sorted order, along with the squared distances.
 * \param queryData [in,out] The set of query point coordinates and IDs; modified to contain the query points
 * corresponding to the neighbors returned by this process.
 * \param rootNpoints The total number of points stored in the tree.
 * \param dupFactor Determines the maximum number of query points that can be handled by a given process (mulitplied by numGlobalQueries/size).
 * \param searchNode The root node for the search.
 * \param range The search radius.
 * \param k The maximum number of neighbors to find.
 * \param c The approximation factor for the LSH search, higher values results in lower accuracy (3.0 is usually a good value).
 * \param max_iters The maximum number of iterations for the LSH search (supersedes automatically selected value of L).
 * \param neighbors [out] The k-r near neighbors found by the search.
 * \param queryIDs [out] The IDs of the query points whose neighbors are stored locally.  queryIDs[i] is the ID of the point
 * whose neighbors are stored in the vector neighbors[i].
 */
void queryR_ApproxK( pMTData queryData, long rootNpoints, double dupFactor,
             pMTNode searchNode,
             double range, int k, int c, int max_iters,
             vector<pair<double,long> > *&neighbors, vector<long> &queryIDs);



/**
 * Exact k-nearest neighbor query using the PCL-tree.
 * \param queryData The set of query point coordinates and IDs; modified but does not contain meaningful output.
 * \param rootNpoints The total number of points stored in the tree.
 * \param dupFactor Determines the maximum number of query points that can be handled by a given process (mulitplied by numGlobalQueries/size).
 * \param searchNode The root of the tree to search.
 * \param k The number of nearest neighbors to find.
 * \param queryIDs [out] A pointer to an externally allocated vector to which the query ID corresponding to each set of neighbors will be written.
 * \param kNN [out] A pointer to an externally allocated vector containing the nearest neighbors found.  The neighbors of point i are stored contiguously beginning at kNN[i*k], and the ID of point i will be stored in queryIDs[i].
 * \note The communicator associated with searchNode *must* be MPI_COMM_WORLD.  That is, the tree must be distributed over all MPI processes in the application.
 * \node If the tree has options.debug_verbose set to true, this function will compare its results against a direct search.  This should be set for testing only.
 */
void queryK( pMTData queryData, long rootNpoints, double dupFactor,
                pMTNode searchNode,
                int k, vector<long> *queryIDs,
                vector< pair<double, long> > *kNN);



/**
 * Performs an approximate k-neareat query on the PCL-tree (using partitioned LSH), returning the approximate k nearest neighbors to each query point, in sorted order, along with the squared distances.
 * \param queryData The set of query point coordinates and IDs; modified but does not contain meaningful output.
 * \param rootNpoints The total number of points stored in the tree.
 * \param dupFactor Determines the maximum number of query points that can be handled by a given process (mulitplied by numGlobalQueries/size).
 * \param searchNode The root node for the search.
 * \param range The search radius.
 * \param k The maximum number of neighbors to find.
 * \param c The approximation factor for the LSH search, higher values results in lower accuracy (3.0 is usually a good value).
 * \param max_iters The maximum number of iterations for the LSH search (supersedes automatically selected value of L, 100-200 is recommended).
 * \param neighbors [out] The near neighbors found by the search.
 * \param queryIDs [out] The IDs of the query points whose neighbors are stored locally.  queryIDs[i] is the ID of the point
 * whose neighbors are stored in the vector neighbors[i].
 * \note The communicator associated with searchNode *must* be MPI_COMM_WORLD.  That is, the tree must be distributed over all MPI processes in the application.
 */
void queryK_Approx( pMTData queryData, long rootNpoints, double dupFactor,
             pMTNode searchNode,
             double range, int k, int c, int max_iters,
             vector<pair<double,long> > *&neighbors, vector<long> &queryIDs);







/**
 * Performs an approximate k-neareat query on the PCL-tree by greedily traversing the tree to the leaf nearest to each query poing (LSH not used).
 * \param queryData The set of query point coordinates and IDs; modified but does not contain meaningful output.
 * \param rootNpoints The total number of points stored in the tree.
 * \param searchNode The root node for the search.
 * \param k The maximum number of neighbors to find.
 * \param queryIDs [out] The IDs of the query points whose neighbors are stored locally.  queryIDs[i] is the ID of the point
 * whose neighbors are stored in the vector kNN[i].
 * \param neighbors [out] The k nearest neighbors found for each query point.
 * \note The communicator associated with searchNode *must* be MPI_COMM_WORLD.  That is, the tree must be distributed over all MPI processes in the application.
 */
void queryK_GreedyApprox( pMTData queryData, long rootNpoints, 
                pMTNode searchNode, 
                int k, vector<long> *queryIDs,
                vector< pair<double, long> > *kNN);



/**
 * Traverses the tree, redistributing query points to the appropriate leaf node(s) using a 
 * single search radius.
 * \param queryData The query points that belong to the current tree node.
 * \param root The root of the (sub)tree to distribute the query points to.
 * \param range The search radius for all query points.
 * \param outData [out] The query points that belong to this process's leaf node.
 * \param leaf [out] Pointer to this process's leaf node.
 */
void distributeToLeaves(pMTData queryData, 
                        pMTNode root, 
                        double range, 
                        pMTData *outData, pMTNode leaf);



/**
 * Traverses the tree, redistributing query points to the appropriate leaf node(s) using a 
 * individual search radii.
 * \param queryData The query points that belong to the current tree node.
 * \param root The root of the (sub)tree to distribute the query points to.
 * \param outData [out] The query points that belong to this process's leaf node.
 * \param leaf [out] Pointer to this process's leaf node.
 * \note Requires that radii array of queryData be populated.
 */
void distributeToLeaves(pMTData queryData, 
                        pMTNode root, 
                        pMTData *outData, pMTNode leaf);



/**
 * Traverses the tree, redistributing query points to the appropriate leaf node(s) using a 
 * single search radius.
 * \param queryData The query points that belong to the current tree node.
 * \param root The root of the (sub)tree to distribute the query points to.
 * \param range The search radius for all query points.
 * \param outData [out] The query points that belong to this process's leaf node.
 * \param leaf [out] Pointer to this process's leaf node.
 */
void distributeToLeaves(pMTData queryData, long rootNpoints, double dupFactor, 
                        pMTNode root, 
                        double range, 
                        pMTData *outData, pMTNode *leaf);


/**
 * Traverses the tree, redistributing query points to the appropriate leaf node(s) using a 
 * individual search radii.
 * \param queryData The query points that belong to the current tree node.
 * \param root The root of the (sub)tree to distribute the query points to.
 * \param outData [out] The query points that belong to this process's leaf node.
 * \param leaf [out] Pointer to this process's leaf node.
 * \note Requires that radii array of queryData be populated.
 */

void distributeToLeaves(pMTData queryData, long rootNpoints, double dupFactor,
                        pMTNode root, 
                        pMTData *outData, pMTNode *leaf);



/**
 * Traverses the tree, redistributing each query point to the leaf node with the nearest cluster center.
 * \param queryData The query points that belong to the current tree node.
 * \param root The root of the (sub)tree to distribute the query points to.
 * \param outData [out] The query points that belong to this process's leaf node.
 * \param leaf [out] Pointer to this process's leaf node.
 */
void distributeToNearestLeaf( pMTData queryData,
            pMTNode searchNode, 
            pMTData *outData, pMTNode *leaf);


#endif





