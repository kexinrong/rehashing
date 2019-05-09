#ifndef __CLUSTERING_H__
#define __CLUSTERING_H__

#include <mpi.h>
#include <vector>

using namespace std;

/** 
  * give m different random int in a range [0, N)
  * \param m number of random integer to generate
  * \param N range of random integer is [0, N)
  * \param arr output arr storing the random integer
  */
void randperm(int m, int N, vector<int>& arr);




 /**
  * Calculating the squared L2 distance between ref points and query points without uisng BLAS
  * it should be faster than with BLAS if the number of reference points or query points is much larger than the other
  * \param ref An array of size [n*dim] containing the reference data points stored on this node
  * \param query An array of size [m*dim] containing the query data points stored on this node
  * \param n Number of reference data points 
  * \param m Number of query data points 
  * \param dim The dimensionality of the data ponts
  * \param dist An array of size [n*m] containing the squared distance among ref and query
  */
void SqEuDist(double *ref, 
	      double *query, 
	      long n, 
	      long m, 
	      int dim, 
	      double *dist);

 /**
  * Clustering all points in the communicator 'comm' by k-means which at first search for good seeds 
  * can be called after "add_seeds()" to reduce the iterations.
  * \param points An array of size [numPoints*dim] containing the initial data points stored on this node
  * \param dim The dimensionality of the data ponts
  * \param numPoints Number of data points 
  * \param numClusters Number of clusters wanted
  * \param maxIter Maximum iteration of k-means algorithm (can be small when after call "add_seeds()")
  * \param seedType type of seeding algorithm (0: random seeds, 1 : ostrovsky seeds)
  * \param membership An array containing cluster labels of each points
  * \param clusters An array containing the centroids of all cluters
  * \param clusterSize An array containing the global size of each clusters
  * \param p_clusterSize An array containing the local size of each clusters (no. of points in each cluster on this node)
  */
int k_clusters(double    *points,     	// input: [numPoints][dim]
               int        dim,   	// input: dimensionality of data
               long        numPoints,    // input: number of data
               int        numClusters, 	// input: number of clusters
	       int 	  maxIter,	// input: maximum iteration of k-means
               int 	 seedType,	// input: seeding type
	       int       *membership,  	// output: cluster lable of each point
               double    *clusters,    	// output: centroids of data [numClusters][dim]
               int 	 *clusterSize,	// output: totoal cluster size [numClusters]
	       int	 *p_clusterSize,
	       MPI_Comm   comm);       	// MPI Communicator


 /**
  * Clustering all points in the communicator 'comm' by k-means which at first search for good seeds, run several times, and choose the centers which give the best load balances, i. e. the maximum of (min(clusterSize)/max(clusterSize) )
  * \param points An array of size [numPoints*dim] containing the initial data points stored on this node
  * \param dim The dimensionality of the data ponts
  * \param numPoints Number of data points 
  * \param numClusters Number of clusters wanted
  * \param maxIter Maximum iteration of k-means algorithm (can be small when after call "add_seeds()")
  * \param seedType type of seeding algorithm (0: random seeds, 1 : ostrovsky seeds)
  * \param nfold Run nfold times to choose the one with best load balance
  * \param membership An array containing cluster labels of each points
  * \param clusters An array containing the centroids of all cluters
  * \param clusterSize An array containing the global size of each clusters
  * \param p_clusterSize An array containing the local size of each clusters (no. of points in each cluster on this node)
  */
int k_clusters_balanced(double    *points,     	// input: [numPoints][dim]
               int        dim,   	// input: dimensionality of data
               long        numPoints,    // input: number of data
               int        numClusters, 	// input: number of clusters
	       int 	  maxIter,	// input: maximum iteration of k-means
               int 	 seedType,	// input: seeding type
	       int 	 nfold,
	       int       *membership,  	// output: cluster lable of each point
               double    *centers,    	// output: centroids of data [numClusters][dim]
               int 	 *clusterSize,	// output: totoal cluster size [numClusters]
	       int	 *p_clusterSize,
	       MPI_Comm   comm);       	// MPI Communicator


 /**
  * Clustering all points in the communicator 'comm' by k-means using the given seeds, 
  * \param points An array of size [numPoints*dim] containing the initial data points stored on this node
  * \param dim The dimensionality of the data ponts
  * \param numPoints Number of data points 
  * \param numClusters Number of clusters wanted
  * \param maxIter Maximum iteration of k-means algorithm (can be small when after call "add_seeds()")
  * \param membership An array containing cluster labels of each points
  * \param clusters An array containing the centroids of all cluters
  * \param clusterSize An array containing the global size of each clusters
  * \param p_clusterSize An array containing the local size of each clusters (no. of points in each cluster on this node)
  */
pair<double, int> mpi_kmeans(double    *points,
               	             int        dim,   
                             long        numPoints,
                             int        numClusters,
	                     int 	  maxIter,
                             int       *membership,  
                             double    *clusters,   
                             int 	 *clusterSize,
	                     int	 *p_clusterSize,
	                     MPI_Comm   comm);


 /**
  * Calculate the Variance Ratio of the clustering results 
  * \param points An array of size [numof_points*dim] containing the initial data points stored on this node
  * \param num_points Number of data points 
  * \param dim The dimensionality of the data ponts
  * \param centroids An array containing the centroids of all cluters
  * \param num_centers Number of clusters wanted
  * \param glb_cluster_size An array containing the global size of each clusters
  * \param comm The MPI communicator to use
  */
double VarianceRatio(double *points, 
		     int numof_points, 
		     int dim,
		     double *centroids, 
		     int numof_centers, 
		     int *glb_cluster_size,
		     MPI_Comm comm);

 /**
  * Computing the mean of all points in the communicator 'comm'
  * \param X An array of size [n*dim] containing the points on this node
  * \param loc_n Local number of points on this node
  * \param glb_n Global number of points on the input comm
  * \param dim The dimensionality of data points
  * \param comm The MPI communicator to use
  */
double *centroids(double *X,
		int loc_n,
		int glb_n,
		int dim,
		MPI_Comm comm);

 /**
  * Sampling 'k' data points out of all points in the communicator 'comm'  
  * according to points' weight
  * \param X An array of size [n*dim] containing the points on this node
  * \param n Number of points on this node
  * \param dim The dimensionality of data points
  * \param k Number of samples we want
  * \param comm The MPI communicator to use
  */
double *sample(double *X, 
		double *pw,
		int n,
		int dim,
		int k,
		int idx_base,
		MPI_Comm comm);


 /**
  * Choosing first two seeds out of all points in this communicator 'comm'  
  * \param X An array of size [n*dim] containing the points on this node
  * \param n Number of points on this node
  * \param dim The dimensionality of data points
  * \param comm The MPI communicator to use
  */
double *initial_two_seeds(double *X, 
		int n,
		int dim,
		MPI_Comm comm);


/**
  * Choosing 'numSeeds' seeds out of all points in this communicator 'comm'
  * \param X An array of size [n*dim] containing the points on this node
  * \param numPoints Number of points on this node
  * \param dim The dimensionality of data points
  * \param seeds An array of size [numSeeds*dim] containing the 'numSeeds' seeds
  * \param numSeeds Number of seeds we want to choose
  * \param current_num_seeds Number of seeds we have alreay haven in the array 'seeds'
  * \param comm The MPI communicator to use
  */
void addSeeds(double *X,
		int numPoints, 
		int dim,
		double *seeds,
		int numSeeds,
		int *current_num_seeds,
		MPI_Comm comm);

/**
  * Excluding point at position idx from the array 'points'
  * \param point An array of size [numof_points*dim] containing the points
  * \param numof_points Number of points
  * \param dim The dimensionality of data points
  * \param idx Position of point need to be excluded. 
  * \param subset An arry of size [(numof_points-1)*dim] containing the remaining points 
  */
void exclusive_subset(double *points, int numof_points, int dim,
		      int idx, double *subset);

 /**
  * eliminating 'bad' seeds to reduce the oversampled seeds to the number of 'numSeeds'
  * \param points An array of size [numof_points*dim] containing the points on this node
  * \param numof_points Number of points on this node
  * \param dim The dimensionality of data points
  * \param oversampled_seeds An array containing oversampled seeds
  * \param numof_seeds Number of seeds we want to eliminate to
  * \param seeds An array of size [numof_seeds*dim] containing the finally sampled seeds
  * \param comm The MPI communicator to use
  */
void eliminateSeeds(double *points, int numof_points, int dim,
		    vector<double> & oversampled_seeds,
		      int numof_seeds, double *seeds,
		      MPI_Comm comm);

  /**
  * Choosing 'numof_clusters' seeds out of all points RANDOMLY in this communicator 'comm'
  * \param points An array of size [numof_points*dim] containing the points on this node
  * \param numof_points Number of points on this node
  * \param dim The dimensionality of data points
  * \param numof_clusters Number of seeds we want to choose
  * \param seeds An array of size [numof_clusters*dim] containing the 'numSeeds' seeds
  * \param comm The MPI communicator to use
  */
void randomSeeds(double *points,
		  int numof_points,
		  int dim,
		  int numof_clusters,
		  double *seeds,
		  MPI_Comm comm);


  /**
  * Choosing 'numof_clusters' seeds out of all points in the Ostrovsky's approach in this communicator 'comm'
  * \param points An array of size [numof_points*dim] containing the points on this node
  * \param numof_points Number of points on this node
  * \param dim The dimensionality of data points
  * \param numof_clusters Number of seeds we want to choose
  * \param seeds An array of size [numof_clusters*dim] containing the 'numSeeds' seeds
  * \param comm The MPI communicator to use
  */
void ostrovskySeeds(double *points, int numof_points, int dim,
		      int numof_seeds, double *seeds,
		      MPI_Comm comm);



void findFurthestPoint(// input
						double *points, 
						int numof_points, 
						int dim, 
						double *query,
						// output
						double *furP,
						MPI_Comm comm);

void calProjDirection(// input
						double * points, 
						int numof_points, 
						int dim,
					  // output
						double * proj,
						MPI_Comm comm);

// select the kth smallest element in arr
// for median, ks = glb_N / 2
double distSelect(	vector<double> &arr, 
					int ks, 
					MPI_Comm comm);

void equal2clusters(double * points, 
					int numof_points, 
					int dim,
					// output
					int* point_to_hyperplane_membership,
					double* centroids,
					int* global_numof_points_per_hyperplane,
					int* local_numof_points_per_hyperplane, 
					MPI_Comm comm);
	

#endif
