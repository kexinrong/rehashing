#ifndef __REPARTITION_H__
#define __REPARTITION_H__

#include <cstdlib>
#include <cmath>
#include <omp.h>
#include "ompUtils.h"
#include "repartition.h"
#include "direct_knn.h"
#include <mpi.h>
#include <map>
#include <queue>
#include <vector>

namespace knn{
	
 namespace repartition {

   void Collect_Query_Results(long *query_ids,
					int *neighbors_per_point, 
					int *neighbor_ids,
					int nPts,
					 MPI_Comm comm);

   /**
    * \param totalClusterSize An array of size numClusters containing the global size of each cluster.
    * \param localClusterSize An array of size numClusters containing the local size of each cluster on this node.
    * \param numClusters Number of clusters
    * \param comm The MPI communicator to use.
    */

     template<typename cSizeType, typename cMembType>
     void redistribute(cSizeType *totalClusterSize, 
                  int numClusters,
                  cMembType *clusterMembership,
                  int n,
                  int num_of_groups,
                  int *group_id_per_point,
                  int *group_id_per_cluster,
                  int **send_count,
                  MPI_Comm comm)
	{
	        int nproc, rank;
	        MPI_Comm_size(comm, &nproc);
	        MPI_Comm_rank(comm, &rank);
	
	        //Priority queue for processes (min heap)
	        std::priority_queue<std::pair<cSizeType, int>, std::vector<std::pair<cSizeType, int> >, std::greater<std::pair<cSizeType, int> > > process_queue;
	
	        for( int i = 0; i < num_of_groups; i++ ) process_queue.push( std::make_pair<cSizeType, int>(0, i) );
	
	        std::pair<cSizeType, int> *pClusters = new std::pair<cSizeType, int>[numClusters];
	        #pragma omp parallel for schedule(static)
	        for(int i = 0; i < numClusters; i++) {
	                pClusters[i].first = totalClusterSize[i];
	                pClusters[i].second = i;
	        }
	
	        omp_par::merge_sort(pClusters, &pClusters[numClusters], std::greater<std::pair<cSizeType,int> >());
	
	        for(int i = 0; i < numClusters; i++) {
	                std::pair<cSizeType, int> emptiestProcess;
	                emptiestProcess.first = process_queue.top().first;
	                emptiestProcess.second = process_queue.top().second;
	                group_id_per_cluster[pClusters[i].second] = emptiestProcess.second;
	
	                //Update queue
	                emptiestProcess.first += pClusters[i].first;
	                process_queue.pop();
	                process_queue.push(emptiestProcess);
	        }
	
	        delete[] pClusters;
	
	        *send_count = new int [num_of_groups];
	        #pragma omp parallel for
	        for(int i = 0; i < num_of_groups; i++)
	                (*send_count)[i] = 0; 
	
	        #pragma omp parallel for
	        for(int i = 0; i < n; i++) 
	                group_id_per_point[i] = group_id_per_cluster[ clusterMembership[i] ];
	
	        for(int i = 0; i < n; i++)
	                (*send_count)[ group_id_per_point[i] ]++;
	
	}


 
  /**
   * repartition query points according the critieria 
   *        dist(center, query) < clusterRadius + range.
   * this code at first duplicate query points if we need to search for knn 
   * on several different processors, and then using 'repartition' function 
   * to exchange data among processors.
   */
  void query_repartition(long *queryIDs,	// in: query ids
		   double *queryPts,		// in: query data
		   double *centers,		// in: cluster centers
		   long nPts,			// in: no. of query points
		   long nClusters,		// in: no. of clusters
		   int dim,			// in: dimensionality of data
		   std::map<int, std::vector<int> > clusterPos,		// in: define each cluster is on which processor
		   double *Radius, 		// in: radius of each cluster [nClusters]
		   double search_range,		// in: search knn within radius 'range'
		   long **new_queryIDs,		// out: query ids after repartition (local)
		   double **new_queryPts,	// out: query points after repartition (local)
		   long *new_nPts,		// out: no. of query points after repartition (local)
		   MPI_Comm comm);
	  
  

   
   
	  
   /**
    * Computing the radius of each clusters
    *
    */
   void Calculate_Cluster_Radius(double * centers,	// in: cluster centers
			  double * points,		// in: points
			  int dim,			// in: dimensionality of each point
			  int * localClusterSize,	// in: indicate each cluster has how many points
			  int numClusters,		// in: no. of clusters
			  double ** Radius,		// out: radius of each cluster
			  MPI_Comm comm);

   /**
    * Redistributes data points and global ids to their respective new owners.  Should be called after
    * cluster_assign and local_rearrange.
    * \param ids An array of size n containing the global identifiers of the corresponding points in data.
    * \param data An array of size n*dim containing the initial data points stored on this node.
    * \param n Number of data points (number of elements in data divided by dim)
    * \param send_count An array of size nproc containing the number of points to send to each process 
    * (obtained by calling cluster_assign).
    * \param new_ids [out] A pointer to an (unallocated) array of new_n longs which will store the ids 
    * after repartitioning. 
    * \param new_data [out] A pointer to an (unallocated) array of new_n*dim longs which will store the data 
    * points after repartitioning.
    * \param new_n [out] The number of data points owned by this process after repartitioning.
    * \param comm The MPI communicator to use.
    */
   int repartition(long *ids, 
			double *data,
			long n,
			int *send_count,
			int dim,
			long **new_ids,	
			double **new_data,
			long *new_n,	
			MPI_Comm comm, int maxNewPoints = 0);
   
   
   
   /**
    * Redistribute query points with individual search radii.
    * \param ids An array of size n containing the global identifiers of the corresponding points in data.
    * \param data An array of size n*dim containing the initial data points stored on this node.
    * \radii The search radii for the local input points.
    * \param n Number of data points (number of elements in data divided by dim)
    * \param send_count An array of size nproc containing the number of points to send to each process 
    * (obtained by calling cluster_assign).
    * \param new_ids [out] A pointer to an (unallocated) array of new_n longs which will store the ids 
    * after repartitioning. 
    * \param new_data [out] A pointer to an (unallocated) array of new_n*dim longs which will store the data 
    * points after repartitioning.
    * \param new_radii [out] The new local search radii after redistributing points (allocated internally).
    * \param new_n [out] The number of data points owned by this process after repartitioning.
    * \param comm The MPI communicator to use.
    */




   void repartition(long *ids,
                                      double *data,
                                      double *radii,
                                      long n,
                                      int *send_count,
                                      int dim,
                                      long **new_ids,
                                      double **new_data, 
                                      double **new_radii,
                                      long *new_n,
                                      MPI_Comm comm);
   


 
   /**
    * Redistributes data points, their global ids, and "secondary IDs" to their respective new owners.  Should be called after
    * cluster_assign and local_rearrange.
    * \param ids An array of size n containing the global identifiers of the corresponding points in data.
    * \param secondaryIDs An array of size n containing the secondary identifiers of the corresponding points in data.
    * \param data An array of size n*dim containing the initial data points stored on this node.
    * \param n Number of data points (number of elements in data divided by dim)
    * \param send_count An array of size nproc containing the number of points to send to each process 
    * (obtained by calling cluster_assign).
    * \param new_ids [out] A pointer to an (unallocated) array of new_n longs which will store the ids 
    * after repartitioning. 
    * \param new_secondaryIDs [out] A pointer to an (unallocated) array of new_n unsigned ints which will store the 
    * secondary ids after repartitioning. 
    * \param new_data [out] A pointer to an (unallocated) array of new_n*dim longs which will store the data 
    * points after repartitioning.
    * \param new_n [out] The number of data points owned by this process after repartitioning.
    * \param comm The MPI communicator to use.
    */
  void repartition(long *ids,
                 unsigned int *secondaryIDs,
                 double *data,
                 long n,
                 int *send_count,
                 int dim,
                 long **new_ids,
                 unsigned int **new_secondaryIDs,
                 double **new_data,
                 long *new_n,
                 MPI_Comm comm);



   /**
    * Rearranges ids, cluster membership labels, and data points so that points sent to a given
    * rank are contiguous and ordered by rank.
    * \param ids A pointer to an array of size n containing the global ids of all locally stored points.
    * \param clusterMembership A pointer to an array of size n containing the id of the cluster 
    * that each point belongs to.
    * \param data A pointer to an array of n*dim doubles containing the coordinates of the points.
    * \param n The number of *local* points.
    * \param dim The dimensionality of the data points.
    */
   void local_rearrange(long **ids, 
		   int **clusterMembership, 
		   double **data, 
		   long n, 
		   int dim);
   /**
    * Rearranges ids, cluster membership labels, and data points so that points sent to a given
    * rank are contiguous and ordered by rank.
    * \param ids A pointer to an array of size n containing the global ids of all locally stored points.
    * \param clusterMembership A pointer to an array of size n containing the id of the cluster 
    * that each point belongs to.
    * \param data A pointer to an array of n*dim doubles containing the coordinates of the points.
    * \param n The number of *local* points.
    * \param dim The dimensionality of the data points.
    */
   void local_rearrange(long **ids, 
		   unsigned int **clusterMembership, 
		   double **data, 
		   long n, 
		   int dim);
  

   /**
    * Rearranges ids, cluster membership labels, and data points so that points sent to a given
    * rank are contiguous and ordered by rank. A secondary identifier (for various uses) is associated
    * with each point and rearranged accordingly.
    * \param ids A pointer to an array of size n containing the global ids of all locally stored points.
    * \param clusterMembership A pointer to an array of size n containing the id of the cluster 
    * that each point belongs to.
    * \param secondaryIDs An array of n "secondary identifiers" for each point (e.g., hash keys).
    * \param data A pointer to an array of n*dim doubles containing the coordinates of the points.
    * \param n The number of *local* points.
    * \param dim The dimensionality of the data points.
    */
   void local_rearrange(long **ids, 
                   int **clusterMembership, 
                   unsigned int **secondaryIDs, 
                   double **data, 
                   long n, 
                   int dim);



   /**
    * Rearranges ids, cluster membership labels, and data points so that points sent to a given
    * rank are contiguous and ordered by rank.
    * \param ids A pointer to an array of size n containing the global ids of all locally stored points.
    * \param clusterMembership A pointer to an array of size n containing the projection of the point
    * that each point belongs to.
    * \param data A pointer to an array of n*dim doubles containing the coordinates of the points.
    * \param n The number of *local* points.
    * \param dim The dimensionality of the data points.
    */
   void local_rearrange(long **ids, 
		   double **clusterMembership, 
		   double **data, 
		   long n, 
		   int dim);

	void pre_all2all(long *ids, int *membership,
					double *data, long n, int dim);


   void rearrange_data( int *clusterMembership, 
		   	double *data, 
		   	long n, 
		   	int dim, 
			double *re_data);

   /**
    * Calculates how many points this process will send to each process in comm.
    * \param totalClusterSize An array of length numClusters which contains the number of *global*
    * points assigned to each cluster.
    * \param numClusters The total number of clusters.
    * \param clusterMembership An array of length n containing the id of the cluster to which
    * each point belongs.
    * \param n The number of locally stored points.
    * \param comm The MPI communicator that contains all processes that have points.
    * \return An array of length equal to the size of comm containing the number of points to 
    * send to each rank.
    */
   int* calculate_send_counts(int *totalClusterSize,  
                    int numClusters,
		    int *clusterMembership,
		    int **clusterLocations,
		    long n, 
                    MPI_Comm comm);
   
   /**
    * Move points to make workload balanced in comm.
    * \param points An array of length 2dn/p to store points
    * \param gids An array of length 2n/p to store globle ids
    * \param numof_points The number of locally stored points.
    * \param dim dimensionality of points
    * \param maxIter The maximum number of points
    * \param dim dimensionality of points
    * \param nem_numof_points The number of points after do load balance.
    * \param comm The MPI communicator that contains all processes that have points.
    */
 	void loadBalance(double *points, long *gids, int numof_points, int dim, int n_over_p, 
					//output
					int &new_numof_points,
					MPI_Comm comm);


   /** 
    * Exchanges points between a pair of processes.
    * \param partner_rank Rank of this process's comminication parter in comm.
    * \param maxpts The maximum number of points allowed per node.
    * \param n_send Number of points to send to partner.
    * \param dim Dimension of points.
    * \param point_sendbuf Send buffer for data points.
    * \param id_sendbuf Send buffer for point IDs.
    * \param point_recvbuf Receive buffer for data points.
    * \param id_recfbuf Receive buffer for point IDs.
    * \param comm MPI communicator.
    * \return The number of points received from partner.
    */
    int pairwise_exchange( int partner_rank, int maxpts, int n_send, int dim, double *point_sendbuf,
                           long *id_sendbuf, double *point_recvbuf, long *id_recvbuf, MPI_Comm comm );




   /**
    * Repartition points for a binTree node split; points must already be correctly ordered (all left child points
    * must be at the beginning of the array).
    * \param ids [inout] An array of size n containing the global identifiers of the corresponding points in data; must have
    * at least 2*maxpoints space allocated..
    * \param data [inout] An array of size n*dim containing the initial data points stored on this node; must have
    * at least 2*maxpoints*dim space allocated.
    * \param nLocal The number of points stored locally.
    * \param maxpts The maximum number of points allowed per node.
    * \param child_tag An array of length nLocal indicating to wich child each point belongs (0=left, 1=right).
    * \param dim The dimensionality of the data points.
    * points after repartitioning.
    * \param comm The MPI communicator to use.
    * \return The new number of points stored on this process.
    */
    int tree_repartition(long *ids,
                 double *data,
                 int nLocal,
                 int maxpts,
                 int *child_tag,
                 int dim,
                 MPI_Comm comm);


    int tree_repartition_arbitraryN(std::vector<long> &ids,
                 std::vector<double> &data,
                 int nLocal,
                 int *child_tag,
                 int *rank_colors,
                 int dim,
                 MPI_Comm comm);

    void loadBalance_arbitraryN(std::vector<double> &points, std::vector<long> &gids, int numof_points, int dim, 
                                 //output
                                 int &new_numof_points,
                                 MPI_Comm comm);

  }
}

#endif
