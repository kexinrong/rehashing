#ifndef __DIRECT_KNN_H__
#define __DIRECT_KNN_H__

#include <mpi.h>
#include <utility>
#include <cmath>
#include <vector>
#include <string>
#include <blas.h>
#include <omp.h>
#include <utility>

#define KNN_MAX_BLOCK_SIZE 512
#define KNN_MAX_MATRIX_SIZE 2e7L

// MPI message tags
enum {
  TAG_R,
  TAG_Q,
  TAG_KMIN,
  TAG_ID,
  TAG_SIZE
}; 

enum {
  VERTICAL,
  HORIZONTAL
};


/**
 * Options struct used to specify how a rectangular-partitioned query should be performed.
 */
struct directQueryParams
{
	char queryType; ///< Perform a k-nearest ('K') or r-near ('R') search.
	int k; ///< Number of neighbors to find (if queryType is 'K').
	double r; ///< Radius for r-near search (if queryType is 'R').
	int refParts; ///< Partition reference points into this many approximately equal parts.
	int queryParts; ///< Partition query points into this many approximately equal parts.
}; 

struct directQueryResults
{
	std::pair<double, long> *neighbors;
	long *neighborCounts;
};

class maxheap_comp {
    public:
        bool operator() (const std::pair<double, long> &a, const std::pair<double, long> &b) {
            double diff = fabs(a.first-b.first)/a.first;
            if( std::isinf(diff) || std::isnan(diff) ) {      // 0/0 or x/0
                return a.first < b.first;
            }
            if( diff < 1.0e-8 ) {
                return a.second < b.second;
            }
            return a.first < b.first;
        }
};


namespace knn {

   /**
    * Compute the k nearest neighbors of each query point (OBSOLETE: For debugging purposes only; use directKQueryLowMem instead)..
    * \param ref an n*dim-length array of data points.
    * \param query an m*dim-length array of query points.
    * \param n number of local reference point
    * \param m number of local query points
    * \param k number of neighbors to find for each query point.
    * \param dim dimensionality of reference/query points
    * \return An array of pairs containing the *squared* distance to each point and its 
    *	index in the local array.  Note that this is the index of the /point/ rather
    *	than the index of its location in the actual array of doubles.  To access the 
    *	point, use (point index)*dim to index ref.
    */
   std::pair<double, long> *directKQuery
              ( double *ref, double *query, long n, long m, long k, int dim);


   /**
    * Compute the k nearest neighbors of each query point (OBSOLETE: For debugging purposes only; use directKQueryLowMem instead)..
    * \param ref an n*dim-length array of data points.
    * \param query an m*dim-length array of query points.
    * \param n number of local reference point
    * \param m number of local query points
    * \param k number of neighbors to find for each query point.
    * \param dim dimensionality of reference/query points
    * \param result [out] An array of m*k pairs (allocated by caller) containing the 
    * *squared* distance to each point and its index in the local array.
    * Note that this is the index of the /point/ rather than the index of its location 
    * in the actual array of doubles.  To access the point, use (point index)*dim to index ref.
    * Results for query point i begin at result[i*k].
    * \param dist An array used for calculating inter-point distances. If NULL or unspecified,
    * the function allocates and deallocates this array internally.  Otherwise, it must be
    * allocated size at least n*B, where B is the value returned by getBlockSize, and
    * deallocated at some point after the function returns.
    * \param sqnormr An array used for calculating inter-point distances. If NULL or unspecified,
    * the function allocates and deallocates this array internally.  Otherwise, it must be
    * allocated size at least n and deallocated at some point after the function returns.
    * \param sqnormq An array used for calculating inter-point distances. If NULL or unspecified,
    * the function allocates and deallocates this array internally.  Otherwise, it must be
    * allocated size at least B, where B is the value returned by getBlockSize, and
    * deallocated at some point after the function returns.
    */
 void directKQuery_small_a2a( double *ref, long n, int dim, int k,
                              std::pair<double, long> *result,
                              double *dist = NULL, double *sqnormr = NULL, double *sqnormq = NULL);



   /**
    * Compute the k nearest neighbors of each query point with a smaller memory footprint and usually better performance.
    * \param ref an n*dim-length array of data points.
    * \param query an m*dim-length array of query points.
    * \param n number of local reference point
    * \param m number of local query points
    * \param k number of neighbors to find for each query point.
    * \param dim dimensionality of reference/query points
    * \param result [out] An array of m*k pairs (allocated by caller) containing the 
    * *squared* distance to each point and its 
    * index in the local array.  Note that this is the index of the /point/ rather
    * than the index of its location in the actual array of doubles.  To access the 
    * point, use (point index)*dim to index ref. Results for query point i begin at result[i*k].
    * \param dist An array used for calculating inter-point distances. If NULL or unspecified,
    * the function allocates and deallocates this array internally.  Otherwise, it must be
    * allocated size at least n*B, where B is the value returned by getBlockSize, and
    * deallocated at some point after the function returns.
    * \param sqnormr An array used for calculating inter-point distances. If NULL or unspecified,
    * the function allocates and deallocates this array internally.  Otherwise, it must be
    * allocated size at least n and deallocated at some point after the function returns.
    * \param sqnormq An array used for calculating inter-point distances. If NULL or unspecified,
    * the function allocates and deallocates this array internally.  Otherwise, it must be
    * allocated size at least B, where B is the value returned by getBlockSize, and
    * deallocated at some point after the function returns.
    */
   void directKQueryLowMem
              ( double *ref, double *query, long n, long m, long k, int dim, std::pair<double, long> *result,
                double *dist = NULL, double* sqnormr = NULL, double* sqnormq = NULL  );

   /**
    * Find all points in ref that lie within distance sqrt(R) of each 
    * point in query. 
    * \param ref an n*dim-length array of data points.
    * \param query an m*dim-length array of query points.
    * \param n number of local reference point
    * \param m number of local query points
    * \param R The *square* of the search radius.
    * \param dim dimensionality of reference/query points
    * \param glob_ids An array of length n containing the global id of each ref point.
    * \param neighbor_count [out] An array of length m containing the number of 
    * neighbors found for the corresponding query point (allocated internally).
    * \param neighbors [out] An array of length sum(neighbor_count) containing 
    * pairs of ids and squared distances for each neighbor found (allocated 
    * internally).
    */
   void directRQuery 
           ( double *ref, double *query, long n, long m, double R, int dim, long* glob_ids,
             int **neighbor_count, std::pair<double, long> **neighbors ); 
  
  
  /**
   * Find all points in ref that lie within distance sqrt(R[i]) of query point i. 
    * \param ref an n*dim-length array of data points.
    * \param query an m*dim-length array of query points.
   * \param n number of local reference point
   * \param m number of local query points
   * \param R Array of length m, the *square* of the search radius for each point.
   * \param dim dimensionality of reference/query points
   * \param glob_ids An array of length n containing the global id of each ref point.
   * \param neighbor_count [out] An array of length m containing the number of 
   * neighbors found for the corresponding query point (allocated internally).
   * \param neighbors [out] An array of length sum(neighbor_count) containing 
   * pairs of ids and squared distances for each neighbor found (allocated 
   * internally).
   */
  void directRQueryIndividual
  ( double *ref, double *query, long n, long m, double *R, int dim, long* glob_ids,
   int **neighbor_count, std::pair<double, long> **neighbors ); 
 


  /**
   * Find all points in ref that lie within distance sqrt(R[i]) of query point i. 
    * \param ref an n*dim-length array of data points.
    * \param query an m*dim-length array of query points.
   * \param n number of local reference point
   * \param m number of local query points
   * \param R Array of length m, the *square* of the search radius for each point.
   * \param dim dimensionality of reference/query points
   * \param glob_ids An array of length n containing the global id of each ref point.
   * \param neighbor_count [out] An array of length m containing the number of 
   * neighbors found for the corresponding query point (allocated internally).
   * \param neighbors [out] An array of length sum(neighbor_count) containing 
   * pairs of ids and squared distances for each neighbor found (allocated 
   * internally).
   */
void directRQueryIndividualK
( double *ref, double *query, long n, long m, int k, double *R, int dim, long *glob_ids,
 int **neighbor_count, std::pair<double, long> **neighbors  );



   /**
    * Computes the distances between all query points and all reference points.
    * \param ref an n*dim-length array of data points.
    * \param query an m*dim-length array of query points.
    * \param n number of reference point
    * \param m number of query points
    * \param dim dimensionality of reference/query points
    * \param dist [out] m*n matrix of the *squared* distances from query points to reference points.
    * \param sqnormr An array of length n used for storing the squared norm of each reference point.
    * If NULL or omitted, allocated and deallocated internally.
    * \param sqnromq An array of length m used for storing the squared norm of each reference point.
    * If NULL or omitted, allocated and deallocated internally.
    * \param useSqnormrInput If true, use precomputed squared norm values contained in sqnormr when 
    * this function is called; if false, compute squared norms internally.
    */
   void compute_distances
              ( double *ref, double *query, long n, long m, int dim, double *dist, 
                  double* sqnormr = NULL, double* sqnormq = NULL, bool useSqnormrInput = false );

   /**
    * Computes the square of the norm of each row of a matrix.
    * \param a The input matrix (row-major).
    * \param n The number of rows of a.
    * \param dim The number of columns of a.
    * \param b [out] The output array (length n).
    */
   void sqnorm ( double *a, long n, int dim, double *b);
 

   /**
    * Find the k-nearest neighbors of each query point using distributed cyclic-shift algorithm.
    * \param ref an nlocal*dim-length array of data points.
    * \param query an mlocal*dim-length array of query points.
    * \param glob_ids An array of length nlocal containing the unique ID of each reference point stored locally.
    * \param nlocal Number of local reference points.
    * \param mlocal Number of local query points.
    * \param k Number of neighbors to find for each query point.
    * \param dim Dimensionality of reference/query points.
    * \param comm The MPI communicator to use.
    * \return An array of pairs containing the distance to each neighbor and its index; 
    * neighbors for point i begin at location i*k.
    * \note All input arrays MUST be allocated with malloc (or equivalent) and deallocated with free;
    * new and delete[] may result in a segmentation fault.
    */
   std::pair<double, long> * dist_directKQuery( double* &ref, double *query, 
   						 long* &glob_ids,
   						 int nlocal, int mlocal,
   						 int k, 
   						 int dim, 
   						 MPI_Comm comm );
	
	
	

   /**
    * Find all reference points lying within distance sqrt(R) of each query point.
    * \param ref an nlocal*dim-length array of data points.
    * \param query an mlocal*dim-length array of query points.
    * \param n Number of global reference points
    * \param m Number of global query points
    * \param nlocal Number of local reference points
    * \param mlocal Number of local query points
    * \param R The *square* of the search radius.
    * \param dim Dimensionality of reference/query points
    * \param glob_ids An array of length nlocal containing the unique ID of each reference point stored locally.
    * \param rneighbors [out] An externally allocated array of mlocal vectors, used to store the neighbors of each 
    * query point.
    * \param comm The MPI communicator to use.
    */
    void dist_directRQuery 
              ( double* &ref, double *query, long n, long m, int nlocal, int mlocal, double R, 
                int dim, long* &glob_ids, std::vector< std::pair<double, long> >* rneighbors, 
                MPI_Comm comm  );
 
   /**
    * Find all reference points lying within distance sqrt(R) of each query point,
    * up to a maximum of max_neighbors (not necessarily the nearest).
    * \param ref an nlocal*dim-length array of data points.
    * \param query an mlocal*dim-length array of query points.
    * \param n Number of global reference points
    * \param m Number of global query points
    * \param nlocal Number of local reference points
    * \param mlocal Number of local query points
    * \param R The *square* of the search radius.
    * \param dim Dimensionality of reference/query points
    * \param glob_ids An array of length nlocal containing the unique ID of each reference point stored locally.
    * \param rneighbors [out] An externally allocated array of mlocal vectors, used to store the neighbors of each 
    * query point.
    * \param max_neighbors The maximum number of neighbors returned for any given query point.
    * \param comm The MPI communicator to use.
    */
    void dist_directRQuery
    ( double* &ref, double *query, long n, long m, int nlocal, int mlocal, double R,
     int dim, long* &glob_ids, std::vector< std::pair<double, long> >* rneighbors, int max_neighbors,
     MPI_Comm comm  );

 
  
  /**
   * Find all reference points lying within distance sqrt(R[i]) of query point i.
   * \param ref an nlocal*dim-length array of data points.
   * \param query an mlocal*dim-length array of query points.
   * \param n Number of global reference points
   * \param m Number of global query points
   * \param nlocal Number of local reference points
   * \param mlocal Number of local query points
   * \param R Array of length m, the *square* of the search radius for each query point.
   * \param dim Dimensionality of reference/query points
   * \param glob_ids An array of length nlocal containing the unique ID of each reference point stored locally.
   * \param rneighbors [out] An externally allocated array of mlocal vectors, used to store the neighbors of each 
   * query point.
   * \param comm The MPI communicator to use.
   */
  void dist_directRQueryIndividual
  ( double* &ref, double *query, long n, long m, int nlocal, int mlocal, double *R, 
   int dim, long* &glob_ids, std::vector< std::pair<double, long> >* rneighbors, 
   MPI_Comm comm  );



  /**
   * Find all reference points lying within distance sqrt(R[i]) of query point i.
   * \param ref an nlocal*dim-length array of data points.
   * \param query an mlocal*dim-length array of query points.
   * \param n Number of global reference points
   * \param m Number of global query points
   * \param nlocal Number of local reference points
   * \param mlocal Number of local query points
   * \param R Array of length m, the *square* of the search radius for each query point.
   * \param dim Dimensionality of reference/query points
   * \param glob_ids An array of length nlocal containing the unique ID of each reference point stored locally.
   * \param rneighbors [out] An externally allocated array of mlocal vectors, used to store the neighbors of each 
   * query point.
   * \param comm The MPI communicator to use.
   */
   void dist_directRQueryIndividualK
( double* &ref, double *query, long n, long m, int nlocal, int mlocal, double *R, int k,
 int dim, long* &glob_ids, std::vector< std::pair<double, long> >* rneighbors,
 MPI_Comm comm  );




  /**
   * Find all reference points lying within distance sqrt(R[i]) of query point i,
   * up to a maximum of max_neighbors points (not necessarily the nearest).
   * \param ref an nlocal*dim-length array of data points.
   * \param query an mlocal*dim-length array of query points.
   * \param n Number of global reference points
   * \param m Number of global query points
   * \param nlocal Number of local reference points
   * \param mlocal Number of local query points
   * \param R Array of length m, the *square* of the search radius for each query point.
   * \param dim Dimensionality of reference/query points
   * \param glob_ids An array of length nlocal containing the unique ID of each reference point stored locally.
   * \param rneighbors [out] An externally allocated array of mlocal vectors, used to store the neighbors of each 
   * query point.
   * \param max_neighbors The maximum number of neighbors to return for any single query point.
   * \param comm The MPI communicator to use.
   */
   void dist_directRQueryIndividual
   ( double* &ref, double *query, long n, long m, int nlocal, int mlocal, double *R,
    int dim, long* &glob_ids, std::vector< std::pair<double, long> >* rneighbors, int max_neighbors,
    MPI_Comm comm  );


   /**
    * Merges a and b and keeps only the k smallest from each of the m sets of distance/index pairs.
    * \param a one array of m*k distance+index pairs
    * \param b the other array of m*k distance+index pairs
    * \param m the number sets of size k
    * \param k k
    * \return A newly allocated array of size m*k
    */
   std::pair<double, long> *kmin_merge( std::pair<double, long> *a, std::pair<double, long> *b, long m, long k );

   /**
    * Reads reference and query points from files in parallel and partitions them for a query with the rectangular algorithm..
    * \param refFile The name of the file which contains the reference points.
    * \param queryFile The name of the file which contains the query points.
    * \param ref [out] an n*dim array of data points.
    * \param query [out] an m*dim array of query points.
    * \param n number of reference points.
    * \param m number of query points.
    * \param dim dimensionality of reference/query points.
    * \param refParts number of parts to split reference points into (refParts * queryParts = processes).
    * \param queryParts number of parts to split query points into (refParts * queryParts = processes).
    * \param comm The MPI communicator to use.
    * \return A pair object containing the number of reference points (first) and query points (second) stored at this process.
    */
   std::pair<int, int> partitionPoints(std::string refFile, std::string queryFile, double *& ref, double *& query, long n, long m, int dim, int refParts, int queryParts, MPI_Comm comm);


   /**
    * Returns the number of local elements in the partitioning used by dist_directKQuery.
    * \param rank This process's MPI rank.
    * \param size The size of the MPI communicator used.
    * \param num The global number of elements.
    * \return The number of elements this process stores locally.
    */
   inline int getNumLocal( int rank, int size, long num ) {
      if( rank < num%size)
         return std::ceil( (double)num / (double)size );
      else
         return num / size;
   }

   /**
    * Returns the offset of the locally stored elements in the global array.
    * \param rank This process's MPI rank.
    * \param size The size of the MPI communicator used.
    * \param num The global number of elements.
    * \return The global array offset.
    */
   inline long getGlobalArrayOffset( int rank, int size, long num ) {
      if( rank < num%size )
         return std::ceil( (double)num / (double)size ) * rank;
      else
         return std::ceil( (double)num / (double)size ) * (num%size) + (rank-(num%size)) * (num/size);
   }

   /**
    * Repartitions points with replication.
    * \param points The points to repartition (modified).
    * \param localPointCount The number of points the process has.
    * \param dim The dimensionality of the points.
    * \param globalIDs The global ID associated with each point (modified).
    * \param partitionCount The number of pieces to partition the points into (the number of replications is implicitly given by size / partitionCount)
    * \param comm The communicator to use.
    * \return The new number of points the process has.
   */
   int directRectRepartition(double *&points, int localPointCount, int dim, int *&globalIDs, int partitionCount, int direction, MPI_Comm comm);

   /**
    * Performs a direct kNN or rNN query using the rectangular-partitioned algorithm, automatically repartitioning points..
    * \param refPts The reference points (modified).
    * \param queryPts The query points (modified).
    * \param localRefCount The number of reference points the process has (modified).
    * \param localQueryCount The number of query points the process has (modified).
    * \param dim The dimensionality of the points.
    * \param refIDs The global IDS of the reference points the process has (modified).
    * \param queryIDs The global IDs of the query points the process has (modified).
    * \param params The parameters of the direct kNN or rNN query.
    * \param comm The communicator to use.
    * \return The result of the query.
   */
   directQueryResults directRectRepartitionAndQuery(double *&refPts, double *&queryPts, long& localRefCount, long& localQueryCount, int dim, int *&refIDs, int *&queryIDs, directQueryParams params, MPI_Comm comm);

   /**
    * Performs a rNN query on a rectangularly partitioned set of points.
    * \param refPoints The reference points.
    * \param queryPoints The query points.
    * \param localRefCount The number of reference points the process has.
    * \param localQueryCount The number of query points the process has.
    * \param dim The dimensionality of the points.
    * \param r The search radius of the query.
    * \param queryComm The communicator to use (all processes within this comm should have the same query points).
    * \return The result of the query.
   */
   directQueryResults directQueryRectR(double *refPoints, double *queryPoints, int localRefCount, int localQueryCount, int dim, double r, MPI_Comm queryComm);

   /**
    * Performs a kNN query on a rectangularly partitioned set of points.
    * \param refPoints The reference points.
    * \param queryPoints The query points.
    * \param localRefCount The number of reference points the process has.
    * \param localQueryCount The number of query points the process has.
    * \param dim The dimensionality of the points.
    * \param k The number of neighbors to find for each query point.
    * \param queryComm The communicator to use (all processes within this comm should have the same query points).
    * \return The result of the query.
   */   
   directQueryResults directQueryRectK(double *refPoints, double *queryPoints, int localRefCount, int localQueryCount, int dim, int k, MPI_Comm queryComm);



   /**
    * Determine the number of query points that will be handled by each iteration of the LowMem queries.
    * \param n The number of reference points.
    * \param m The number of query points.
    * \return The block size the query will use.  Can be multiplied by n to determine size of distance matrix.
    */
   inline int getBlockSize(long n, long m) {
     int blocksize;
     if( m > KNN_MAX_BLOCK_SIZE || n > 10000L) {
       blocksize = std::min((long)KNN_MAX_BLOCK_SIZE, m); //number of query points handled in a given iteration
       if(n * (long)blocksize > KNN_MAX_MATRIX_SIZE) blocksize = std::min((long)(KNN_MAX_MATRIX_SIZE/n), (long)blocksize); //Shrink block size if n is huge.
       blocksize = std::max(blocksize, omp_get_max_threads()); //Make sure each thread has some work.
     } else {
        blocksize = m;
     }
     return blocksize;
   }


}
#endif

