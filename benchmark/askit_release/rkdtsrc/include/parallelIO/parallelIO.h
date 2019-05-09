#ifndef _PARALLEL_IO_HPP_
#define _PARALLEL_IO_HPP_
#include <mpi.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <vector>

using namespace std;

namespace knn
{

/**
 * (split_filename) Split input full filename into two parts: the direcotry (path) and the filename
 * \param strFullFilename Input full filename (path/file)
 * \param strDir Input Path
 * \param strFile Input file
 */
inline void split_filename(const string &strFullFilename, string &strDir, string &strFile)
{
	size_t found=strFullFilename.find_last_of("/\\");
	strFile = strFullFilename.substr(found+1);
	strDir = strFullFilename.substr(0, found);
}


/**
 * (itoa) Convert input integer to string
 * \param value Input integer
 * \return Converted string
 */
inline string itoa(int value)
{
	ostringstream o;
	if(!(o << value)) 
		return "";
	return o.str();
}

/**
 * find file exist or not
 * \param filename File to check
 * \return true or false
 */
bool is_file_exist(const char *filename);


/**
 * (line_count) Count how many lines in a given file
 * \param filename Input file
 * \return Number of lines in the input file
 */
int line_count(const char * filename);


/**
 * (dir_traverse) Traverse a given directory to find all files in it or its subdirectories.
 * It is possible to specify a special format as well
 * !!! The filename in the filelist is unordered
 * \param ptrdir Input directory
 * \param filelist Output filelist storing all files
 * \param format Special format of files to search for, e.g. '.png'
 * \return If success, return TURE, otherwise return FALSE
 */
bool dir_traverse(const char *ptrdir, vector<string> &filelist);
bool dir_traverse(const char *ptrdir, const char *format, vector<string> &filelist);


/**
 * (dir_ls) Traverse a given directory to find all files in it or its subdirectories.
 * It is possible to specify a special format as well
 * !!! The filename in the filelist is in an ascending order
 * \param ptrdir Input directory
 * \param filelist Output filelist storing all files
 * \param format Special format of files to search for, e.g. '.png'
 * \return If success, return TURE, otherwise return FALSE
 */
bool dir_ls(const char *ptrdir, vector<string> &filelist);
bool dir_ls(const char *ptrdir, const char *format, vector<string> &filelist);


/**
 * (home_rank) Calculate the home rank given the global id of points
 * *IFF* the data are read by the provided IO functions (mpi_dlm/binread);
 * \param glb_numof_points Global number of points across all processors
 * \param nproc Number of processors in the communicator
 * \param gid Global id of input point
 * \return Rank of its home process
 */
int home_rank(long glb_numof_points, int nproc, long gid);



/**
 * (dlmread) Read ascii file of numeric data into matrix
 *
 * ascii format: each row is one point of dim, the delimiter is a space character ' '
 *
 * \param filename Filename of the input file
 * \param numof_points (input/output)
 *      flag_read_all = true: (output) real number of points read into arr
 *      flag_read_all = false: (1st as input) number of points to read from the input file
 *                             (2nd as output) real number of points read into arr
 * \param dim (output value) dimensionality of data
 * \param points double array store point coordinates
 * \param flag_read_all true: read all data in the input binary file
 *                      false: real numof_points in the input binary
 * \return If read successfully, return TURE, otherwise return FALSE
 */
template <typename T>
bool dlmread(const char *filename, int &numof_points, int &dim, vector<T> &arr, bool flag_read_all = true);
template <typename T>
bool dlmread(const char *filename, int &numof_points, int &dim, T *&arr, bool flag_read_all = true);



/**
 * (mpi_dlmread) Read ascii file evenly into each rank one by one
 *
 * ascii format: each row is one point of dim, the delimiter is a space character ' '
 *
 * \param filename Filename of the input file
 * \param numof_points (input/output)
 *      flag_read_all = true: (output) local number of points read on its own processor
 *      flag_read_all = false: (1st as input) global number of points to read across all processors,
 *                             (2nd as output) local number of points read on its own processor
 * \param dim (output) dimensionality of data
 * \param points double array store point coordinates
 * \param comm MPI communicator
 * \param flag_read_all true: read all data in the input binary file
 *                      false: real numof_points in the input binary
 * \return If read successfully, return TURE, otherwise return FALSE
 */
template <typename T>
bool mpi_dlmread(const char *filename, long &numof_points, int &dim, vector<T> &arr, MPI_Comm comm, bool flag_real_all = true);
template <typename T>
bool mpi_dlmread(const char *filename, long &numof_points, int &dim, T *&arr, MPI_Comm comm, bool flag_real_all = true);



/**
 * (binread) Read binary file on its own processor
 *
 * binary format: data coordinates one by one, points concatenation (double)
 *
 * \param filename Filename of the input file
 * \param numof_points (input) number of points to read
 * \param dim (input) dimensionality of data
 * \param points double array store point coordinates
 * \return If read successfully, return TURE, otherwise return FALSE
 */
template <typename T>
bool binread(const char *filename, int numof_points, int dim, vector<T> &points);
template <typename T>
bool binread(const char *filename, int numof_points, int dim, T *&points);



/**
 * (mpi_binread) Read binary file evenly into each rank one by one
 *
 * binary format: data coordinates one by one (double)
 *
 * \param filename Filename of the input file
 * \param glb_numof_points (input) total number of points across all processor to read (global number of points)
 * \param dim (input) dimensionality of data
 * \param numof_points (output) number of points loaded on its own processor (local number of points)
 * \param points double array store point coordinates
 * \param comm MPI communicator
 * \return If read successfully, return TURE, otherwise return FALSE
 */
template <typename T>
bool mpi_binread(const char *filename, long glb_numof_points, int dim, int &numof_points, vector<T> &points, MPI_Comm comm);
template <typename T>
bool mpi_binread(const char *filename, long glb_numof_points, int dim, int &numof_points, T *&points, MPI_Comm comm);


/**
 * (dlmwrite) Write a numeric matrix into an ascii file, each row is one point,
 * the delimiter is a space character ' '
 * \param filename Filename of the written file
 * \param numof_points Number of points (rows) in the given array
 * \param dim Dimensionality of points (numof cols) in the given array
 * \param arr Input array
 * \return If write successfully, return TURE, otherwise return FALSE
 */
template <typename T>
bool dlmwrite(const char *filename, int numof_points, int dim, const T *arr);


/**
 * (binwrite) Write binary file
 * binary format: data coordinates one by one (double)
 * \param filename Filename of the written file
 * \param numof_points numof_points to write
 * \param dim dimensionality of data
 * \param points double array store point coordinates
 * \return If read successfully, return TURE, otherwise return FALSE
 */
template <typename T>
bool binwrite(const char *filename, int numof_points, int dim, const T *points);


/**
 * (mpi_dlmwrite) Write given points one by one from rank 0 to rank size-1,
 * each row is one point, delimiter is a space character ' '
 * \param filename Filename of the written file
 * \param numof_points Local number of points (rows) in the given array
 * \param dim Dimensionality of points (numof cols) in the given array
 * \param arr Input array
 * \param comm MPI communicator
 * \return If write successfully, return TURE, otherwise return FALSE
 */
template <typename T>
bool mpi_dlmwrite(const char *filename, int numof_points, int dim, const T *arr, MPI_Comm comm);



template <typename T>
bool mpi_binwrite(const char *filename, int numof_points, int dim, const T *arr, MPI_Comm comm);




/**
 * (mpi_dlmwrite_knn) Write nearest neighbors result into ASCII file
 * one by one from rank 0 to rank size-1,
 * each row is the NN for one query point, delimiter is a space character ' '
 * ascii format: query_global_id, 1st_nn_dist, 1st_nn_gid, 2nd_nn_dist, 2nd_nn_dist, ...
 * \param filename Filename of the written file
 * \param queryIDs_rkdt Array of query ids
 * \param kNN_rkdt Array of knn results
 * \param comm MPI communicator
 */
void mpi_dlmwrite_knn(const char *filename,
                      vector<long> &queryIDs_rkdt, vector< pair<double, long> > *kNN_rkdt,
                      MPI_Comm comm);


/**
 * (mpi_binwrite_knn) Write nearest neighbors result into BINARY file
 * one by one from rank 0 to rank size-1,
 * binary format: **BigK**, query_global_id, 1st_nn_dist, 1st_nn_gid, 2nd_nn_dist, 2nd_nn_dist, ...
 * the first element in the output file is **BigK**, i.e., the number of NN for each query.
 * \param filename Filename of the written file
 * \param queryIDs_rkdt Array of query ids
 * \param kNN_rkdt Array of knn results
 * \param comm MPI communicator
 */
void mpi_binwrite_knn(const char *filename,
                      vector<long> &queryIDs_rkdt, vector< pair<double, long> > *kNN_rkdt,
                      MPI_Comm comm);


/**
 * (binread_knn) Read nearest neighbors for given queries from the input BINARY file
 * binary format: **BigK**, query_global_id, 1st_nn_dist, 1st_nn_gid, 2nd_nn_gid, 2nd_nn_dist, ...
 * the first element in the output file is **BigK**, i.e., the number of NN for each query.
 * \param filename Filename of the read file
 * \param queryIDs Array of query global ids which need to read their NNs
 * \param k Number of NN want to read, do not exceed 'BigK', the largest k can be retrieved.
 * \param kNN Array of knn results
 */
bool binread_knn(const char *filename, const vector<long> &queryIDs, int k,
                 vector< pair<double, long> > *kNN);

/**
 * (binread_knn) Read nearest neighbors for contiguous points from the input BINARY file
 * binary format: **BigK**, query_global_id, 1st_nn_dist, 1st_nn_gid, 2nd_nn_gid, 2nd_nn_dist, ...
 * the first element in the output file is **BigK**, i.e., the number of NN for each query.
 * \param filename Filename of the read file
 * \param queryIDs Array of query global ids which need to read their NNs
 * \param k Number of NN want to read, do not exceed 'BigK', the largest k can be retrieved.
 * \param kNN Array of knn results
 */
bool binread_knn(const char *filename, long first_gid, int n, int k,
                 vector< pair<double, long> > *kNN);



/**
 * (dlmread_knn) Read nearest neighbors for given queries from the input ASCII file
 * ascii format: query_global_id, 1st_nn_dist, 1st_nn_gid, 2nd_nn_gid, 2nd_nn_dist, ...
 * \param filename Filename of the read file
 * \param queryIDs Array of query global ids which need to read their NNs
 * \param k Number of NN want to read
 * \param kNN Array of knn results
 */
void dlmread_knn(const char *filename, const vector<long> &queryIDs, int k,
                 vector< pair<double, long> > *kNN);

#include "parallelIO.txx"


}

#endif // _PARALLEL_IO_HPP_
