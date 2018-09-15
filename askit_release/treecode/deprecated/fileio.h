#ifndef __FILEIO_H__
#define __FILEIO_H__

#include <mpi.h>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <cstring>
#include <vector>

using namespace std;

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
 * (dlmread) Read ascii file of numeric data into matrix. 
 * The output matrix is stored in a row-wise 1d array, 
 * and the delimiter in the given file havs to be a space character ' '.
 * \param filename Input data filename
 * \param numof_rows Number of rows of the 2d matrix, output, automatically determined when given an input file
 * \param numof_cols Number of columns of the 2d matrix, output, automatically determined when given an input file
 * \param arr Output array storing the input 2d matrix
 * \return If read successfully, return TURE, otherwise return FALSE
 */
template <typename T>
bool dlmread(const char *filename, int &numof_rows, int &numof_cols, vector<T> &arr);


/**
 * (dlmwrite) Write a numeric matrix into an ascii file
 * \param filename Filename of the written file
 * \param numof_rows Number of rows of the input 2d matrix
 * \param numof_cols Number of columns of the input 2d matrix
 * \param arr Input array
 * \return If write successfully, return TURE, otherwise return FALSE
 */
template <typename T>
bool dlmwrite(const char *filename, int numof_rows, int numof_cols, const vector<T> &arr);


/**
 * (mpi_dlmwrite) Write given matrix one by one from rank 0 to rank size-1
 * \param filename Filename of the written file
 * \param pArr Array on its own rank
 * \param numof_rows Number of rows of the array on its own processor
 * \param numof_cols Number of rows of the array on its own processor
 * \param comm MPI communicator
 * \return If write successfully, return TURE, otherwise return FALSE
 */
template <typename T>
bool mpi_dlmwrite(const char *filename, const T *pArr, const int &numof_rows, const int &numof_cols, MPI_Comm comm);

#include "fileio.txx"

#endif




