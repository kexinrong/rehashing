#ifndef __FILE_IO_H__
#define __FILE_IO_H__

#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string>

using std::ofstream;
using std::string;
using std::ifstream;
using std::ios;
using std::istringstream;

bool dirTraverse(const char * ptrDirPath, ofstream & fwList);
void SplitFilename (const string& strFilename, string & strFile);
void SplitFilename (const string& strFilename, string & strFile, string & strDir);
int LineCount(const char * filename);
bool readdir(const char * ptrdir, const char * ptrfile);

double * file_read(const char * filename, long * numPoints, int *dim);
void mpi_read_bcast(const char *filename, long *numPoints, int *dim, 
			double **points, long **ids, MPI_Comm comm);
int mpi_kmeans_write(const char *filename, int numClusters, int numPoints, int dim, 
		double *clusters, int *membership, int totalNumPoints, MPI_Comm comm);

#endif
