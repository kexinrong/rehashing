#include <mpi.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <math.h>
#include <string.h>


#include "file_io.h"

using namespace std;

/* ----------------- functions used to traverse a directory -------------- */
bool dirTraverse(const char * ptrDirPath, ofstream & fwList)
{
	struct dirent * entry = NULL;
	DIR * pdir = opendir(ptrDirPath);
	struct stat buf;
	while( NULL != (entry = readdir(pdir)) ) {
		char pPath[1024] = {0};
		sprintf(pPath, "%s/%s", ptrDirPath, entry->d_name);
		if( 0 != stat(pPath, &buf) ) {
			std::cerr<<"cannot stat\n";
			return false;
		}
		else if( S_ISDIR(buf.st_mode) ) {
			//std::cout<<entry->d_name<<" is a directory\n";
			if('.' != entry->d_name[0])
				dirTraverse(pPath, fwList);
		}
		else {
			//std::cout<<entry->d_name<<" is a file\n";
			//std::cout<<pPath<<std::endl;
			if('.' != entry->d_name[0])
				fwList<<pPath<<std::endl;
		}
	}

	closedir(pdir);
	return true;
}

bool readdir(const char * ptrdir, const char * ptrfile)
{
	ofstream fwOutput(ptrfile);
	if(!fwOutput.is_open()) {
		std::cerr<<"cannot open training list"<<ptrfile<<std::endl;
		return false;
	}
	dirTraverse(ptrdir, fwOutput);
	fwOutput.close();

	ifstream frInfile(ptrfile);
	vector<string> filepath;
	string strline;
	while(getline(frInfile, strline)) {
		filepath.push_back(strline);
	}
	frInfile.close();
	sort(filepath.begin(), filepath.end());

	ofstream fwOutfile(ptrfile);
	for(vector<string>::iterator it = filepath.begin(); it < filepath.end(); it++) {
		fwOutfile<<*it<<std::endl;
	}
	fwOutfile.close();
	return true;
}


void SplitFilename (const string& strFilename, string & strFile)
{
	size_t found;
	//std::cout << "Splitting: " << str << endl;
	found=strFilename.find_last_of("/\\");
	//cout << " folder: " << strFilename.substr(0,found) << endl;
	//cout << " file: " << strFilename.substr(found+1) << endl;
	strFile = strFilename.substr(found+1);
}


void SplitFilename (const string& strFilename, string & strFile, string &  strDir)
{
	size_t found;
	//std::cout << "Splitting: " << str << endl;
	found=strFilename.find_last_of("/\\");
	//cout << " folder: " << strFilename.substr(0,found) << endl;
	//cout << " file: " << strFilename.substr(found+1) << endl;
	strFile = strFilename.substr(found+1);
	strDir = strFilename.substr(0, found);
}


int LineCount(const char * filename)
{
	int n = 0;
	ifstream infile(filename);
	if(!infile.is_open()) {
		std::cerr<<"cannot open file "<<filename<<std::endl;
		return -1;
	}
	string strline;
	while ( getline(infile, strline) )
		n++;
	
	infile.close();

	return n;
}


/* --------------- functions used to read/write matrix into files ------------*/
double * file_read(const char * filename, long * numPoints, int *dim)
{
	// try to open the input file
	ifstream InputFile;
	InputFile.open(filename);
	if(!InputFile.is_open()) {
		std::cout<<"cannot open file "<<filename<<std::endl;
		return false;
	}
	// determine numPoints and dim
	(* numPoints) = 1;
	(* dim) = 0;
	string strLine, strele;
	getline(InputFile, strLine);
	istringstream szline(strLine);
	while( szline >> strele ) {
		(*dim)++;
	}
	while( getline(InputFile, strLine) ) {
		(*numPoints)++;
	}
	InputFile.close();

	
	// allocate space for points[][] and read all data
	long len = 0;
	len = (*numPoints)*(*dim);
	double *points = NULL;
	points = new double [len]; 

	long m = 0, n = 0;
	InputFile.open(filename);
	while( getline(InputFile, strLine) ) {
		n = 0;
		istringstream sz_line(strLine);
		while( sz_line >> strele ) {
			points[m*(*dim)+n] = atof(strele.c_str());
			n++;
		}
		m++;
	}
	InputFile.close();
	
	return points;
}


/*------------------------------ mpi_read_bcast() ----------------------- */
void mpi_read_bcast(const char *filename, 		// in
		    long *numPoints, int *dim, 		// out
		    double **points, long **ids,	// out
		    MPI_Comm comm)
{
    	long       divd, rem;
    	int        rank, nproc;
    	MPI_Status status;
    	MPI_Comm_rank(comm, &rank);
    	MPI_Comm_size(comm, &nproc);

    	// ASCII format: let proc 0 read and distribute to others
	if (rank == 0) {
		*points = file_read(filename, numPoints, dim);
        	if (*points == NULL) {
			*numPoints = -1;
		}
		else {
			*ids = new long [*numPoints];
		  #pragma omp parallel for
			for(long i = 0; i < (*numPoints); i++)
				*(*ids+i) = i;
		}
        }

        // broadcast global numPoints and dim to the rest proc
        MPI_Bcast(numPoints, 1, MPI_LONG, 0, comm);
        MPI_Bcast(dim, 1, MPI_INT, 0, comm);

        if (*numPoints == -1) {
            MPI_Finalize();
            exit(1);
        }

        divd = (*numPoints) / nproc;
        rem  = (*numPoints) % nproc;

        if (rank == 0) {
            long index = (rem > 0) ? divd+1 : divd;
            (*numPoints) = index;

            for (int i=1; i<nproc; i++) {
                int msg_size = (i < rem) ? (divd+1) : divd;
                MPI_Send(*points+index*(*dim), msg_size*(*dim), MPI_DOUBLE, i, i, comm);
                MPI_Send(*ids+index, msg_size, MPI_LONG, i, i, comm);
		index += msg_size;
            }

            	// reduce the points[] to local size
            	*points = (double *)realloc(*points, (*numPoints)*(*dim)*sizeof(double));
            	assert(*points != NULL);
        }
        else {
            	//  local numPoints
            	(*numPoints) = (rank < rem) ? divd+1 : divd;

            	// allocate space for data points
            	*points = new double [(*numPoints)*(*dim)];
		*ids = new long [(*numPoints)];	
            	MPI_Recv(*points, (*numPoints)*(*dim), MPI_DOUBLE, 0, rank, comm, &status);
       		MPI_Recv(*ids, (*numPoints), MPI_LONG, 0, rank, comm, &status);
	}

}


int mpi_kmeans_write(const char *filename, int numClusters, int numPoints, int dim, 
		double *clusters, int *membership, int totalNumPoints, MPI_Comm comm)
{
	int        err;
    	int        i, j, k, rank, nproc;
    	char       outFileName[1024], str[32], newline[32];
	MPI_File   fh;
    	MPI_Status status;

    	MPI_Comm_rank(comm, &rank);
    	MPI_Comm_size(comm, &nproc);

	sprintf(newline, "\n");
    	// output: the coordinates of the cluster centres
    	// only proc 0 do this, because clusters[] are the same across all proc 
    	if (rank == 0) {
        	sprintf(outFileName, "%s.cluster_centers", filename);
        	err = MPI_File_open(MPI_COMM_SELF, outFileName, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
        	if (err != MPI_SUCCESS) {
            		char errstr[MPI_MAX_ERROR_STRING];
            		int  errlen;
            		MPI_Error_string(err, errstr, &errlen);
            		printf("Error at opening file %s (%s)\n", outFileName,errstr);
            		MPI_Finalize();
            		exit(1);
        	}
		for (i=0; i<numClusters; i++) {
                	sprintf(str, "%d ", i);
                	MPI_File_write(fh, str, strlen(str), MPI_CHAR, &status);
                	for (j=0; j<dim; j++) {
                    		sprintf(str, "%f ", clusters[i*dim+j]);
                    		MPI_File_write(fh, str, strlen(str), MPI_CHAR, &status);
                	}
               	 	MPI_File_write(fh, newline, 1, MPI_CHAR, &status);
        	}
        	MPI_File_close(&fh);
    	}

    	// output: the closest cluster centre to each of the data points
    	
 	if (rank == 0) { // gather membership[] from all processes
            	int divd = totalNumPoints / nproc;
            	int rem  = totalNumPoints % nproc;

            	sprintf(outFileName, "%s.membership", filename);
            	err = MPI_File_open(MPI_COMM_SELF, outFileName, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
            	if (err != MPI_SUCCESS) {
                	char errstr[MPI_MAX_ERROR_STRING];
                	int  errlen;
                	MPI_Error_string(err, errstr, &errlen);
                	printf("Error at opening file %s (%s)\n", outFileName,errstr);
                	MPI_Finalize();
                	exit(1);
            	}

            	// first, print out local membership[]
            	for (j=0; j<numPoints; j++) {
                	sprintf(str, "%d %d\n", j, membership[j]);
                	MPI_File_write(fh, str, strlen(str), MPI_CHAR, &status);
            	}

            	k = numPoints;
            	for (i=1; i<nproc; i++) {
                	numPoints = (i < rem) ? divd+1 : divd;
                	MPI_Recv(membership, numPoints, MPI_INT, i, i, comm, &status);

                	for (j=0; j<numPoints; j++) {
                    		sprintf(str, "%d %d\n", k++, membership[j]);
                    		MPI_File_write(fh, str, strlen(str), MPI_CHAR, &status);
                	}
            	} // for (i = 1: nproc)
            	MPI_File_close(&fh);
       	 } // if (rank == 0)
       	else {
            MPI_Send(membership, numPoints, MPI_INT, 0, rank, comm);
        }
    
    	return 1;
}


