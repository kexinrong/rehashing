#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <omp.h>

#include "parallelIO.h"

using namespace std;

namespace knn
{

bool is_file_exist(const char *filename)
{
    ifstream infile(filename);
    bool is_exist = infile;
    infile.close();
    return is_exist;
}


int line_count(const char *filename)
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


bool dir_traverse(const char *ptrdir, vector<string> &filelist)
{
	struct dirent *entry = NULL;
	DIR *pdir = opendir(ptrdir);
	struct stat buf;
	while( NULL != (entry = readdir(pdir)) ) {
		char pPath[1024] = {0};
		sprintf(pPath, "%s/%s", ptrdir, entry->d_name);
		string strFile = pPath;
		if( 0 != stat(pPath, &buf) ) {
			std::cerr<<"cannot stat\n";
			return false;
		}
		else if( S_ISDIR(buf.st_mode) ) {
			if('.' != entry->d_name[0])
				dir_traverse(pPath, filelist);
		}
		else {
			if('.' != entry->d_name[0])
				filelist.push_back(strFile);
		}
	}
	closedir(pdir);

	return true;
}


bool dir_traverse(const char *ptrdir, const char *format, vector<string> &filelist)
{
	struct dirent * entry = NULL;
	DIR * pdir = opendir(ptrdir);
	struct stat buf;
	while( NULL != (entry = readdir(pdir)) ) {
		char pPath[1024] = {0};
		sprintf(pPath, "%s/%s", ptrdir, entry->d_name);
		string strFile = pPath;
		if( 0 != stat(pPath, &buf) ) {
			std::cerr<<"cannot stat\n";
			return false;
		}
		else if( S_ISDIR(buf.st_mode) ) {
			if('.' != entry->d_name[0])
				dir_traverse(pPath, format, filelist);
		}
		else {
			//size_t found_pos = strFile.rfind(format);
			size_t found_pos = strFile.find_last_of(".");
			string strFormat = strFile.substr(found_pos);
			if('.' != entry->d_name[0]&& 0 == strFormat.compare(format))
				filelist.push_back(strFile);
		}
	}
	closedir(pdir);

	return true;
}


bool dir_ls(const char *ptrdir, vector<string> &filelist)
{
	bool success = dir_traverse(ptrdir, filelist);
	if(success) 
		sort(filelist.begin(), filelist.end());
	return success;
}


bool dir_ls(const char *ptrdir, const char *format, vector<string> &filelist)
{
	bool success = dir_traverse(ptrdir, format, filelist);
	if(success)
		sort(filelist.begin(), filelist.end());
	return success;
}


int home_rank(long glb_numof_points, int nproc, long gid)
{
    long lower_bound = glb_numof_points / nproc;
    int nlarge = glb_numof_points % nproc;
    long upper_bound = (nlarge == 0 ? lower_bound : lower_bound+1);
    long large_gid = upper_bound*(long)nlarge;
    if(gid < large_gid) {
        return gid / upper_bound;
    }
    else {
        return (nlarge + (gid-large_gid) / lower_bound);
    }
}


void mpi_dlmwrite_knn(const char *filename,
                      vector<long> &queryIDs_rkdt,
                      vector< pair<double, long> > *kNN_rkdt,
                      MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int k = kNN_rkdt->size() / queryIDs_rkdt.size();
    if(0 == rank) remove(filename);
    MPI_Barrier(comm);

    ofstream output;
    for(int r = 0; r < size; r++) {
        if(rank == r) {
            output.open(filename, ios::app|ios::out);
            for(int i = 0; i < queryIDs_rkdt.size(); i++) {
                output<<queryIDs_rkdt[i]<<" ";
                for(int j = 0; j < k; j++) {
                    output<<(*kNN_rkdt)[i*k+j].first<<" "<<(*kNN_rkdt)[i*k+j].second<<" ";
                }
                output<<endl;
            }
            output.flush();
            output.close();
        }
        MPI_Barrier(comm);
    }
}


void dlmread_knn(const char *filename, const vector<long> &queryIDs, int k,
                 vector< pair<double, long> > *kNN)
{
    kNN->resize(queryIDs.size()*k);

    ifstream infile;
    infile.open(filename);
    long qgid, gid;
    double dist;
    for(int i = 0; i < queryIDs.size(); i++) {
        do {
            string strLine, strValue;
            getline(infile, strLine);
            istringstream ssline(strLine);
            ssline >> qgid;
            if(qgid == queryIDs[i]) {
                for(int j = 0; j < k; j++) {
                    ssline >> dist;
                    ssline >> gid;
                    (*kNN)[i*k+j].first = dist;
                    (*kNN)[i*k+j].second = gid;
                 }
            }
        } while(qgid != queryIDs[i]);
    }
    infile.close();

}


void mpi_binwrite_knn(const char *filename,
                      vector<long> &queryIDs_rkdt,
                      vector< pair<double, long> > *kNN_rkdt,
                      MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    long k = (long)kNN_rkdt->size() / (long)queryIDs_rkdt.size();

    double start_t = omp_get_wtime();

    ofstream output;
    if(0 == rank) {
        remove(filename);
        output.open(filename, ios::binary|ios::app);
        output.write((char*)(&k), sizeof(int));
        output.close();
    }
    MPI_Barrier(comm);


    // each time, write 20,000 query in the same time, 20k*1024*2*8 ~ 320M, not too much
    long grain_write_size = 20000;
    long nfold = (long)ceil((double)queryIDs_rkdt.size() / (double)grain_write_size);
    double *buff = new double [grain_write_size*(2*k+1)];

    for(int r = 0; r < size; r++) {
        if(rank == r) {
            cout<<"rank "<<rank<<": write "<<k<<" nn of "<<queryIDs_rkdt.size()<<" points ... ";
            output.open(filename, ios::binary|ios::app);
            // each time, write 10,000 query in the same time, 10k*1024*2*8 ~ 2M, not too much
            for(long i = 0; i < nfold; i++) {
                long np = (i == nfold-1) ? queryIDs_rkdt.size()-i*grain_write_size : grain_write_size;
                for(long j = 0; j < np; j++) {
                    long idx = i*grain_write_size+j;
                    buff[j*(2*k+1)+0] = (double)queryIDs_rkdt[idx];
                    for(long t = 0; t < k; t++) {
                        buff[j*(2*k+1)+1+2*t+0] = (*kNN_rkdt)[idx*k+t].first;
                        buff[j*(2*k+1)+1+2*t+1] = (double)(*kNN_rkdt)[idx*k+t].second;
                    }
                }
                output.write((char *)(buff), np*(2*k+1)*sizeof(double));
            }
            /*
            for(int i = 0; i < queryIDs_rkdt.size(); i++) {
                long qgid = queryIDs_rkdt[i];
                output.write((char*)(&qgid), sizeof(long));
                //output<<queryIDs_rkdt[i]<<" ";
                for(int j = 0; j < k; j++) {
                    double dist = (*kNN_rkdt)[i*k+j].first;
                    long gid = (*kNN_rkdt)[i*k+j].second;
                    output.write((char *)(&dist), sizeof(double));
                    output.write((char *)(&gid), sizeof(long));
                    //output<<(*kNN_rkdt)[i*k+j].second<<" "<<(*kNN_rkdt)[i*k+j].first<<" ";
                }
            }*/
            output.close();
            cout<<"done "<<omp_get_wtime()-start_t<<endl;
        }
        MPI_Barrier(comm);
    }
}


bool binread_knn(const char *filename, const vector<long> &queryIDs, int k,
                 vector< pair<double, long> > *kNN)
{
    bool success = true;
    kNN->resize((long)queryIDs.size()*(long)k);

    int BigK = -1;
    ifstream infile;
    infile.open(filename, ifstream::binary);
    infile.read((char*)(&BigK), sizeof(int));
    infile.close();

    long unit_offset = sizeof(long) + (long)BigK*(sizeof(long)+sizeof(double));

    #pragma omp parallel
    {
        ifstream infile2;
        infile2.open(filename, ifstream::binary);
        #pragma omp for
        for(long i = 0; i < queryIDs.size(); i++) {
            long byte_offset = sizeof(int) + queryIDs[i]*unit_offset;
            infile2.seekg(byte_offset, ifstream::beg);
            long qgid, gid;
            double dummy_gid, dummy_qgid, dist;
            infile2.read( (char*)(&dummy_qgid), sizeof(double) );
            qgid = (long)dummy_qgid;
            if(qgid != queryIDs[i]) {
                cerr<<"error: missmatch found for query gid = "<<queryIDs[i]
                    <<", the gid I read is "<<qgid<<endl;
                success = false;
                //return false;
            }
            for(long j = 0; j < k; j++) {
                infile2.read( (char*)(&dist), sizeof(double) );
                infile2.read( (char*)(&dummy_gid), sizeof(double) );
                gid = (long)dummy_gid;
                (*kNN)[i*(long)k+j].first = dist;
                (*kNN)[i*(long)k+j].second = dummy_gid;
            }
        }
        infile2.close();
    }

    return success;
}


bool binread_knn(const char *filename, long first_gid, int n, int k,
                 vector< pair<double, long> > *kNN)
{
    bool success = true;

    assert(sizeof(double) == sizeof(long) );

    kNN->resize((long)n*(long)k);

    int BigK = -1;
    ifstream infile;
    infile.open(filename, ifstream::binary);
    infile.read((char*)(&BigK), sizeof(int));

    long buff_size = (long)n*(2*(long)BigK+1);
    double *buff = new double [buff_size];
    //cout<<"BigK = "<<BigK<<", n = "<<n<<", buff size = "<<buff_size*(long)sizeof(double)/1e9<<endl;

    long unit_offset = sizeof(long) + (long)BigK*(sizeof(long)+sizeof(double));
    long byte_offset = sizeof(int) + first_gid*unit_offset;
    infile.seekg(byte_offset, ifstream::beg);
    infile.read( (char*)(&(buff[0])), buff_size*sizeof(double) );
    infile.close();

    #pragma omp parallel for
    for(long i = 0; i < n; i++) {
        long idx = i*(2*(long)BigK+1);
        long gid = buff[idx+0];
	if(i%5000 == 0) {
	  cout<<"read "<<gid<<" knn done"<<endl;
	}
        if( gid != (first_gid+(long)i) ) {
            cerr<<"error: missmatch found for query gid = "<<first_gid+(long)i
                <<", the gid I read is "<<gid<<endl;
            //return false;
            success = false;
        }
        for(long j = 0; j < k; j++) {
            (*kNN)[i*(long)k+j].first = buff[idx+1+2*j+0];
            (*kNN)[i*(long)k+j].second = buff[idx+1+2*j+1];
        }
    }

    delete [] buff;
    return false;
}


}

