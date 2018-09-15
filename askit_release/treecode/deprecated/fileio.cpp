#include <mpi.h>

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

#include "fileio.h"

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






