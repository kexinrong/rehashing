#include <mpi.h>
#include <iostream>
#include <fstream>
#include <cctype>
#include <omp.h>
#include <string>
#include <climits>

#include "CmdLine.h"
#include "parallelIO.h"

using namespace Torch;
using namespace std;

int main(int argc, char **argv) {

	// command lines
	CmdLine cmd;
	const char *pchHelp = "convert ascii to binary, each row is a datum in ascii";
	cmd.addInfo(pchHelp);

	char *ptrInputFile = NULL;
	cmd.addSCmdOption("-infile", &ptrInputFile, "data.ascii", "input (ascii) data file");
	char *ptrOutputFile = NULL;
	cmd.addSCmdOption("-outfile", &ptrOutputFile, "data.bin", "output (binary) data file");

    int numof_points;
	cmd.addICmdOption("-n", &numof_points, 1, "total number of points to convert");

	cmd.read(argc,argv);
	ifstream ascii_infile;
	ascii_infile.open(ptrInputFile);
    if(!ascii_infile.is_open()) {
		std::cout<<"cannot open file "<<ptrInputFile<<std::endl;
		return false;
	}

	// determine number of rows and cols
	int total_numof_points = 1;
	int dim = 0;
	string strLine, strele;
	getline(ascii_infile, strLine);
	istringstream szline(strLine);
	while( szline >> strele ) dim++;
	while( getline(ascii_infile, strLine) ) {
        total_numof_points++;
        if(total_numof_points % 50000 == 0) cout<<"read "<<total_numof_points<<" lines"<<endl;
    }
	ascii_infile.close();
    if(numof_points > total_numof_points) {
        cout<<"warning: no enough data in "<<ptrInputFile
            <<", read all available "<<total_numof_points<<" points"<<endl;
        numof_points = total_numof_points;
    }

    cout << "Data set has " << numof_points << " points with " << dim << " features.\n";

    if(knn::is_file_exist(ptrOutputFile)) remove(ptrOutputFile);
    ofstream bin_outfile;
    bin_outfile.open(ptrOutputFile, ios::binary|ios::app);
	ascii_infile.open(ptrInputFile);
    long N = (long)dim * (long)numof_points;
    for(long i = 0; i < N; i++) {
		double buff;
        ascii_infile >> buff;
        bin_outfile.write((char*)(&buff), sizeof(buff));
    }
    ascii_infile.close();
    bin_outfile.close();

    return 0;
}



