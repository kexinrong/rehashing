#include<direct_knn.h>
#include<knnreduce.h>
#include<lsh.h>
#include<generator.h>
#include<vector>
#include<cassert>
#include<cmath>
#include<utility>
#include<omp.h>
#include<mpi.h>
#include<CmdLine.h>
#include <ompUtils.h>
#include <ctime>

#include "mpitree.h"
#include "clustering.h"
#include "repartition.h"
#include "binTree.h"
#include "binQuery.h"
#include "distributeToLeaf.h"
#include "parallelIO.h"
#include "eval.h"
#include "papi_perf.h"
#include "verbose.h"
#include "forest.h"

using namespace Torch;
using namespace std;


int main(int argc, char **argv) {

	CmdLine cmd;
	const char *phelp = "Help";
    cmd.addInfo(phelp);

	char *ptrInputFile = NULL;
    cmd.addSCmdOption("-file", &ptrInputFile, "data.bin", "input binary file storing points");
    
	int dim;
	cmd.addICmdOption("-d", &dim, 4, "Dimension of the space (default = 4).");
    
	int rn;
	cmd.addICmdOption("-rn", &rn, 1000, "Number of referrence data points per process (default = 1000).");
 
	int start_id;
	cmd.addICmdOption("-sid", &start_id, 0, "start id (default = 0).");
	
	cmd.read(argc, argv);

    int numof_points = rn;
    long glb_numof_points;
    ifstream infile;
    infile.open(ptrInputFile, ifstream::binary);
    infile.read((char*)&glb_numof_points, sizeof(long));
    infile.seekg(sizeof(long), ifstream::beg);
    infile.read((char*)&dim, sizeof(int));

    if(start_id+numof_points > glb_numof_points) {
        cout<<"warning: no enough data points, read available "
            <<glb_numof_points-start_id<<" points"<<endl;
        numof_points = glb_numof_points-start_id;
    }

    double *points = new double[numof_points * dim];
    long byte_offset = start_id*dim*sizeof(double) + sizeof(long) + sizeof(int);
    infile.seekg(byte_offset, ifstream::beg);
    infile.read((char*)points, numof_points*dim*sizeof(points[0]));

    if(infile.gcount() != numof_points*dim*sizeof(points[0])) {
        cout<<"warning: corrupted read detected: insufficient data in file."<<endl;
        return false;
    }
    infile.close();


    for(int i = 0; i < numof_points; i++) {
        for(int j = 0; j < dim; j++) {
            cout<<points[i*dim+j]<<" ";
        }
        cout<<endl;
    }

    delete [] points;

	return 0;
}


