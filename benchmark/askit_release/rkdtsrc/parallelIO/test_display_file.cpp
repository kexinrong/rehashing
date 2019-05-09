#include <mpi.h>
#include <iostream>
#include <fstream>
#include <cctype>
#include <omp.h>
#include <string>
#include <climits>

#include "CmdLine.h"
#include "parallelIO.h"
#include "generator.h"

using namespace Torch;
using namespace std;

void printpbyp(double *arr, int numof_points, int dim, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    long dummy_n = numof_points;
    long offset;
    MPI_Scan(&dummy_n, &offset, 1, MPI_LONG, MPI_SUM, comm);
    offset -= dummy_n;

    for(int r = 0; r < size; r++) {
        if(rank == r) {
            for(int i = 0; i < numof_points; i++) {
                cout<<"(rank "<<rank<<") "<<offset+i<<": ";
                for(int j = 0; j < dim; j++)
                    cout<<arr[i*dim+j]<<" ";
                cout<<endl;
            }
        }
        cout.flush();
        MPI_Barrier(comm);
    }
}


int main(int argc, char **argv) {

	// command lines
	CmdLine cmd;
	const char *pchHelp = "General info.";
	cmd.addInfo(pchHelp);

	char *ptrInputFile = NULL;
	cmd.addSCmdOption("-file", &ptrInputFile, "data.bx", "file");
    long numof_points;
	cmd.addLCmdOption("-n", &numof_points, 1000, "number of points per rank");
	int dim;
	cmd.addICmdOption("-d", &dim, 1, "dim");

    bool isBinary;
    cmd.addBCmdOption("-binary", &isBinary, false, "use binary file");

	cmd.read(argc,argv);

	// program start
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

    if(isBinary) {
        vector<double> arr;
        int dummy_numof_points;
        knn::mpi_binread(ptrInputFile, numof_points, dim, dummy_numof_points, arr, comm);

        if(rank == 0) cout<<"data read: "<<endl;
        printpbyp(&(arr[0]), dummy_numof_points, dim, comm);
        if(rank == 0) cout<<endl;
    }
    else {
        vector<double> arr;
        knn::mpi_dlmread(ptrInputFile, numof_points, dim, arr, comm, false);

        if(rank == 0) cout<<"data read: "<<endl;
        printpbyp(&(arr[0]), numof_points, dim, comm);
        if(rank == 0) cout<<endl;
    }

	MPI_Finalize();
	return 0;

}



