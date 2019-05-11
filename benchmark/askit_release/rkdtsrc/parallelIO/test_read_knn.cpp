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

void print_knn(int numof_points, int k, long first_gid,
            vector< pair<double, long> > *kNN, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    for(int r = 0; r < size; r++) {
        if(rank == r) {
            for(int i = 0; i < numof_points; i++) {
                cout<<first_gid+i<<": ";
                for(int j = 0; j < k; j++) {
                    cout<<"("<<(*kNN)[i*k+j].second<<" "
                        <<(*kNN)[i*k+j].first<<")  ";
                }
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
    int numof_points;
	cmd.addICmdOption("-n", &numof_points, 1000, "number of points per rank");
	int k;
	cmd.addICmdOption("-k", &k, 1, "dim");

    bool isBinary;
    cmd.addBCmdOption("-binary", &isBinary, false, "use binary file");

    bool isDisp;
    cmd.addBCmdOption("-disp", &isDisp, false, "display read results");

	cmd.read(argc,argv);

	// program start
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

    int id_offset = 0;
    MPI_Scan(&numof_points, &id_offset, 1, MPI_INT, MPI_SUM, comm);
    id_offset -= numof_points;

    vector< pair<double, long> > *knn_rkdt 
                        = new vector< pair<double, long> >();
    vector< pair<double, long> > *knn_rkdt_old 
                        = new vector< pair<double, long> >();
    if(isBinary) {
        double start_t = omp_get_wtime();
        knn::binread_knn(ptrInputFile, id_offset, numof_points, k, knn_rkdt);
        cout<<"new bin read time: "<<omp_get_wtime()-start_t<<endl;

        if(isDisp) {
            if(rank == 0) cout<<"new bin read: "<<endl;
            print_knn(numof_points, k, id_offset, knn_rkdt, comm);
        }

        vector<long> queryIDs(numof_points);
        for(int i = 0; i < queryIDs.size(); i++)
            queryIDs[i] = id_offset + i;
        start_t = omp_get_wtime();
        knn::binread_knn(ptrInputFile, queryIDs, k, knn_rkdt_old);
        cout<<"old bin read time: "<<omp_get_wtime()-start_t<<endl;

        if(isDisp) {
            if(rank == 0) cout<<"old bin read: "<<endl;
            print_knn(numof_points, k, id_offset, knn_rkdt_old, comm);
        }

    }
    else {
        // do nothing
    }

	MPI_Finalize();
	return 0;

}



