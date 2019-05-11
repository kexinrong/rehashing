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

    for(int r = 0; r < size; r++) {
        if(rank == r) {
            for(int i = 0; i < numof_points; i++) {
                cout<<"(rank "<<rank<<") ";
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
    int numof_ref_points;
	cmd.addICmdOption("-nref", &numof_ref_points, 1000, "number of points per rank");
	int dim;
	cmd.addICmdOption("-d", &dim, 1, "dim");
	int intrinsicDim;
	cmd.addICmdOption("-id", &intrinsicDim, 1, "intrinsic dim");
	int gen;
	cmd.addICmdOption("-gen", &gen, 0, "0: uniform, 1: normal, 2: embeded uniform, 3: embeded normal");

    bool isBinary;
    cmd.addBCmdOption("-binary", &isBinary, false, "use binary file");

	cmd.read(argc,argv);

	// program start
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	double *ref;
	ref = new double[numof_ref_points*dim];

	srand((unsigned)time(NULL)*rank);
    switch(gen) {
		case 0:
        {
			if(rank == 0)
                cout << "Distribution: Uniform" << endl;
			generateUniform(numof_ref_points, dim, ref, comm);
			break;
        }

		case 1:
        {
			if(rank == 0)
                cout << "Distribution: Unit gaussian" << endl;
			generateNormal(numof_ref_points, dim, ref, comm);
			break;
        }

		case 2:
        {
			if(rank == 0)
                cout << "Distribution: "<<intrinsicDim<<"-d Unit Gaussian Embedding into "<<dim<<"-d space" << endl;
			generateUniformEmbedding(numof_ref_points, dim, intrinsicDim, ref, comm);
			break;
		}

		case 3:
        {
			if(rank == 0)
                cout << "Distribution: "<<intrinsicDim<<"-d Unit Gaussian Embedding into "<<dim<<"-d space" << endl;
			generateNormalEmbedding(numof_ref_points, dim, intrinsicDim, ref, comm);
			break;
		}

		default:
		cerr << "Invalid generator selection" << endl;
		exit(1);
	}

    if(rank == 0) cout<<"data generated!"<<endl;

    //if(rank == 0) cout<<"data generated: "<<endl;
    //printpbyp(ref, numof_ref_points, dim, comm);
    //if(rank == 0) cout<<endl;

    long loc_n = numof_ref_points;
    long glb_n;
    MPI_Allreduce(&loc_n, &glb_n, 1, MPI_LONG, MPI_SUM, comm);

    if(isBinary) {
        knn::mpi_binwrite(ptrInputFile, numof_ref_points, dim, ref, comm);

        //vector<double> arr;
        //int dummy_numof_points;
        //knn::mpi_binread(ptrInputFile, glb_n, dim, dummy_numof_points, arr, comm);

        //if(rank == 0) cout<<"data read: "<<endl;
        //printpbyp(&(arr[0]), dummy_numof_points, dim, comm);
        //if(rank == 0) cout<<endl;
    }
    else {
        knn::mpi_dlmwrite(ptrInputFile, numof_ref_points, dim, ref, comm);

        //vector<double> arr;
        //long dummy_numof_points;
        //knn::mpi_dlmread( ptrInputFile,  dummy_numof_points, dim, arr, comm );

        //if(rank == 0) cout<<"data read: "<<endl;
        //printpbyp(&(arr[0]), dummy_numof_points, dim, comm);
        //if(rank == 0) cout<<endl;
    }

    delete [] ref;

	MPI_Finalize();
	return 0;

}



