#include <iostream>
#include <fstream>
#include <cctype>
#include <omp.h>
#include <string>
#include <mpi.h>
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
    char *ptrOutputDir = NULL;
	cmd.addSCmdOption("-outdir", &ptrOutputDir, "/scratch/bo", "output directory");

	cmd.read(argc,argv);

	// program start
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

    vector<double> arr;
    int dummy_numof_points;
    knn::mpi_binread(ptrInputFile, numof_points, dim, dummy_numof_points, arr, comm);
    long my_numof_points = dummy_numof_points;
    long my_dim = dim;

    long numof_training_points = my_numof_points * 0.8;
    long numof_testing_points = my_numof_points * 0.1;
    long numof_validating_points = my_numof_points * 0.1;
    cout<<"rank "<<rank
        <<", total = "<<my_numof_points
        <<", train = "<<numof_training_points
        <<", validate = "<<numof_validating_points
        <<", test = "<<numof_testing_points
        <<endl;

    long trn_start = 0;
    long val_start = numof_training_points;
    long tst_start = numof_training_points + numof_validating_points;

    char filename[1024];
    // trn 80%
    double *trn_points = arr.data();
    ofstream trn_outfile;
    sprintf(filename, "%s/trn.bin", ptrOutputDir);
    trn_outfile.open(filename, ios::binary|ios::app);
    if(!trn_outfile.is_open()) {
        cout<<"cannot open trn.bin"<<endl;
        return -1;
    }
    trn_outfile.write((char*)trn_points, numof_training_points*my_dim*(long)sizeof(double));
    trn_outfile.close();

    // val 10%
    double *val_points = arr.data() + numof_training_points*my_dim;
    ofstream val_outfile;
    sprintf(filename, "%s/val.bin", ptrOutputDir);
    val_outfile.open(filename, ios::binary|ios::app);
    if(!val_outfile.is_open()) {
        cout<<"cannot open val.bin"<<endl;
        return -1;
    }
    val_outfile.write((char*)val_points, numof_validating_points*my_dim*(long)sizeof(double));
    val_outfile.close();

    // tst 10%
    double *tst_points = arr.data() + (numof_training_points + numof_validating_points)*my_dim;
    ofstream tst_outfile;
    sprintf(filename, "%s/tst.bin", ptrOutputDir);
    tst_outfile.open(filename, ios::binary|ios::app);
    if(!tst_outfile.is_open()) {
        cout<<"cannot open tst.bin"<<endl;
        return -1;
    }
    tst_outfile.write((char*)tst_points, numof_testing_points*my_dim*(long)sizeof(double));
    tst_outfile.close();


	MPI_Finalize();
	return 0;
}



