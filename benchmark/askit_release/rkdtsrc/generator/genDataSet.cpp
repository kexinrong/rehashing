#include "sphere.cpp"

#include <fstream>
#include <mpi.h>
#include <iostream>

int main(int argc, char ** argv)
{
	// Set up MPI
	MPI_Init(&argc, &argv);
	int rank;
	int size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// Generate points
	int npts = 1000000;
	int dim = 4;
	double r = 5;
	double *buf = mpi_gene_4dsphere_bcast(&npts, &r, MPI_COMM_WORLD); 

	// Write metadata to file
   	const int64_t endiancheck = 0x1234ABCD;
        std::ofstream ofile("data.bin", std::ofstream::binary);
        ofile.write((char*)&endiancheck, sizeof(endiancheck)); 	
	ofile.close();

	// Write points to file
	for(int i = 0; i < size; i++)
	{
        	if(rank == i)
		{
        	        std::ofstream ofile("data.bin", std::ofstream::binary | std::ofstream::app);
        	        ofile.write((char*)buf, npts * dim * sizeof(double));
        	        ofile.close();
			std::cout << "Rank " << i << " successfully wrote to file." << std::endl;
			std::cout.flush();
        	}
		
		MPI_Barrier(MPI_COMM_WORLD);
	}

	delete[] buf;
	MPI_Finalize();
	return 0;
}
