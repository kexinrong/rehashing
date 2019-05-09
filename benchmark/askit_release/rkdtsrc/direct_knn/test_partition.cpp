#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <utility>

#include <CmdLine.h>
#include <mpi.h>
#include <ompUtils.h>

#include "direct_knn.h"
#include "generator.h"
#include "knnreduce.h"
#include "parallelIO.h"

using namespace Torch;

void getArgs(int argc, char **argv);			// get command-line arguments

// Command line arguments
int k;					// number of nearest neighbors to find
int dim;				// dimension
long nPts;				// number of reference points
long nQuery;				// number of query points
std::string dataFile;			// name of file containing reference points
std::string queryFile;			// name of file containing query points
int dataParts;				// Number of data partitions
int queryParts;				// Number of query partitions
double r;				// Search radius
char queryType;				// Type of search
char partitionType;			// Type of partitioning to use
bool gen;				// Generate random points
int root = 0;                           // root process where query_k is reduced

const bool DEBUG = false;
bool DEBUG_PRINT = false;

void printPt(std::ostream &out, double* p)			// print point
{
        out << "(" << p[0];
        for (int i = 1; i < dim; i++) {
                out << ", " << p[i];
        }
        out << ")\n";
}

void printPartitioning(double *refPoints, double *queryPoints, int refCount, int queryCount, int dim, MPI_Comm comm)
{
	int rank;
	int size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	for(int printRank = 0; printRank < size; printRank++)
	{
		if(printRank == rank)
		{
			std::cout << "Node #" << rank << " has received " << refCount << " reference points and " << queryCount << " query points." << std::endl;
			std::cout << "Reference points:" << std::endl;
			for(int i = 0; i < refCount; i++)
			{
				printPt(std::cout, &(refPoints[i * dim]));
			}
			std::cout << "Query points:" << std::endl;
			for(int i = 0; i < queryCount; i++)
			{
				printPt(std::cout, &(queryPoints[i * dim]));
			}
			std::cout << std::endl;
			std::cout.flush();
		}
		MPI_Barrier(comm);
	}
}

void printResults(directQueryResults results, int localQueryCount, int *queryIDs, MPI_Comm comm)
{
	int rank;
	int size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	int globalQueryCount = 0;
	MPI_Allreduce(&localQueryCount, &globalQueryCount, 1, MPI_INT, MPI_SUM, comm);
	int queryOffset = knn::getGlobalArrayOffset(rank, size, globalQueryCount);

	for(int printRank = 0; printRank < size; printRank++)
	{
		if(printRank == rank)
		{
			std::cout << "Results calculated by node #" << printRank << ":" << std::endl;
			int neighborOffset = 0;
			for(int i = 0; i < localQueryCount; i++)
			{
				std::cout << "Matches for query #" << queryIDs[i] << ":" << std::endl;
				for(int j = 0; j < results.neighborCounts[i]; j++, neighborOffset++)
				{
					std::cout << results.neighbors[neighborOffset].second << " (distance = " << results.neighbors[neighborOffset].first << ") ";
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
			std::cout.flush();
		}
		MPI_Barrier(comm);
	}
}


int main(int argc, char **argv)
{
        int rank;		// MPI rank
        int size;		// MPI size
        double *refPts;		// data points
        double *queryPts;	// query point
        MPI_Comm comm = MPI_COMM_WORLD;

	// Initialize performance logger

        // initialize buffers to receive the reduced results.
        dist_t *rcvBufR = NULL;
        long *rcvNumR = NULL;

        // Start up MPI and read in arguments.
	MPI_Init(&argc, &argv);
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
        getArgs(argc, argv);

	if(partitionType == 'R' || partitionType == 'r')
	{
		assert(size == dataParts * queryParts);
	}

	long localRefCount;
	long localQueryCount;
	
	int *refIDs;
	int *queryIDs;

	if(gen == true)
	{
		localRefCount = knn::getNumLocal(rank, size, nPts);
		localQueryCount = knn::getNumLocal(rank, size, nQuery);

		refPts = new double[localRefCount * dim];
		queryPts = new double[localQueryCount * dim];

		int *refIDs = new int[localRefCount];
		int *queryIDs = new int[localQueryCount];

		genPointInRandomLine(localRefCount, dim, refPts, refIDs, comm, false, knn::getGlobalArrayOffset(rank, size, nPts));
		genPointInRandomLine(localQueryCount, dim, queryPts, queryIDs, comm, false, knn::getGlobalArrayOffset(rank, size, nQuery));
	}	
	else
	{
		localRefCount = nPts;
		localQueryCount = nQuery;


		knn::parallelIO(dataFile, localRefCount, dim, refPts, comm);
		knn::parallelIO(queryFile, localQueryCount, dim, queryPts, comm);

		refIDs = new int[localRefCount];
		queryIDs = new int[localQueryCount];
		
		int refOffset = knn::getGlobalArrayOffset(rank, size, nPts);
		for(int i = 0; i < localRefCount; i++)
		{
			refIDs[i] = i + refOffset;
		}

		int queryOffset = knn::getGlobalArrayOffset(rank, size, nQuery);
		for(int i = 0; i < localQueryCount; i++)
		{
			queryIDs[i] = i + queryOffset;
		}
	}

	directQueryParams params;
	params.queryType = queryType;
	params.partitionType = partitionType;
	params.k = k;
	params.r = r;
	params.refParts = dataParts;
	params.queryParts = queryParts;

	directQueryResults results = knn::directQuery(refPts, queryPts, localRefCount, localQueryCount, dim, refIDs, queryIDs, params, comm);

	if(DEBUG_PRINT == true)
	{
		printPartitioning(refPts, queryPts, localRefCount, localQueryCount, dim, comm);
		printResults(results, localQueryCount, queryIDs, comm);
	}

	delete[] results.neighbors;
	delete[] results.neighborCounts;
	delete[] refPts;
	delete[] queryPts;
	delete[] refIDs;
	delete[] queryIDs;
	MPI_Finalize();
	return 0;
}



// Read command line arguments
void getArgs(int argc, char **argv)
{
	// Read in the options
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	CmdLine cmd;
	const char *help = "Help";
	char *df = NULL;
	char *qf = NULL;
	char *qt = NULL;
	char *pt = NULL;
	int dc;
	int qc;
	cmd.addInfo(help);
	cmd.addBCmdOption("-g", &gen, false, "Use random points as input (default = false).");
	cmd.addSCmdOption("-df", &df, "data.bin", "Name of file containing data points (default = \"data.bin\").");
	cmd.addSCmdOption("-qf", &qf, "query.bin", "Name of file containing query points (default = \"query.bin\").");
	cmd.addICmdOption("-d", &dim, 2, "Dimension of the space (default = 2).");
	cmd.addICmdOption("-dc", &dc, 20, "Number of data points (default = 20).");
	cmd.addICmdOption("-qc", &qc, 10, "Number of query points (default = 10).");
	cmd.addSCmdOption("-pt", &pt, "R", "Type of partitioning to use ('C' = cyclic or 'R' = rectangular') (default = 'R').");
	cmd.addICmdOption("-dp", &dataParts, 2, "Number of partitions for data vector (only relevant for '-pt R') (default = 2).");
	cmd.addICmdOption("-qp", &queryParts, 2, "Number of partitions for query vector (only relevant for '-pt R') (default = 2).");
	cmd.addSCmdOption("-qt", &qt, "K", "Type of query to run ('K' or 'R') (only relevant for '-pt R') (default = 'K').");
	cmd.addICmdOption("-k", &k, 1, "Number of nearest neighbors per query for queryK (only relevant for '-qt k') (default = 1).");
	cmd.addRCmdOption("-r", &r, 1, "Search radius for queryR (only relevant for '-qt r') (default = 1).");
	cmd.addBCmdOption("-p", &DEBUG_PRINT, false, "Print query results (default = false).");
	cmd.read(argc, argv);

	// Store file names and point counts
	dataFile = df;
	queryFile = qf;
	nPts = dc;
	nQuery = qc;

	if(k > dc)
	{
		if(rank == 0)
		{
			std::cout << "k cannot be larger than dc, because we cannot have more nearest neighbors than data points." << std::endl;
		}
		MPI_Finalize();
		exit(0);
	}

	// Check query type option
	if(qt[0] == 'K' || qt[0] == 'k')
	{
		queryType = 'K';
	}
	else if(qt[0] == 'R' || qt[0] == 'r')
	{
		queryType = 'R';
	}
	else
	{
		if(rank == 0)
		{
			std::cout << "queryType must be 'K' or 'R'." << std::endl;
		}
		MPI_Finalize();
		exit(0);
	}

	// Check partition type option
	if(pt[0] == 'C' || pt[0] == 'c')
	{
		partitionType = 'C';
	}
	else if(pt[0] == 'R' || pt[0] == 'r')
	{
		partitionType = 'R';
	}
	else
	{
		if(rank == 0)
		{
			std::cout << "partitionType must be 'C' or 'R'." << std::endl;
		}
		MPI_Finalize();
		exit(0);
	}
}
