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

const bool DEBUG = true;
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

	int printRank = 0;
	while(printRank < size)
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
			printRank++;
			MPI_Bcast(&printRank, 1, MPI_INT, printRank - 1, comm);
		}
		else
		{
			MPI_Bcast(&printRank, 1, MPI_INT, printRank, comm);
		}
	}
}

void printRectKResults(dist_t *rcvBufK, int queryParts, int nQuery, int queryOffset, MPI_Comm comm)
{
	int rank;
	MPI_Comm_rank(comm, &rank);

	int printRank = 0;
	while(printRank < queryParts)
	{
		if(printRank == rank)
		{
			for(long i = 0; i < nQuery; i++)
			{
				std::cout << "Matches for query point #" << (queryOffset + i) << ": " << std::endl;
				for(long j = 0; j < k; j++)
				{
					long global_idx = rcvBufK[i * k + j].second;
					// print the global index of the ref pt.
					std::cout << global_idx << " (distance = " << rcvBufK[i * k + j].first << ") ";
				}
				std::cout << std::endl;
			}
			printRank++;
			MPI_Bcast(&printRank, 1, MPI_INT, printRank - 1, comm);
		}
		else
		{
			MPI_Bcast(&printRank, 1, MPI_INT, printRank, comm);
		}
	}
}

void printRectRResults(dist_t *rcvBufR, long *rcvNumR, int queryParts, int nQuery, int queryOffset, int nPts, MPI_Comm comm)
{
	int rank;
	int size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	long offset = 0;
	int printRank = 0;
	while(printRank < queryParts)
	{
		if(printRank == rank)
		{
			std::cout << "Results calculated by node #" << rank << ":" << std::endl;
			for(long i = 0; i < nQuery; i++)
			{
				std::cout << "Matches for query #" << (queryOffset + i) << ":" << std::endl;
				long num = rcvNumR[i];
				for(long j = 0; j < num; j++)
				{
					long global_idx = (rcvBufR + offset)[j].second;
					// print the global index of the ref pt.
					std::cout << global_idx << " (distance = " << (rcvBufR + offset)[j].first << ") ";
				}
				std::cout << std::endl;
				offset += nPts;
			}
			std::cout << std::endl;
			printRank++;
			MPI_Bcast(&printRank, 1, MPI_INT, printRank - 1, comm);
		}
		else
		{
			MPI_Bcast(&printRank, 1, MPI_INT, printRank, comm);
		}
	}
}

void printCyclicResults(std::pair<double, long> *results, int nQuery, int k, MPI_Comm comm)
{
	int rank;
	int size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	int printRank = 0;
	while(printRank < size)
	{
		if(printRank == rank)
		{
			std::cout << "Results calculated by node #" << rank << ":" << std::endl;
			for(int i = 0; i < nQuery; i++)
			{
				std::cout << "Matches for query #" << i << ": " << std::endl;
				for(int j = 0; j < k; j++)
				{
					std::cout << results[i * k + j].second << " (distance = " << results[i * k + j].first << ") ";
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
			printRank++;
			MPI_Bcast(&printRank, 1, MPI_INT, printRank - 1, comm);
		}
		else
		{
			MPI_Bcast(&printRank, 1, MPI_INT, printRank, comm);
		}
	}
}


int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
        int rank;		// MPI rank
        int size;		// MPI size
        double *dataPts;	// data points
        double *queryPts;	// query point
        MPI_Comm comm = MPI_COMM_WORLD;

        // initialize buffers to receive the reduced results.
        dist_t *rcvBufR = NULL;
        long *rcvNumR = NULL;

        // Start up MPI and read in arguments.
        //MPI_Init(&argc, &argv);
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
        getArgs(argc, argv);

	if(partitionType == 'R')
	{
		// Verify that we have the correct number of nodes for our partitioning.
		if(size != dataParts * queryParts)
		{
			if(rank == 0)
			{
				std::cerr << "Wrong number of nodes to do the partitioning.\n";
			}
			MPI_Finalize();
			exit(1);
		}

		// Partition points
		std::pair<int, int> points;
		if(gen == true)
		{
			// Group reference points
			int *ranks = new int[dataParts];
			for(int i = 0; i < dataParts; i++)
			{
				ranks[i] = rank % queryParts + i * queryParts;
			}	
			MPI_Comm queryComm; // Group of nodes with same query points
			MPI_Group mainGroup;
			MPI_Comm_group(comm, &mainGroup);
			MPI_Group group;
			MPI_Group_incl(mainGroup, dataParts, ranks, &group);
			delete[] ranks;
			MPI_Comm_create(comm, group, &queryComm);

			// Group query points
			ranks = new int[queryParts];
			for(int i = 0; i < queryParts; i++)
			{
				ranks[i] = rank / queryParts * queryParts + i;
			}
			MPI_Group_incl(mainGroup, queryParts, ranks, &group);
			delete[] ranks;
			MPI_Comm refComm; // Group of nodes with same ref points
			MPI_Comm_create(comm, group, &refComm);

			int localPoints;
			int localQueries;
			if(rank < size - 1)
			{
				localPoints = nPts / dataParts;
				localQueries = nQuery / queryParts;
			}
			else
			{
				localPoints = nPts % dataParts ? nPts % dataParts : nPts / dataParts;
				localQueries = nQuery % queryParts ? nQuery % queryParts : nQuery / queryParts;
			}

			dataPts = new double[localPoints * dim];
			int *dataIDs = new int[localPoints];

			// Generate reference points
			if(rank % queryParts == 0)
			{
				genPointInRandomLine(localPoints, dim, dataPts, dataIDs, queryComm, false, nPts / dataParts * rank);	
			}

			// Broadcast ref points
			MPI_Bcast(dataPts, localPoints * dim, MPI_DOUBLE, 0, refComm);

			queryPts = new double[localQueries * dim];
			int *queryIDs = new int[localQueries];

			// Generate query points
			if(rank < queryParts)
			{
				genPointInRandomLine(localQueries, dim, queryPts, queryIDs, refComm, false, nQuery / queryParts * rank);
			}

			// Broadcast query points
			MPI_Bcast(queryPts, localQueries * dim, MPI_DOUBLE, 0, queryComm);

			// Clean-up
			delete[] dataIDs;
			delete[] queryIDs;
			MPI_Comm_free(&refComm);
			MPI_Comm_free(&queryComm);

			points = std::make_pair(localPoints, localQueries);
		}
		else
		{
			points = knn::partitionPoints(dataFile, queryFile, dataPts, queryPts, nPts, nQuery, dim, dataParts, queryParts, comm);
		}

		// Determine the global index of the first reference and query points locally and store how many points we have locally.
		long numPtsOffset = (nPts / dataParts) * (rank / queryParts);
		long queryOffset = (nQuery / queryParts) * (rank % queryParts);
		int globalPts = points.first;
		int globalQuery = points.second;
		nPts = points.first;
		nQuery = points.second;

		// Print our partitioning.
		if(DEBUG_PRINT == true)
		{
			printPartitioning(dataPts, queryPts, nPts, nQuery, dim, comm);
		}

		// Calculate distances
		double *D = new double[nPts * nQuery];
		double startTime, endTime;
		if(DEBUG == true)
		{
			startTime = MPI_Wtime();	
		}
		knn::compute_distances(dataPts, queryPts, nPts, nQuery, dim, D);
		if(DEBUG == true)
		{
			endTime = MPI_Wtime();
			pl.logTime(0, endTime - startTime);
			pl.logFlops(0, nPts * nQuery * (2 * dim + 1));
		}

		// Group distances with reference index and sort
		dist_t * inbuf = new dist_t[nPts * nQuery];
		#pragma omp parallel for
		for(int i = 0; i < nQuery; i++)
		{
			for(int j = 0; j < nPts; j++)
			{
				inbuf[i * nPts + j].first = D[i * nPts + j];
				// store global indices of ref pts
				inbuf[i * nPts + j].second = numPtsOffset + j;
			}
		}

		if(DEBUG == true)
		{
			startTime = MPI_Wtime();
		}
		#pragma omp parallel for
		for(int i = 0; i < nQuery; i++)
		{
			//omp_par::merge_sort(inbuf + i * nPts, inbuf + (i + 1) * nPts);
                        std::sort(inbuf + i * nPts, inbuf + (i+1) * nPts);
		}
		if(DEBUG == true)
		{
			endTime = MPI_Wtime();
			pl.logTime(1, endTime - startTime);
		}

		// Run our query
		if(DEBUG == true)
		{
			startTime = MPI_Wtime();
		}
		if(queryType == 'K')
		{
			// Group nodes by query partition.
			int *ranks = new int[dataParts];
			#pragma omp parallel for
			for(int i = 0; i < dataParts; i++)
			{
				ranks[i] = rank % queryParts + i * queryParts;
			}
			MPI_Group mainGroup;
			MPI_Comm_group(comm, &mainGroup);
			MPI_Group group;
			MPI_Group_incl(mainGroup, dataParts, ranks, &group);
			delete[] ranks;
			MPI_Comm tmpComm;
			MPI_Comm_create(comm, group, &tmpComm);

			// Perform the search
		        dist_t *rcvBufK = NULL;
			knn::query_k(tmpComm, k, 0, inbuf, nQuery, nPts, rcvBufK);
			MPI_Comm_free(&tmpComm);
	
			// Print results
			if(DEBUG_PRINT == true)
			{
				printRectKResults(rcvBufK, queryParts, nQuery, queryOffset, comm);
			}
		}
		else // queryType == 'R'
		{
			// Perform the serach
			knn::query_r(comm, r * r, inbuf, nQuery, nPts, rcvBufR, rcvNumR);

			// Group by query points in common
			int *ranks = new int[dataParts];
			for(int i = 0; i < dataParts; i++)
			{
				ranks[i] = rank % queryParts + i * queryParts;
			}
			MPI_Group mainGroup;
			MPI_Comm_group(comm, &mainGroup);
			MPI_Group group;
			MPI_Group_incl(mainGroup, dataParts, ranks, &group);
			delete[] ranks;
			MPI_Comm tmpComm;
			MPI_Comm_create(comm, group, &tmpComm);

			int newRank;
			int newSize;
			MPI_Comm_rank(tmpComm, &newRank);
			MPI_Comm_size(tmpComm, &newSize);

			// Gather number of points you will receive
			long *receiveCounts = new long[nQuery];
			MPI_Allreduce(rcvNumR, receiveCounts, nQuery, MPI_LONG, MPI_SUM, tmpComm);
		
			long totalPoints = 0;	
			for(int i = 0; i < nQuery; i++)
			{
				totalPoints += receiveCounts[i];
			}

			// Calculate total number of points to receive
			dist_t *result = new dist_t[totalPoints];
			long *indexResults = new long[totalPoints];
			double *distanceResults = new double[totalPoints];

			// Distribute results
			int offset = 0;
			int globalOffset = 0;
			for(int i = 0; i < nQuery; i++)
			{
				// Copy values to send
				long *tmpIndices = new long[rcvNumR[i]];
				double *tmpDistances = new double[rcvNumR[i]];
				for(int j = 0; j < rcvNumR[i]; j++)
				{
					tmpIndices[j] = rcvBufR[j + offset].second;
					tmpDistances[j] = rcvBufR[j + offset].first;
				}

				// Gather number of values each node is sending
				int *receive = new int[newSize];
				int *displ = new int[newSize];
				MPI_Gather(&rcvNumR[i], 1, MPI_INT, receive, 1, MPI_INT, 0, tmpComm);
				std::partial_sum(receive, receive + newSize - 1, displ + 1);
				displ[0] = 0;

				MPI_Gatherv(tmpIndices, rcvNumR[i], MPI_LONG_LONG, indexResults + globalOffset, receive, displ, MPI_LONG_LONG, 0, tmpComm);
				MPI_Gatherv(tmpDistances, rcvNumR[i], MPI_DOUBLE, distanceResults + globalOffset, receive, displ, MPI_DOUBLE, 0, tmpComm);
		
				delete[] receive;
				delete[] displ;
				delete[] tmpIndices;
				delete[] tmpDistances;

				offset += rcvNumR[i];
				globalOffset += receiveCounts[i];
			}

			// Copy results
			if(newRank == 0)
			{
				for(int i = 0; i < totalPoints; i++)
				{
					result[i].second = indexResults[i];
					result[i].first = distanceResults[i];
				}
			}
			
			MPI_Comm_free(&tmpComm);

			// Print results
			if(DEBUG_PRINT == true && newRank == 0)
			{
				printRectRResults(result, receiveCounts, queryParts, nQuery, queryOffset, nPts, comm);
			}

			delete[] distanceResults;
			delete[] indexResults;
			delete[] result;
			delete[] receiveCounts;
		}
		if(DEBUG == true)
		{
			endTime = MPI_Wtime();
			pl.logTime(2, endTime - startTime);
		}

		if(DEBUG == true)
		{
			printPerfResults(pl, comm);
		}

	        // Clean up
	        delete[] D;
	        delete[] inbuf;
	}
	else // cyclic partitioning
	{
		int globalPts = nPts;
		int globalQuery = nQuery;
	
		if(gen == true)
		{
			if(rank < size - 1)
			{
				nPts = globalPts / size;
				nQuery = globalQuery / size;
			}
			else
			{
				nPts = globalPts % size ? globalPts % size : globalPts / size;
				nQuery = globalQuery % size ? globalQuery % size : globalQuery / size;
			}

			dataPts = new double[globalPts * dim];
			queryPts = new double[globalQuery * dim];
			int *dataID = new int[globalPts];
			int *queryID = new int[globalQuery];

			genPointInRandomLine(globalPts, dim, dataPts, dataID, comm, false, globalPts / size * rank);
			genPointInRandomLine(globalQuery, dim, queryPts, queryID, comm, false, globalQuery / size * rank);

			delete[] dataID;
			delete[] queryID;
		}
		else
		{
			knn::parallelIO(dataFile, nPts, dim, dataPts, comm);
			knn::parallelIO(queryFile, nQuery, dim, queryPts, comm);
		}


		std::pair<double, long> *results = knn::directKQuery(dataPts, queryPts, globalPts, globalQuery, k, dim);
		if(DEBUG_PRINT == true)
		{
			printCyclicResults(results, nQuery, k, comm);
		}

		if(DEBUG == true)
		{
			printPerfResults(pl, comm);
		}

		delete[] results;
	}

	delete[] dataPts;
	delete[] queryPts;
        MPI_Finalize();
        return EXIT_SUCCESS;
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
