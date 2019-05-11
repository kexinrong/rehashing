#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>

#include "generator.h"

void genPointInRandomLine(long localPoints, int dim, double *points, int *globalIDs, MPI_Comm comm, bool shuffle, int offset)
{
	int rank;
	int size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	srand((unsigned)time(NULL)+rank);

	// Create projection random normalize projection vector
	double *projection = new double[dim];
	double norm = 0;
	for(int i = 0; i < dim; i++)
	{
		projection[i] = (double) rand() / RAND_MAX;
		norm += (projection[i] * projection[i]);
	}
	norm = sqrt(norm);
	
	for(int i = 0; i < dim; i++)
	{
		projection[i] /= norm;
	}
	
	MPI_Bcast(projection, dim, MPI_DOUBLE, 0, comm);

	// Generate uniformly spaced points on the interval [rank / size, (rank + 1) / size]
	double last = (double) (rank + 1) / size;
	double first = (double) rank / size;
	double step = (last - first) / (localPoints - 1);
	double *originalPoints = new double[localPoints];
	originalPoints[0] = first;
	globalIDs[0] = offset;
	for(int i = 1; i < localPoints; i++)
	{
		originalPoints[i] = originalPoints[i - 1] + step;
		globalIDs[i] = i + offset;
	}
	
	// Shuffle, if necessary
	if(shuffle == true)
	{
		// Shuffle both arrays in exactly the same way
		for(int i = localPoints - 1; i > 0; i--)
		{
			int randomIndex = rank % (i + 1);
			std::swap(globalIDs[i], globalIDs[randomIndex]);
			std::swap(originalPoints[i], originalPoints[randomIndex]);
		}
	}

	// Project the points
	for(int i = 0; i < localPoints; i++)
	{
		for(int j = 0; j < dim; j++)
		{
			points[i * dim + j] = originalPoints[i] * projection[j];
		}
	}

	// Clean-up
	delete[] originalPoints;
	delete[] projection;
}
