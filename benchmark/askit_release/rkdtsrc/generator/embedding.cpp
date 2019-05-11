#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <stdlib.h>
#include <assert.h>
#include <vector>
#include <iostream>

#include "rotation.h"
#include "generator.h"

using namespace std;

void generateUniformEmbedding(int numof_points, int dim, int intrinsicDim, double *points, MPI_Comm comm)
{
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	assert(intrinsicDim <= dim);

	// padding low dim points with zeros
	vector<double> tmpPoints(numof_points*dim);
	generateUniform(numof_points, dim, &(tmpPoints[0]), comm);
	#pragma omp parallel for
	for(int i = 0; i < numof_points; i++) {
		for(int j = intrinsicDim; j < dim; j++) {
			tmpPoints[i*dim+j] = 0.0;
		}
	}

	// create a random orthogonal matrix
	vector<double> rw;
	generateRotation(dim, rw, comm);

	//vector<double> Mrand(dim*dim);
	//vector<double> Morth(dim*dim);
	//for(int i = 0; i < dim*dim; i++)
	//	Mrand[i] = (double)rand() / (double)RAND_MAX;
	//GramSchmidt(&(Mrand[0]), &(Morth[0]), dim, dim);
	//MPI_Bcast(&(Morth[0]), dim*dim, MPI_DOUBLE, 0, comm);

	// rotate points
	rotatePoints(&(tmpPoints[0]), numof_points, dim, rw, points);

	//#pragma omp parallel for
	//for(int i = 0; i < numof_points; i++) {
	//	for(int j = 0; j < dim; j++)
	//		points[i*dim+j] = prod(&(Morth[j*dim]), &(tmpPoints[i*dim]), dim);
	//}

}



void generateNormalEmbedding(int numof_points, int dim, int intrinsicDim, double *points, MPI_Comm comm)
{
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	assert(intrinsicDim <= dim);

    //MPI_Barrier(comm);
    //if(rank == 0) {
    //    cout.flush();
    //    cout<<"rank "<<rank<<": enter generateNormalEmbedding"<<endl;
    //}

	// padding low dim points with zeros
	vector<double> tmpPoints(numof_points*dim);
	generateNormal(numof_points, dim, &(tmpPoints[0]), comm);
	#pragma omp parallel for
    for(int i = 0; i < numof_points; i++) {
        for(int j = intrinsicDim; j < dim; j++) {
			tmpPoints[i*dim+j] = 0.0;
		}
	}

    MPI_Barrier(comm);
    if(rank == 0) {
        cout.flush();
        cout<<"rank "<<rank<<": generateNormal done"<<endl;
    }

	// create a random orthogonal matrix
	vector<double> rw;
	generateRotation(dim, rw, comm);
	//vector<double> Mrand(dim*dim);
	//vector<double> Morth(dim*dim);
	//for(int i = 0; i < dim*dim; i++)
	//	Mrand[i] = (double)rand() / (double)RAND_MAX;
	//GramSchmidt(&(Mrand[0]), &(Morth[0]), dim, dim);
	//MPI_Bcast(&(Morth[0]), dim*dim, MPI_DOUBLE, 0, comm);

	// rotate points
	rotatePoints(&(tmpPoints[0]), numof_points, dim, rw, points);

    //#pragma omp parallel for
	//for(int i = 0; i < numof_points; i++) {
	//	for(int j = 0; j < dim; j++)
	//		points[i*dim+j] = prod(&(Morth[j*dim]), &(tmpPoints[i*dim]), dim);
	//}

    MPI_Barrier(comm);
    if(rank == 0) {
        cout.flush();
        cout<<"rank "<<rank<<": rotate embedding points done"<<endl;
    }
}


