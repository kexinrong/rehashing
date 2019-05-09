#ifndef __GENEDATA_H__
#define __GENEDATA_H__

#include <mpi.h>
#include <stdlib.h>
#include <utility>

#define PI 3.1416

double* mpi_gene_4dsphere_bcast(int *numPoints, double *radius, MPI_Comm comm);
double* n_sphere(int numof_points, int dim, double max_radius, MPI_Comm comm);

#endif
