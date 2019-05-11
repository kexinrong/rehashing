#include <cstring>
#include <assert.h>
#include "rotation.h"
#include "generator.h"
#include "direct_knn.h"


void generateRotation(int dim, vector<double> &w) 
{
	int nsteps = 10;
	int sz = int(ceil(3*nsteps*dim + 2*dim + dim/4.0 + 50*sizeof(double) +1));
	w.resize(sz);
	int keep;
	RROT_INIT( &nsteps, &dim, &(w[0]), &keep );
}


void generateRotation(int dim, vector<double> &w, MPI_Comm comm) 
{
	int nsteps = 10;
	int sz = int(ceil(3*nsteps*dim + 2*dim + dim/4.0 + 50*sizeof(double) +1));
	w.resize(sz);
	int keep;
	RROT_INIT( &nsteps, &dim, &(w[0]), &keep );

	MPI_Bcast(&(w[0]), sz, MPI_DOUBLE, 0, comm);
}


// NOTE: RROT and RROT_INV change the values of input
// need to make a copy to rotate points
void rotatePoints(const double *points, int numof_points, int dim,
				  vector<double> &w,
				  double *newPoints)
{
	//double *points_clone = new double [numof_points*dim];
	//memcpy(points_clone, points, sizeof(double)*numof_points*dim);
	#pragma omp parallel if(numof_points > 500)
	{
		double *point_clone = new double [dim];
		#pragma omp for
		for(int i = 0; i < numof_points; i++) {
			//RROT(points_clone+i*dim, newPoints+i*dim, &(w[0]));
			//double point_clone[dim];
			memcpy(point_clone, points+i*dim, sizeof(double)*dim);
			RROT(point_clone, newPoints+i*dim, &(w[0]));
		}
		delete [] point_clone;
	}
	//delete [] points_clone;
}


void inverseRotatePoints(const double *points, int numof_points, int dim,
				  vector<double> &w,
				  double *newPoints)
{
	//double *points_clone = new double [numof_points*dim];
	//memcpy(points_clone, points, sizeof(double)*numof_points*dim);
	//double *point_clone = new double [dim];
	#pragma omp parallel if(numof_points > 500)
	{
		double *point_clone = new double [dim];
		#pragma omp for
		for(int i = 0; i < numof_points; i++) {
			memcpy(point_clone, points+i*dim, sizeof(double)*dim);
			RROT_INV(point_clone, newPoints+i*dim, &(w[0]));
		}
		delete [] point_clone;
	}
	//delete [] points_clone;
}



void newRotatePoints(double *points, int numof_points, int dim, vector<double> &w)
{
	double *point_clone = new double [dim];
	for(int i = 0; i < numof_points; i++) {
	    memcpy(point_clone, points+i*dim, sizeof(double)*dim);
		RROT(point_clone, points+i*dim, &(w[0]));
	}
	delete [] point_clone;
}


void newInverseRotatePoints(double *points, int numof_points, int dim, vector<double> &w)
{
	double *point_clone = new double [dim];
	for(int i = 0; i < numof_points; i++) {
		memcpy(point_clone, points+i*dim, sizeof(double)*dim);
		RROT_INV(point_clone, points+i*dim, &(w[0]));
	}
	delete [] point_clone;
}



/*
void generateRotation(int dim, double *R)
{
	vector<double> MatRand(dim*dim);
	for(int i = 0; i < dim*dim; i++)
		MatRand[i] = (double)rand() / (double)RAND_MAX;
	
	GramSchmidt( &(MatRand[0]), R, dim, dim );
}


void generateRotation(int dim, double *R, MPI_Comm comm)
{
	vector<double> MatRand(dim*dim);
	for(int i = 0; i < dim*dim; i++)
		MatRand[i] = (double)rand() / (double)RAND_MAX;
	
	GramSchmidt( &(MatRand[0]), R, dim, dim );

	MPI_Bcast(R, dim*dim, MPI_DOUBLE, 0, comm);

}


void rotatePoints(double *points, int numof_points, int dim, 
				  double *MatRotation,
				  // output
				  double *newPoints)
{
	int blocksize;
	if( numof_points > KNN_MAX_BLOCK_SIZE ) {
		blocksize = std::min(KNN_MAX_BLOCK_SIZE, numof_points);
    } else {
        blocksize = numof_points;
	}

	assert(blocksize > 0);
    int nblocks = (int) numof_points / blocksize;
	int iters = (int) ceil((double)numof_points/(double)blocksize);

    double alpha = 1.0;
	double beta = 0.0;
	for(int i = 0; i < iters; i++) {
		double *currpts = points + i*blocksize*dim;
		double *currnewpts = newPoints + i*blocksize*dim;
        if( (i == iters-1) && (numof_points % blocksize) ) {
            blocksize = numof_points%blocksize;
        }
        bool omptest = blocksize > 4 * omp_get_max_threads();
        #pragma omp parallel if( omptest )
        {
            int omp_num_points, last_omp_num_points;
            int t = omp_get_thread_num();
            int numt = omp_get_num_threads();
            omp_num_points = blocksize / numt;
            last_omp_num_points = blocksize - (omp_num_points * (numt-1));

            //This thread's number of points
            int npoints = (t == numt-1) ? last_omp_num_points : omp_num_points;
            dgemm( "T", "N", &dim, &npoints, &dim, &alpha, MatRotation, &dim,
                    currpts  + (dim*t*omp_num_points),
                    &dim, &beta, currnewpts + (dim*t*omp_num_points), &dim );
        }

    }	// end for (i < iters)

}


void inverseRotatePoints(double *points, int numof_points, int dim, 
				  double *MatRotation,
				  // output
				  double *newPoints)
{
	int blocksize;
	if( numof_points > KNN_MAX_BLOCK_SIZE ) {
		blocksize = std::min(KNN_MAX_BLOCK_SIZE, numof_points);
    } else {
        blocksize = numof_points;
	}

	assert(blocksize > 0);
    int nblocks = (int) numof_points / blocksize;
	int iters = (int) ceil((double)numof_points/(double)blocksize);

    double alpha = 1.0;
	double beta = 0.0;
	for(int i = 0; i < iters; i++) {
		double *currpts = points + i*blocksize*dim;
		double *currnewpts = newPoints + i*blocksize*dim;
        if( (i == iters-1) && (numof_points % blocksize) ) {
            blocksize = numof_points%blocksize;
        }
        bool omptest = blocksize > 4 * omp_get_max_threads();
        #pragma omp parallel if( omptest )
        {
            int omp_num_points, last_omp_num_points;
            int t = omp_get_thread_num();
            int numt = omp_get_num_threads();
            omp_num_points = blocksize / numt;
            last_omp_num_points = blocksize - (omp_num_points * (numt-1));

            //This thread's number of points
            int npoints = (t == numt-1) ? last_omp_num_points : omp_num_points;
            dgemm( "N", "N", &dim, &npoints, &dim, &alpha, MatRotation, &dim,
                    currpts  + (dim*t*omp_num_points),
                    &dim, &beta, currnewpts + (dim*t*omp_num_points), &dim );
        }

    }	// end for (i < iters)

}
*/





