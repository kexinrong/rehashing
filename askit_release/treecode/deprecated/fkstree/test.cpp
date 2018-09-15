#include<vector>
#include<cassert>
#include<cmath>
#include<utility>
#include<CmdLine.h>
#include <ompUtils.h>
#include <ctime>
#include<omp.h>
#include<mpi.h>
#include<float.h>

#include "generator.h"

#include "fksTree.h"

using namespace Torch;
using namespace std;


int main(int argc, char **argv) {

	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

    CmdLine cmd;
	const char *phelp = "Help";
    cmd.addInfo(phelp);

    int gen;
	cmd.addICmdOption("-gen", &gen, 0, "Data generator (*0*). 0=uniform embedding, 1=unit gaussian embedding");
    int numof_ref_points;
	cmd.addICmdOption("-rn", &numof_ref_points, 1000, "Number of referrence data points per process (1000).");
	int numof_query_points;
	cmd.addICmdOption("-qn", &numof_query_points, 1000, "Number of query data points per process (1000).");
	int dim;
	cmd.addICmdOption("-d", &dim, 4, "Dimension of the space (4).");
	int intrinsic_dim;
	cmd.addICmdOption("-id", &intrinsic_dim, 2, "Intrinsic dimension of the embedding data (2)");

    int k;
	cmd.addICmdOption("-k", &k, 2, "Number of nearest neighbors to find (2).");
	int min_comm_size_per_node;
	cmd.addICmdOption("-mcsptn", &min_comm_size_per_node, 1, "min comm size per tree node (1)");
    int rkdt_maxLevel;
	cmd.addICmdOption("-rkdt_mtl", &rkdt_maxLevel, 10, "maximum random projection tree depth [knn] (default = 10)");
	int rkdt_mppn;
	cmd.addICmdOption("-rkdt_mppn", &rkdt_mppn, 1000, "maximum number of points per random projection tree node [knn] (1000)");
    int rkdt_niters;
	cmd.addICmdOption("-rkdt_iter", &rkdt_niters, 4, "number of random projection trees used [knn] (4)");
    int fks_maxLevel;
	cmd.addICmdOption("-fks_mtl", &fks_maxLevel, 10, "maximum kernel summation tree depth [fks] (default = 10)");
	int fks_mppn;
	cmd.addICmdOption("-fks_mppn", &fks_mppn, 1000, "maximum number of points per kernel summation tree node [fks] (1000)");

	int debug;
	cmd.addICmdOption("-debug", &debug, 0, "output debug informaiton (1) or not (*0*)");

    cmd.read(argc, argv);

    // .1 generate data
	srand((unsigned)time(NULL)*rank);
    fksData *refData = new fksData();
    fksData *queryData = new fksData();

    switch(gen) {
		case 0:
        {
			if(rank == 0) {
                cout << "Distribution: "<<intrinsic_dim
                    <<"-d Uniform Embedding into "<<dim<<"-d space" << endl;
            }
			int np = numof_ref_points + numof_query_points;
			double *tmpX = new double [np*dim];
			generateUniformEmbedding(np, dim, intrinsic_dim, tmpX, comm);

            refData->dim = dim;
            refData->numof_points = numof_ref_points;
            refData->X.resize(numof_ref_points*dim);
            #pragma omp parallel for
			for(int i = 0; i < dim*numof_ref_points; i++)
				refData->X[i] = tmpX[i];

            queryData->dim = dim;
            queryData->numof_points = numof_query_points;
            queryData->X.resize(numof_query_points*dim);
			#pragma omp parallel for
			for(int i = 0; i < dim*numof_query_points; i++)
				queryData->X[i] = tmpX[dim*numof_ref_points+i];

            delete [] tmpX;

            break;
		}

		case 1:
        {
		    if(rank == 0) {
                cout << "Distribution: "<<intrinsic_dim
                    <<"-d Unit Gaussian Embedding into "<<dim<<"-d space" << endl;
            }
			int np = numof_ref_points + numof_query_points;
			double *tmpX = new double [np*dim];
			generateNormalEmbedding(np, dim, intrinsic_dim, tmpX, comm);

            refData->dim = dim;
            refData->numof_points = numof_ref_points;
            refData->X.resize(numof_ref_points*dim);
            #pragma omp parallel for
			for(int i = 0; i < dim*numof_ref_points; i++)
				refData->X[i] = tmpX[i];

            queryData->dim = dim;
            queryData->numof_points = numof_query_points;
            queryData->X.resize(numof_query_points*dim);
			#pragma omp parallel for
			for(int i = 0; i < dim*numof_query_points; i++)
				queryData->X[i] = tmpX[dim*numof_ref_points+i];

            delete [] tmpX;

            break;
		}

		default:
		cerr << "Invalid generator selection" << endl;
		exit(1);
	}

    long nref = numof_ref_points;
    long nquery = numof_query_points;
    long glb_numof_ref_points, glb_numof_query_points;
    long refid_offset, queryid_offset;

	MPI_Allreduce( &nref, &glb_numof_ref_points, 1, MPI_LONG, MPI_SUM, comm );
	MPI_Scan( &nref, &refid_offset, 1, MPI_LONG, MPI_SUM, comm );
	refid_offset -= nref;
    refData->gids.resize(numof_ref_points);
    refData->charges.resize(numof_ref_points);
    #pragma omp parallel for
    for(int i = 0; i < numof_ref_points; i++) {
        refData->gids[i] = refid_offset + (long)i;
        refData->charges[i] = (double)rand()/(double)RAND_MAX;
    }

	MPI_Allreduce( &nquery, &glb_numof_query_points, 1, MPI_LONG, MPI_SUM, comm );
	MPI_Scan( &nquery, &queryid_offset, 1, MPI_LONG, MPI_SUM, comm );
	queryid_offset -= nquery;
	queryData->gids.resize(numof_query_points);
	queryData->charges.resize(numof_query_points);
    #pragma omp parallel for
    for(int i = 0; i < numof_query_points; i++) {
        queryData->gids[i] = queryid_offset + (long)i;
        queryData->charges[i] = (double)rand()/(double)RAND_MAX;
    }

    if(debug) {
        print(refData, comm);
        //print(queryData, comm);
    }


    // .2 build tree
    fksCtx *ctx = new fksCtx();
    ctx->k = k;
    ctx->minCommSize = min_comm_size_per_node;
    ctx->rkdt_niters = rkdt_niters;
    ctx->rkdt_mppn = rkdt_mppn;
    ctx->rkdt_maxLevel = rkdt_maxLevel;
    ctx->fks_mppn = fks_mppn;
    ctx->fks_maxLevel = fks_maxLevel;

    fksTree *tree = new fksTree;
    tree->build(refData, ctx);

    delete tree;


	MPI_Finalize();
	return 0;
}




