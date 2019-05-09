#include<direct_knn.h>
#include<knnreduce.h>
#include<lsh.h>
#include<generator.h>
#include<vector>
#include<cassert>
#include<cmath>
#include<utility>
#include<omp.h>
#include<mpi.h>
#include<CmdLine.h>
#include <ompUtils.h>
#include <ctime>

#include "mpitree.h"
#include "clustering.h"
#include "repartition.h"
#include "binTree.h"
#include "binQuery.h"
#include "distributeToLeaf.h"
#include "parallelIO.h"
#include "eval.h"
//#include "random123wrapper.h"
#include "papi_perf.h"
#include "verbose.h"


using namespace Torch;
using namespace std;


int main(int argc, char **argv) {

	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if(rank == 0) cout<<"tree starts ... np = "<<size<<endl;

	CmdLine cmd;
	const char *phelp = "Help";
    cmd.addInfo(phelp);

    int gen;
	cmd.addICmdOption("-gen", &gen, 0, "Data generator (0). 0=uniform, 1=hypersphere, 2=unit guassian, 3=mix of gaussians, 4=line, 5=user specified, 6=uniform embedding, 7=unit gaussian embedding, 8=all 1.1");

	char *ptrTrainingFile = NULL;
    cmd.addSCmdOption("-trn_file", &ptrTrainingFile, "data.bin", "input training binary file");

	char *ptrTestingFile = NULL;
    cmd.addSCmdOption("-tst_file", &ptrTestingFile, "data.bin", "input testing binary file");

	int dim;
	cmd.addICmdOption("-d", &dim, 4, "Dimension of the space (default = 4).");

	int intrinsicDim;
	cmd.addICmdOption("-id", &intrinsicDim, 2, "Intrinsic dimension of the embedding data (default = 2)");

    long glb_nref;
	cmd.addLCmdOption("-glb_nref", &glb_nref, 1000, "global number of referrence data points to read (default = 1000).");

    long glb_nquery;
	cmd.addLCmdOption("-glb_nquery", &glb_nquery, 1000, "global number of referrence data points to read (default = 1000).");

	int k;
	cmd.addICmdOption("-k", &k, 2, "Number of nearest neighbors to find (2).");

	double tol_hit;
	cmd.addRCmdOption("-tol_hit", &tol_hit, 99.0, "hit rate tolerance in percentile 0 < hit < 100 (99).");

	double tol_err;
	cmd.addRCmdOption("-tol_err", &tol_err, 1e-3, "relative distance error tolerance 0 < err < 1 (1e-3).");


	int splitter_type;
	cmd.addICmdOption("-spl", &splitter_type, 0, "splitter type (0), 0:mtree, 1:maximum variance");

	int tree_dbg_out;
	cmd.addICmdOption("-dbg", &tree_dbg_out, 0, "Enable tree debugging output (0)");

	int tree_time_out;
	cmd.addICmdOption("-time", &tree_time_out, 0, "Enable tree timing output (0)");

	int max_tree_level;
	cmd.addICmdOption("-mtl", &max_tree_level, 10, "maximum tree depth (10)");

	int max_points_per_node;
	cmd.addICmdOption("-mppn", &max_points_per_node, -1, "maximum number of points per tree node ( = numof_ref_points)");

	int min_comm_size_per_node;
	cmd.addICmdOption("-mcsptn", &min_comm_size_per_node, 1, "min comm size per tree node (1)");

	int numiters;
	cmd.addICmdOption("-iter", &numiters, 5, "iterations of random rotation for greedy tree traverse (5)");

	int eval_verbose;
	cmd.addICmdOption("-eval", &eval_verbose, 0, "Evaluate results? (0). 1: check uisng logm sampled points");

	int flops_verbose;
	cmd.addICmdOption("-flops", &flops_verbose, 0, "measure flops (1) or not (*0*)");

	int tree_rotation_type;
	cmd.addICmdOption("-fr", &tree_rotation_type, 0, "rotation type (*0*). 0: do not rotate, 1: only rotate once at root, 2: rotate at every level");

	int tree_coord_type;
	cmd.addICmdOption("-fc", &tree_coord_type, 0, "kd tree coordinates choice (*0*). 0: choose randomly, 1: choose coord with max variance.");

	int super_charging_verbose;
	cmd.addICmdOption("-sc", &super_charging_verbose, 0, "do supercharing (1) or not (*0*).");

	int a2a_verbose;
	cmd.addICmdOption("-ata", &a2a_verbose, 0, "specify it is all-to-all case (1) or not(*0*)");

	int hypertree_verbose;
	cmd.addICmdOption("-ht", &hypertree_verbose, 1, "repartition points using point wise send/recv (*1*) or alltoallv (0)");

	cmd.read(argc, argv);


    long divd = glb_nref / size;
    long rem = glb_nref % size;
    long numof_ref_points = rank < rem ? (divd+1) : divd;

    divd = glb_nquery / size;
    rem = glb_nquery % size;
    long numof_query_points = rank < rem ? (divd+1) : divd;

	if(max_points_per_node == -1) max_points_per_node = 2*k;

	treeParams params;
	params.hypertree = hypertree_verbose;
	params.splitter = splitter_type;
	params.debug_verbose = tree_dbg_out;
	params.timing_verbose = tree_time_out;
	params.pruning_verbose = 1;
	params.max_points_per_node = max_points_per_node;
	params.max_tree_level = max_tree_level;
	params.min_comm_size_per_node = min_comm_size_per_node;
	params.flops_verbose = flops_verbose;
	params.eval_verbose = eval_verbose;

	// ----------- parsing cmd complete! ------------------

	double *ref, *query;
	long *refids, *queryids;
	long nglobal, mglobal;
	long refid_offset, queryid_offset;
	ref = new double[numof_ref_points*dim];
    if(!a2a_verbose) {
        query = new double[numof_query_points*dim];
	}

	double start_t, end_t;
	double max_t, min_t, avg_t;

	srand((unsigned)time(NULL)*rank);

    switch(gen) {
		case 0:
        {
			if(rank == 0) cout << "Distribution: Uniform" << endl;
			generateUniform(numof_ref_points, dim, ref, MPI_COMM_WORLD);
            if(!a2a_verbose) {
				generateUniform(numof_query_points, dim, query, MPI_COMM_WORLD);
			}
			break;
        }

		case 1:
        {
			if(rank == 0) cout << "Distribution: Hypersphere shell" << endl;
			generateUnitHypersphere(numof_ref_points, dim, ref, MPI_COMM_WORLD);
            if(!a2a_verbose) {
				generateUnitHypersphere(numof_query_points, dim, query, MPI_COMM_WORLD);
			}
			break;
        }

		case 2:
        {
			if(rank == 0) cout << "Distribution: Unit gaussian" << endl;
			generateNormal(numof_ref_points, dim, ref, MPI_COMM_WORLD);
			if(!a2a_verbose) {
				generateNormal(numof_query_points, dim, query, MPI_COMM_WORLD);
			}
			break;
        }

		case 3:
        {
			if(rank == 0) cout << "Distribution: Mixture of random gaussians" << endl;
			assert(numof_ref_points%2==0);
			generateNormal(numof_ref_points, dim, ref, MPI_COMM_WORLD);
			#pragma omp parallel for
			for(int i = 0; i < (numof_ref_points/2)*dim; i++) ref[i] *= 2.0;
			#pragma omp parallel for
			for(int i = numof_ref_points/2; i < numof_ref_points; i++) ref[i*dim] += 1.0;

            if(!a2a_verbose) {
				assert(numof_query_points%2==0);
				generateNormal(numof_query_points, dim, query, MPI_COMM_WORLD);
				#pragma omp parallel for
				for(int i = 0; i < (numof_query_points/2)*dim; i++) query[i] *= 2.0;
				#pragma omp parallel for
				for(int i = numof_query_points/2; i < numof_query_points; i++) query[i*dim] += 1.0;
			}
			break;
		}

		case 4:
        {
			if(rank == 0) cout << "Distribution: Line" << endl;
			int * dummy_refids = new int[numof_ref_points];
			genPointInRandomLine(numof_ref_points, dim, ref, dummy_refids, comm, 1, rank*numof_ref_points);
			delete [] dummy_refids;
			if(!a2a_verbose) {
				int * dummy_queryids = new int[numof_query_points];
				genPointInRandomLine(numof_query_points, dim, query, dummy_queryids, comm, 1, rank*numof_query_points);
				delete [] dummy_queryids;
			}
			break;
		}

		case 5:
        {
			if(rank == 0) cout << "Use Input Training File: " << ptrTrainingFile << endl;
			double *dummy_ref = NULL;
            int dummy_numof_ref_points;
            knn::mpi_binread(ptrTrainingFile, glb_nref, dim, dummy_numof_ref_points, dummy_ref, comm);
            assert(numof_ref_points == dummy_numof_ref_points);
            long lsize = (long)dim*(long)numof_ref_points;
            memcpy(ref, dummy_ref, sizeof(double)*lsize);
			//for(int i = 0; i < numof_ref_points*dim; i++) {
            //    ref[i] = dummy_ref[i];
            //}
            delete [] dummy_ref;

            if(!a2a_verbose) {
			    if(rank == 0) cout << "Use Input Testing File: " << ptrTestingFile << endl;
                double *dummy_query = NULL;
                int dummy_numof_query_points;
                knn::mpi_binread(ptrTestingFile, glb_nquery, dim, dummy_numof_query_points, dummy_query, comm);
                lsize = (long)dim*(long)numof_query_points;
                memcpy(query, dummy_query, sizeof(double)*lsize);
                //for(int i = 0; i < numof_query_points*dim; i++) {
                //    query[i] = dummy_query[i];
                //}
                delete [] dummy_query;
            }

			break;
		}

		case 6:
        {
			if(rank == 0) cout << "Distribution: "<<intrinsicDim<<"-d Unit Gaussian Embedding into "<<dim<<"-d space" << endl;
			if(a2a_verbose) {
				generateUniformEmbedding(numof_ref_points, dim, intrinsicDim, ref, MPI_COMM_WORLD);
			}
			else {
				delete [] ref;
				int np = numof_ref_points + numof_query_points;
				ref = new double [np*dim];
				generateUniformEmbedding(np, dim, intrinsicDim, ref, MPI_COMM_WORLD);
				double *newref = new double [numof_ref_points*dim];
				#pragma omp parallel for
				for(int i = 0; i < dim*numof_ref_points; i++)
					newref[i] = ref[i];
				#pragma omp parallel for
				for(int i = 0; i < dim*numof_query_points; i++)
					query[i] = ref[dim*numof_ref_points+i];
				delete [] ref;
				ref = newref;
			}
			break;
		}

		case 7:
        {
			if(rank == 0) cout << "Distribution: "<<intrinsicDim<<"-d Unit Gaussian Embedding into "<<dim<<"-d space" << endl;
			if(a2a_verbose) {
				generateNormalEmbedding(numof_ref_points, dim, intrinsicDim, ref, MPI_COMM_WORLD);
			}
			else {
				delete [] ref;
				int np = numof_ref_points + numof_query_points;
				ref = new double [np*dim];
				generateNormalEmbedding(np, dim, intrinsicDim, ref, MPI_COMM_WORLD);
				double *newref = new double [numof_ref_points*dim];
				#pragma omp parallel for
				for(int i = 0; i < dim*numof_ref_points; i++)
					newref[i] = ref[i];
				#pragma omp parallel for
				for(int i = 0; i < dim*numof_query_points; i++)
					query[i] = ref[dim*numof_ref_points+i];
				delete [] ref;
				ref = newref;
			}
			break;
		}

		case 8:
        {
			if(rank == 0) cout << "Distribution: all 1.1"<< endl;
			for(int i = 0; i < dim*numof_ref_points; i++)
				ref[i] = 1.1;
			if(!a2a_verbose) {
				for(int i = 0; i < dim*numof_query_points; i++)
					query[i] = 1.1;
			}
			break;
		}

		default:
		cerr << "Invalid generator selection" << endl;
		exit(1);
	}

	if(rank == 0) cout << "generate points done!" << endl;


//	if(rank == size-1) {
	    //int r = rand();
		//numof_ref_points = (int)( (double)r/(double)RAND_MAX * numof_ref_points );
//	}

	MPI_Allreduce( &numof_ref_points, &nglobal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
	MPI_Scan(&numof_ref_points, &refid_offset, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
	refid_offset -= numof_ref_points;
	refids = new long[numof_ref_points];
	for(int i = 0; i < numof_ref_points; i++) refids[i] = refid_offset + (long)i;

	if(!rank) cout << "nGlobal = " << nglobal << endl;

	if(!a2a_verbose) {
		MPI_Allreduce( &numof_query_points, &mglobal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
		MPI_Scan(&numof_query_points, &queryid_offset, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
		queryid_offset -= numof_query_points;
		queryids = new long[numof_query_points];
		for(int i = 0; i < numof_query_points; i++) queryids[i] = queryid_offset + (long)i;
	}

	if(tree_dbg_out == 3 || SC_DEBUG_VERBOSE) {
		MPI_Barrier(MPI_COMM_WORLD);
		for(int i = 0; i < numof_ref_points; i++) {
			cout<<rank<<" "<<refids[i]<<" ";
			for(int j = 0; j < dim; j++)
				cout<<ref[i*dim+j]<<" ";
			cout<<endl;
		}
		MPI_Barrier(MPI_COMM_WORLD);
		if(rank == 0) cout<<endl;
	}

	if(a2a_verbose) {
		numof_query_points = numof_ref_points;
		query = ref;
		queryids = refids;
	}

	// knn tree search
	vector<long> sampleIDs;
	vector<double> globalKdist;
	vector<long> globalKid;
	//if(eval_verbose && super_charging_verbose) {
	if(super_charging_verbose) {
		get_sample_info(ref, query, refids, queryids, numof_ref_points, numof_query_points, dim, k,
					sampleIDs, globalKdist, globalKid);
	}

	if(rank == 0) cout<<"get verify info done."<<endl;

	float mflops = -1.0, papi_real_time, papi_proc_time;
	long long flpins = 0;
	float max_mflops = -1.0, min_mflops = -1.0, avg_mflops = -1.0;

	//if(rank == 0) cout<<endl<<"*************** total iters = "<<numiters<<" ***************** "<<endl;

	// 1. copy data
	binData refData;
	refData.X.resize(dim*numof_ref_points);
	refData.gids.resize(numof_ref_points);
	refData.dim = dim;
	refData.numof_points = numof_ref_points;
	#pragma omp parallel for
	for(int i = 0; i < numof_ref_points*dim; i++) refData.X[i] = ref[i];
	#pragma omp parallel for
	for(int i = 0; i < numof_ref_points; i++) refData.gids[i] = refids[i];

	binData queryData;
	if(!a2a_verbose) {
		queryData.X.resize(dim*numof_query_points);
		queryData.gids.resize(numof_query_points);
		queryData.dim = dim;
		queryData.numof_points = numof_query_points;
		#pragma omp parallel for
		for(int i = 0; i < numof_query_points*dim; i++) queryData.X[i] = query[i];
		#pragma omp parallel for
		for(int i = 0; i < numof_query_points; i++) queryData.gids[i] = queryids[i];
	}

	delete[] ref;
	delete[] refids;

	if(!a2a_verbose) {
		delete[] query;
		delete[] queryids;
	}


	// 2. search
	vector<long> queryIDs_rkdt;
	vector< pair<double, long> > *kNN_rkdt = new vector< pair<double, long> >();

	init_papi();

	Tree_Const_T_ = 0.0;
	Tree_Search_T_ = 0.0;
	STree_Const_T_ = 0.0;
	STree_Search_T_ = 0.0;
	Direct_Kernel_T_ = 0.0;
	MPI_Collective_T_ = 0.0;
	Repartition_Tree_Build_T_ = 0.0;
	Repartition_Query_T_ = 0.0;
	Repartition_T_ = 0.0;
	Comm_Split_T_ = 0.0;
	MPI_Collective_Query_T_ = 0.0;
	MPI_Collective_Const_T_ = 0.0;
    COMPUTE_DIST_T_ = 0.0;
    MAX_HEAP_T_ = 0.0;

	if (flops_verbose == 1) papi_thread_flops(&papi_real_time, &papi_proc_time, &flpins, &mflops);

    MPI_Barrier(comm);
	start_t = omp_get_wtime();
    if(a2a_verbose == 1) {		// use all to all special case
		bintree::knnTreeSearch_RandomRotation_a2a(&refData, k,
								numiters, params, tree_rotation_type, tree_coord_type,
								queryIDs_rkdt, kNN_rkdt, tol_hit, tol_err);
    }
    else {
		bintree::knnTreeSearch_RandomRotation(&refData, &queryData, k,
									numiters, params, tree_rotation_type, tree_coord_type,
									queryIDs_rkdt, kNN_rkdt, tol_hit, tol_err);
	}
    MPI_Barrier(comm);
	end_t = omp_get_wtime() - start_t;

	if(flops_verbose == 1) {
		// Tflops
		papi_thread_flops(&papi_real_time, &papi_proc_time, &flpins, &mflops);
		MPI_Reduce(&mflops, &max_mflops, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&mflops, &min_mflops, 1, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&mflops, &avg_mflops, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
		if(rank == 0) cout<<"Summary: "<<endl;
        if(rank == 0) {
			cout<<"- TFlops: \t"<<avg_mflops / 1000000.0
				<<"  \t(min: "<<min_mflops / 1000000.0
				<<"	 max: "<<max_mflops / 1000000.0
				<<"  avg: "<<avg_mflops / (double)size / 1000000.0<<")"
				<<endl;
		}

		// const time
		double const_t = Tree_Const_T_;
		double max_const_t = 0.0;
		MPI_Reduce(&const_t, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&const_t, &min_t, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&const_t, &avg_t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		max_const_t = max_t;
		if(rank == 0) {
			cout<<"- Const. Time: "
				<<"\tmin: "<<min_t
				<<"\tmax: "<<max_t
				<<"\tavg: "<<avg_t / (double)size
				<<endl;
		}

		// overall search time
		double search_t = Tree_Search_T_;
		double max_search_t = 0.0;
		MPI_Reduce(&search_t, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&search_t, &min_t, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&search_t, &avg_t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		max_search_t = max_t;
        if(rank == 0) {
			cout<<"- Search Time: "
				<<"\tmin: "<<min_t
				<<"\tmax: "<<max_t
				<<"\tavg: "<<avg_t / (double)size
				<<endl;
		}

		// overall compute distance time
		double dist_t = COMPUTE_DIST_T_;
		MPI_Reduce(&dist_t, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&dist_t, &min_t, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&dist_t, &avg_t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if(rank == 0) {
			cout<<"- Compute Distance Time: "
				<<"\tmin: "<<min_t
				<<"\tmax: "<<max_t
				<<"\tavg: "<<avg_t / (double)size
				<<endl;
		}

		// overall search nn time
		double heap_t = MAX_HEAP_T_;
		MPI_Reduce(&heap_t, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&heap_t, &min_t, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&heap_t, &avg_t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if(rank == 0) {
			cout<<"- Search NN Time: "
				<<"\tmin: "<<min_t
				<<"\tmax: "<<max_t
				<<"\tavg: "<<avg_t / (double)size
				<<endl;
		}


		// overall time:
		if(rank == 0)
			cout<<"- Total Time: \t"<<end_t<<endl;

		MPI_Reduce(&STree_Const_T_, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&STree_Const_T_, &min_t, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&STree_Const_T_, &avg_t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		max_const_t = max_t;
		if(rank == 0) {
			cout<<" + Shared Memory Const. Time: "
				<<"\tmin: "<<min_t
				<<"\tmax: "<<max_t
				<<"\tavg: "<<avg_t / (double)size
				<<endl;
		}

		MPI_Reduce(&STree_Search_T_, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&STree_Search_T_, &min_t, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&STree_Search_T_, &avg_t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		max_const_t = max_t;
		if(rank == 0) {
			cout<<" + Shared Memory Search Time: "
				<<"\tmin: "<<min_t
				<<"\tmax: "<<max_t
				<<"\tavg: "<<avg_t / (double)size
				<<endl;
		}

		// overall repartition time in tree building
		MPI_Reduce(&Direct_Kernel_T_, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&Direct_Kernel_T_, &min_t, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&Direct_Kernel_T_, &avg_t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		if(rank == 0) {
			cout<<" + Direkct KNN Kernel Time: "
				<<"\tmin: "<<min_t
				<<"\tmax: "<<max_t
				<<"\tavg: "<<avg_t / (double)size
				<<endl;
		}

		// overall repartition time in tree building
		MPI_Reduce(&Repartition_Tree_Build_T_, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&Repartition_Tree_Build_T_, &min_t, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&Repartition_Tree_Build_T_, &avg_t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		if(rank == 0) {
			cout<<" + Repartition Time (Const.): "
				<<"\tmin: "<<min_t
				<<"\tmax: "<<max_t
				<<"\tavg: "<<avg_t / (double)size
				<<endl;
		}

		// overall repartition time in query
		MPI_Reduce(&Repartition_Query_T_, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&Repartition_Query_T_, &min_t, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&Repartition_Query_T_, &avg_t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		if(rank == 0) {
			cout<<" + Repartition Time (Query): "
				<<"\tmin: "<<min_t
				<<"\tmax: "<<max_t
				<<"\tavg: "<<avg_t / (double)size
				<<endl;
		}

		// overall repartition time  
		Repartition_T_ = Repartition_Tree_Build_T_ + Repartition_Query_T_;
		MPI_Reduce(&Repartition_T_, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&Repartition_T_, &min_t, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&Repartition_T_, &avg_t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		if(rank == 0) {
			cout<<" + Repartition Time (Overall): "
				<<"\tmin: "<<min_t
				<<"\tmax: "<<max_t
				<<"\tavg: "<<avg_t / (double)size
				<<endl;
		}
		
		// overall mpi collective time  
		MPI_Reduce(&MPI_Collective_Const_T_, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&MPI_Collective_Const_T_, &min_t, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&MPI_Collective_Const_T_, &avg_t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		if(rank == 0) {
			cout<<" + MPI Collective Time (Const): "
				<<"\tmin: "<<min_t
				<<"\tmax: "<<max_t
				<<"\tavg: "<<avg_t / (double)size
				<<endl;
		}
		
		// overall mpi collective time  
		MPI_Reduce(&MPI_Collective_Query_T_, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&MPI_Collective_Query_T_, &min_t, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&MPI_Collective_Query_T_, &avg_t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		if(rank == 0) {
			cout<<" + MPI Collective Time (Query): "
				<<"\tmin: "<<min_t
				<<"\tmax: "<<max_t
				<<"\tavg: "<<avg_t / (double)size
				<<endl;
		}
	
		// overall mpi collective time  
		double mpi_t = MPI_Collective_Const_T_ + MPI_Collective_Query_T_;
		MPI_Reduce(&mpi_t, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&mpi_t, &min_t, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&mpi_t, &avg_t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		if(rank == 0) {
			cout<<" + MPI Collective Time (Overall): "
				<<"\tmin: "<<min_t
				<<"\tmax: "<<max_t
				<<"\tavg: "<<avg_t / (double)size
				<<endl;
		}
	
		// overall comm split time  
		MPI_Reduce(&Comm_Split_T_, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&Comm_Split_T_, &min_t, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&Comm_Split_T_, &avg_t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		if(rank == 0) {
			cout<<" + Comm Splitting Time: "
				<<"\tmin: "<<min_t
				<<"\tmax: "<<max_t
				<<"\tavg: "<<avg_t / (double)size
				<<endl;
		}

	}	// end if (flops_verbose)

	//if(eval_verbose) {
	//	double hit_rate = 0.0, relative_error = 0.0;
	//	int nmiss = 0;
	//	verify(sampleIDs, globalKdist, globalKid, queryIDs_rkdt, *kNN_rkdt_a2a, nmiss, hit_rate, relative_error);
	//	if(rank == 0) cout<<"- Hit Rate ("<<sampleIDs.size()<<" samples): "<< hit_rate << "%" << endl;
	///	if(rank == 0) cout<<"- Relative Error ("<<sampleIDs.size()<<" samples): "<< relative_error << "%" << endl;
	//}

	if(rank == 0) cout<<endl;


	if(super_charging_verbose) {
			
		start_t = MPI_Wtime();
		vector< pair<double, long> > kNN_rkdt_sc;
		if (flops_verbose == 1) papi_thread_flops(&papi_real_time, &papi_proc_time, &flpins, &mflops);
		
		bintree::superCharging(&(refData.X[0]), numof_ref_points, dim, k, *kNN_rkdt, 
							kNN_rkdt_sc, MPI_COMM_WORLD);
			
		if(flops_verbose == 1) {
			papi_thread_flops(&papi_real_time, &papi_proc_time, &flpins, &mflops);
			MPI_Reduce(&mflops, &max_mflops, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
			MPI_Reduce(&mflops, &min_mflops, 1, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
			MPI_Reduce(&mflops, &avg_mflops, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
			if (rank == 0) cout<<"\n++++++ super charging (blockwise) ++++++"<<endl;
			if(rank == 0) cout<<"- Super Charging Overall Time: "<<MPI_Wtime() - start_t<<endl;
			if(rank == 0) {
				cout<<"- TFlops: "
					<<"  min: "<<min_mflops / 1000000.0
					<<"  max: "<<max_mflops / 1000000.0
					<<"  avg: "<<avg_mflops / (double)size / 1000000.0
					<<"  total: "<<avg_mflops/1000000.0
					<<endl;
			}
		}
		if(eval_verbose) {
			double hit_rate = 0.0, relative_error = 0.0;
			int nmiss = 0;
			verify(sampleIDs, globalKdist, globalKid, queryIDs_rkdt, kNN_rkdt_sc, nmiss, hit_rate, relative_error);
			if(rank == 0) cout<<"- Hit Rate ("<<sampleIDs.size()<<" samples): "<< hit_rate << "%" << endl;
			if(rank == 0) cout<<"- Relative Error ("<<sampleIDs.size()<<" samples): "<< relative_error << "%" << endl;
		}
	}	// end if (super charging)

	delete kNN_rkdt;

	MPI_Finalize();

	return 0;

}




