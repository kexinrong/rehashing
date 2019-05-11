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

	CmdLine cmd;
	const char *phelp = "Help";
    cmd.addInfo(phelp);

	char *ptrInputRefFile = NULL;
    cmd.addSCmdOption("-ref_file", &ptrInputRefFile, "ref.txt", "input file storing reference points (ascii or binary)");

	char *ptrInputQueryFile = NULL;
    cmd.addSCmdOption("-query_file", &ptrInputQueryFile, "query.txt", "input file storing query points (ascii or binary)");

    char *ptrOutputFile = NULL;
    cmd.addSCmdOption("-knn_file", &ptrOutputFile, "knn_results.txt", "output ascii file storing knn results, each row is the result for one query point in a form [query_id knn_id dist^2 knn_id dist^2 ...]");

    bool bl_all2all;
    cmd.addBCmdOption("-search_all2all", &bl_all2all, false, "if query and reference points are the same, specify -search_all2all");

	int nref;
	cmd.addICmdOption("-nref", &nref, 100, "Number of referrence data points per process to read (default = 1000).");

	int nquery;
	cmd.addICmdOption("-nquery", &nquery, 100, "Number of query data points per process (default = 1000).");

	int k;
	cmd.addICmdOption("-k", &k, 2, "Number of nearest neighbors to find (2).");

	int max_tree_level;
	cmd.addICmdOption("-mtl", &max_tree_level, 10, "maximum tree depth (10)");

	int max_points_per_node;
	cmd.addICmdOption("-mppn", &max_points_per_node, -1, "maximum number of points per tree node ( = numof_ref_points)");

	int numiters;
	cmd.addICmdOption("-iter", &numiters, 5, "iterations of random rotation for greedy tree traverse (5)");

    bool bl_eval;
    cmd.addBCmdOption("-eval", &bl_eval, false, "evaluate results using log(nquery) sampled points");

	cmd.read(argc, argv);

	if(rank == 0) cout<<"nproc = "<<size<<endl;

    int splitter_type = 0;
	int tree_dbg_out = 0;
	int tree_time_out = 0;
	int min_comm_size_per_node = 1;
	int flops_verbose = 0;
	int tree_rotation_type = 1;
	int tree_coord_type = 0;
	int super_charging_verbose = 0;
	int hypertree_verbose = 1;

    int eval_verbose = bl_eval;
    long numof_ref_points = nref;
    long numof_query_points = nquery;
	if(max_points_per_node == -1) max_points_per_node = numof_ref_points;

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

	double start_t, end_t;
	double max_t, min_t, avg_t;

	srand((unsigned)time(NULL)*rank);

    // ---------- read data ------------------

    int dim;

	double *ref = NULL, *query = NULL;
	long *refids = NULL, *queryids = NULL;
	long nglobal, mglobal;
	long refid_offset, queryid_offset;

	long glb_numof_ref_points = (long)numof_ref_points*size;
    knn::mpi_dlmread(ptrInputRefFile, glb_numof_ref_points, dim, ref, comm, false);
    numof_ref_points = glb_numof_ref_points;
    if(rank == 0) cout << "read ref input: " << ptrInputRefFile << endl;

	MPI_Allreduce( &numof_ref_points, &nglobal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
	MPI_Scan(&numof_ref_points, &refid_offset, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
	refid_offset -= numof_ref_points;
	refids = new long[numof_ref_points];
	for(int i = 0; i < numof_ref_points; i++) refids[i] = refid_offset + (long)i;

    binData refData;
	refData.X.resize(dim*numof_ref_points);
	refData.gids.resize(numof_ref_points);
	refData.dim = dim;
	refData.numof_points = numof_ref_points;
	#pragma omp parallel for
	for(int i = 0; i < numof_ref_points*dim; i++) refData.X[i] = ref[i];
	#pragma omp parallel for
	for(int i = 0; i < numof_ref_points; i++) refData.gids[i] = refids[i];
    delete[] ref;
	delete[] refids;

    numof_query_points = numof_ref_points;
	binData queryData;
    if(!bl_all2all) {
        long glb_numof_query_points = (long)numof_query_points*size;
        knn::mpi_dlmread(ptrInputQueryFile, glb_numof_query_points, dim, query, comm, false);
        numof_query_points = glb_numof_query_points;
        if(rank == 0) cout << "read query input: " << ptrInputQueryFile << endl;

        MPI_Allreduce( &numof_query_points, &mglobal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
		MPI_Scan(&numof_query_points, &queryid_offset, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
		queryid_offset -= numof_query_points;
		queryids = new long[numof_query_points];
		for(int i = 0; i < numof_query_points; i++) queryids[i] = queryid_offset + (long)i;

		queryData.X.resize(dim*numof_query_points);
		queryData.gids.resize(numof_query_points);
		queryData.dim = dim;
		queryData.numof_points = numof_query_points;
		#pragma omp parallel for
		for(int i = 0; i < numof_query_points*dim; i++) queryData.X[i] = query[i];
		#pragma omp parallel for
		for(int i = 0; i < numof_query_points; i++) queryData.gids[i] = queryids[i];
		delete[] query;
		delete[] queryids;
    }


    // ---------- read data ------------------

    // .1 get sample points to check results
    vector<long> sampleIDs;
	vector<double> globalKdist;
	vector<long> globalKid;
    if(eval_verbose && super_charging_verbose) {
		get_sample_info(ref, query, refids, queryids, numof_ref_points, numof_query_points, dim, k,
					sampleIDs, globalKdist, globalKid);
	    if(rank == 0) cout<<"get verify info done."<<endl;
	}


    // .2 search
	float mflops = -1.0, papi_real_time, papi_proc_time;
	long long flpins = 0;
	float max_mflops = -1.0, min_mflops = -1.0, avg_mflops = -1.0;
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

    if(rank == 0) cout<<endl<<"*************** total iters = "<<numiters<<" ***************** "<<endl;

	init_papi();
	if (flops_verbose == 1) papi_thread_flops(&papi_real_time, &papi_proc_time, &flpins, &mflops);

	start_t = omp_get_wtime();

    vector<long> queryIDs_rkdt;
	vector< pair<double, long> > *kNN_rkdt = new vector< pair<double, long> >();

    if(bl_all2all) {		// use all to all special case
		bintree::knnTreeSearch_RandomRotation_a2a(&refData, k,
								numiters, params, tree_rotation_type, tree_coord_type,
								queryIDs_rkdt, kNN_rkdt);
	}
    else {
		bintree::knnTreeSearch_RandomRotation(&refData, &queryData, k,
									numiters, params, tree_rotation_type, tree_coord_type,
									queryIDs_rkdt, kNN_rkdt);
	}
	end_t = omp_get_wtime() - start_t;

    // ---------- save results ------------------
    if(0 == rank) remove(ptrOutputFile);
    MPI_Barrier(comm);

    ofstream output;
    for(int r = 0; r < size; r++) {
        if(rank == r) {
            output.open(ptrOutputFile, ios::app|ios::out);
            for(int i = 0; i < numof_query_points; i++) {
                output<<queryIDs_rkdt[i]<<" ";
                for(int j = 0; j < k; j++) {
                    output<<(*kNN_rkdt)[i*k+j].second<<" "<<(*kNN_rkdt)[i*k+j].first<<" ";
                }
                output<<endl;
            }
            output.flush();
            output.close();
        }
        MPI_Barrier(comm);
    }


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
		double const_t = Tree_Const_T_ + STree_Const_T_;
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
		double search_t = Tree_Search_T_ - STree_Const_T_;
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

		// overall time:
		if(rank == 0) 
			cout<<"- Total Time: \t"<<max_const_t + max_search_t<<endl;

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




