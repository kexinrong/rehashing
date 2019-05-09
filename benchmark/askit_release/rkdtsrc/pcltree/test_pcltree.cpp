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
   
	if(rank == 0) cout<<"pcl tree starts ... np = "<<size<<endl;
  
	CmdLine cmd;
	const char *phelp = "Help";
    cmd.addInfo(phelp);

	char *ptrInputFile = NULL;
    cmd.addSCmdOption("-file", &ptrInputFile, "data.bin", "input binary file storing points");
    
	int dim;
	cmd.addICmdOption("-d", &dim, 4, "Dimension of the space (default = 4).");
    
	int intrinsicDim;
	cmd.addICmdOption("-id", &intrinsicDim, 2, "Intrinsic dimension of the embedding data (default = 2)");

	int rn;
	cmd.addICmdOption("-rn", &rn, 1000, "Number of referrence data points per process (default = 1000).");
	
	int qn;
	cmd.addICmdOption("-qn", &qn, 1000, "Number of query data points per process (default = 1000).");
    
	int k;
	cmd.addICmdOption("-k", &k, 2, "Number of nearest neighbors to find (2).");
    
	int gen;
	cmd.addICmdOption("-gen", &gen, 0, "Data generator (0). 0=uniform, 1=hypersphere, 2=unit guassian, 3=mix of gaussians, 4=line, 5=user specified, 6=uniform embedding, 7=unit gaussian embedding");
   	
	int seed_type;
	cmd.addICmdOption("-sd", &seed_type, 0, "seeding method for clustering: ostrovskey (1) or random (*0*)");
 
	int max_tree_level;
	cmd.addICmdOption("-mtl", &max_tree_level, 10, "maximum tree depth (10)");
    
	int max_points_per_node;
	cmd.addICmdOption("-mppn", &max_points_per_node, -1, "maximum number of points per tree node ( = numof_ref_points)");
    
	int min_comm_size_per_node;
	cmd.addICmdOption("-mcsptn", &min_comm_size_per_node, 1, "min comm size per tree node (1)");
   	
	int eval_verbose;
	cmd.addICmdOption("-eval", &eval_verbose, 0, "Evaluate results? (0). 1: check uisng logm sampled points");
    
	int flops_verbose;
	cmd.addICmdOption("-flops", &flops_verbose, 0, "measure flops (1) or not (*0*)");
    
	int a2a_verbose;
	cmd.addICmdOption("-ata", &a2a_verbose, 0, "specify it is all-to-all case (1) or not(*0*)");
	
	int cluster_factor;
	cmd.addICmdOption("-cf", &cluster_factor, 1, "numof clusters used to group points (*1*). numof_clusters = 2*cf");
 

	cmd.read(argc, argv);

    long numof_ref_points = rn;
    long numof_query_points = qn;
	if(max_points_per_node == -1) max_points_per_node = numof_ref_points;

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
			if(rank == 0) cout << "Distribution: Uniform" << endl;
			generateUniform(numof_ref_points, dim, ref, MPI_COMM_WORLD);
			if(!a2a_verbose) {
				generateUniform(numof_query_points, dim, query, MPI_COMM_WORLD);
			}
			break;

		case 1:
			if(rank == 0) cout << "Distribution: Hypersphere shell" << endl;
			generateUnitHypersphere(numof_ref_points, dim, ref, MPI_COMM_WORLD);
			if(!a2a_verbose) {
				generateUnitHypersphere(numof_query_points, dim, query, MPI_COMM_WORLD);
			}
			break;

		case 2:
			if(rank == 0) cout << "Distribution: Unit gaussian" << endl;
			generateNormal(numof_ref_points, dim, ref, MPI_COMM_WORLD);
			if(!a2a_verbose) {
				generateNormal(numof_query_points, dim, query, MPI_COMM_WORLD);
			}
			break;
     
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
			genPointInRandomLine(numof_ref_points, dim, ref, dummy_refids, MPI_COMM_WORLD, 1, rank*numof_ref_points);
			delete [] dummy_refids;
			if(!a2a_verbose) {
				int * dummy_queryids = new int[numof_query_points];
				genPointInRandomLine(numof_query_points, dim, query, dummy_queryids, MPI_COMM_WORLD, 1, rank*numof_query_points);
				delete [] dummy_queryids;
			}
			break;
		}
	 
		case 5:
		{
			string strInputFile = ptrInputFile;
			if(rank == 0) cout << "User Input File: " << strInputFile << endl;
			double *dummy_points;
			if(a2a_verbose) {
				long total_numof_input_points = numof_ref_points*size;
				knn::parallelIO(strInputFile, total_numof_input_points, dim, dummy_points, MPI_COMM_WORLD);
				for(int i = 0; i < total_numof_input_points*dim; i++) ref[i] = dummy_points[i];
			}
			else {
				long total_numof_input_points = (numof_ref_points+numof_query_points)*size;
				knn::parallelIO(strInputFile, total_numof_input_points, dim, dummy_points, MPI_COMM_WORLD);
				assert(numof_ref_points+numof_query_points == total_numof_input_points);
				for(int i = 0; i < numof_ref_points*dim; i++) ref[i] = dummy_points[i];
				for(int i = 0; i < numof_query_points*dim; i++) query[i] = dummy_points[numof_ref_points*dim+i];
			}
			delete [] dummy_points;
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
		
		default:
		cerr << "Invalid generator selection" << endl;
		exit(1);
	}

	if(rank == 0) cout << "generate points done!" << endl;
	
	MPI_Allreduce( &numof_ref_points, &nglobal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
	MPI_Scan(&numof_ref_points, &refid_offset, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
	refid_offset -= numof_ref_points;
	refids = new long[numof_ref_points];
	for(int i = 0; i < numof_ref_points; i++) refids[i] = refid_offset + (long)i;

	if(!a2a_verbose) {
		MPI_Allreduce( &numof_query_points, &mglobal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
		MPI_Scan(&numof_query_points, &queryid_offset, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
		queryid_offset -= numof_query_points;
		queryids = new long[numof_query_points];
		for(int i = 0; i < numof_query_points; i++) queryids[i] = queryid_offset + (long)i;
	}
	
	if(a2a_verbose) {
		numof_query_points = numof_ref_points;
		query = ref;
		queryids = refids;
	}

	// knn tree search
	float mflops = -1.0, papi_real_time, papi_proc_time;
	long long flpins = 0;
	float max_mflops = -1.0, min_mflops = -1.0, avg_mflops = -1.0;
	
	// 1. copy data
	MTData refData;
	refData.X.resize(dim*numof_ref_points);
	refData.gids.resize(numof_ref_points);
	refData.dim = dim;
	//refData.numof_points = numof_ref_points;
	#pragma omp parallel for
	for(int i = 0; i < numof_ref_points*dim; i++) refData.X[i] = ref[i];
	#pragma omp parallel for
	for(int i = 0; i < numof_ref_points; i++) refData.gids[i] = refids[i];

	MTData queryData;
	if(!a2a_verbose) {
		queryData.X.resize(dim*numof_query_points);
		queryData.gids.resize(numof_query_points);
		queryData.dim = dim;
		//queryData.numof_points = numof_query_points;
		#pragma omp parallel for
		for(int i = 0; i < numof_query_points*dim; i++) queryData.X[i] = query[i];
		#pragma omp parallel for
		for(int i = 0; i < numof_query_points; i++) queryData.gids[i] = queryids[i];
	}
	else {
		numof_query_points = numof_ref_points;
		queryData.X.resize(dim*numof_ref_points);
		queryData.gids.resize(numof_ref_points);
		queryData.dim = dim;
		//queryData.numof_points = numof_ref_points;
		#pragma omp parallel for
		for(int i = 0; i < numof_ref_points*dim; i++) queryData.X[i] = ref[i];
		#pragma omp parallel for
		for(int i = 0; i < numof_ref_points; i++) queryData.gids[i] = refids[i];
	}

	delete[] ref;
	delete[] refids;
	
	if(!a2a_verbose) {
		delete[] query;
		delete[] queryids;
	}
 
	// correctness
	vector<long> sampleIDs;
	vector<double> globalKdist;
	vector<long> globalKid;
	if(eval_verbose) {
		get_sample_info(&(refData.X[0]), &(queryData.X[0]), &(refData.gids[0]), &(queryData.gids[0]),
						numof_ref_points, numof_query_points, dim, k, 
						sampleIDs, globalKdist, globalKid);
	}

	// 2. search
	
	Tree_Const_T_ = 0.0;
	Tree_Search_T_ = 0.0;
	Direct_Kernel_T_ = 0.0;
	MPI_Collective_T_ = 0.0;
	Repartition_Tree_Build_T_ = 0.0;
	Repartition_Query_T_ = 0.0;
	Repartition_T_ = 0.0;
	Comm_Split_T_ = 0.0;
	MPI_Collective_Query_T_ = 0.0;
	MPI_Collective_Const_T_ = 0.0;


	init_papi();
	if (flops_verbose == 1) papi_thread_flops(&papi_real_time, &papi_proc_time, &flpins, &mflops);
	
	MPI_Barrier(MPI_COMM_WORLD);
	start_t = omp_get_wtime();
	MTNode root;
	root.options.pruning_verbose=true;
	root.options.cluster_factor=cluster_factor;
	root.Insert(NULL, max_points_per_node, max_tree_level, min_comm_size_per_node, MPI_COMM_WORLD, &refData, seed_type);
	Tree_Const_T_ += omp_get_wtime() - start_t;

	MPI_Barrier(MPI_COMM_WORLD);
	start_t = omp_get_wtime();
	vector<long> *queryIDs = new vector<long>();
	vector< pair<double, long> > *kNN = new vector< pair<double, long> >();
	queryK(&queryData, numof_ref_points*size, 16.0, &root, k, queryIDs, kNN);
	Tree_Search_T_ += omp_get_wtime() - start_t;

	if(eval_verbose == 1) {
		double hit_rate = 0.0, error = 0.0;
		int nmiss = 0;
		verify(sampleIDs, globalKdist, globalKid, *queryIDs, *kNN, nmiss, hit_rate, error);
		if(rank == 0) {
			cout<<" - Accuracy: "
				<<"\tnsamples: "<<sampleIDs.size()
				<<"\thit rate: "<<hit_rate
				<<"\trelative error: "<<error
				<<"\tnmiss: "<<nmiss
				<<endl;
		}
	}

	delete queryIDs;
	delete kNN;
	//delete root;

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
	

	MPI_Finalize();
  
	return 0;

}




