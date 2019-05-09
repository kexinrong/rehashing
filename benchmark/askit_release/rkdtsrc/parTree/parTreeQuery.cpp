#include <omp.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <ctime>
#include <float.h>
#include <algorithm>
#include <mpi.h>

#include "stTreeSearch.h"
#include "stTree.h"
#include "verbose.h"
#include "eval.h"
#include "ompUtils.h"
#include "lsh.h"
#include "parTree.h"
#include "rotation.h"

// if some A[i].first == B[j].first, always choose A[i], and remove B[j], the value of B might changed
void part_knn_merge(	vector< pair<double, long> > &A, vector< pair<double, long> > &B,
						int n, int k,
						vector< pair<double, long> > &result)
{
	result.resize(n*k);
	#pragma omp parallel for
	for(int i = 0; i < n; i++) {
		int aloc = i*k;
		int bloc = i*k;
		int resultloc = i*k;
		for(int j = 0; j < k; j++) {
			if( (A[aloc].second == B[bloc].second) && (bloc == (i+1)*k-1) ) B[bloc] = make_pair(DBL_MAX, -1);
			if( (A[aloc].second == B[bloc].second) && (bloc < (i+1)*k-1) ) bloc++;
			if( A[aloc] <= B[bloc] ) {
				result[resultloc++] = A[aloc++];
			}
			else {
				result[resultloc++] = B[bloc++];
			}
		}
	} // end for (i < n)
}


void part_query_greedy_a2a(pbinData refData, int k, int flag_r, 
						   int max_points, int max_tree_level,
						   vector<long>* queryIDs,
						   vector< pair<double, long> >*kNN, 
					       MPI_Comm comm)
{
	double start_t;

	int worldrank, worldsize;
	MPI_Comm_rank(comm, &worldrank);
	MPI_Comm_size(comm, &worldsize);

	int dim = refData->dim;
	int numof_ref_points = refData->X.size() / dim;
	int glb_numof_ref_points;
	MPI_CALL(MPI_Allreduce(&numof_ref_points, &glb_numof_ref_points, 1, MPI_INT, MPI_SUM, comm));
	int ppn = glb_numof_ref_points/worldsize; //Number of points per node
	int homepoints = (worldrank==worldsize-1) ? ppn+glb_numof_ref_points%ppn : ppn; //Number of query points "owned" by each process

	int totalneighbors = k*numof_ref_points;
	triple<long, double, long> *tNeighbors = new triple<long, double, long>[totalneighbors];
	long * glb_ref_ids = new long [numof_ref_points];
	#pragma omp parallel for
	for(int i = 0; i < numof_ref_points; i++)
		glb_ref_ids[i] = refData->gids[i];

	int flag_stree_r = 0;
	if(flag_r == 2) flag_stree_r = 2;
	vector< pair<double, long> >*kneighbors = new vector< pair<double, long> >;
	stTreeSearch_rkdt_a2a_me(refData, k, flag_stree_r, 
							 max_points, max_tree_level, kneighbors);
	// -> after build the tree, refData has already been deleted
	refData = NULL;

	#if STAGE_OUTPUT_VERBOSE 
		MPI_Barrier(comm);
		if(worldrank == 0) cout<<"    > Query(a2a): all proc find knn in stree done! "<<endl;
	#endif


	//Pack neighbors into array of (queryID, distance, refID) triples. 
	#pragma omp parallel for
	for(int i = 0; i < numof_ref_points; i++) {
		for(int j = 0; j < k; j++) {
			triple<long, double, long> *currNeigh;
			currNeigh = &(tNeighbors[i*k+j]);
			currNeigh->first = glb_ref_ids[i];	//refData->gids[i];
			currNeigh->second = (*kneighbors)[i*k+j].first;
			currNeigh->third = (*kneighbors)[i*k+j].second;
		}
	}

	delete kneighbors;
	delete [] glb_ref_ids;
 
	//Sort array of triples and transimit to appropriate home processes.
	if(totalneighbors > 0) 
		omp_par::merge_sort(tNeighbors, &(tNeighbors[totalneighbors]));
 

	#if COMM_TIMING_VERBOSE
		MPI_Barrier(MPI_COMM_WORLD);
		start_t = omp_get_wtime();
	#endif

	#if STAGE_OUTPUT_VERBOSE 
		if(worldrank == 0) cout<<"    > Query(a2a): pack neighbors into array done! "<<endl;
	#endif

	int *sendcounts = new int[worldsize];
	int *rcvcounts = new int[worldsize];
	int *senddisp = new int[worldsize];
	int *rcvdisp = new int[worldsize];
	for(int i = 0; i < worldsize; i++) 
		sendcounts[i] = 0;
	for(int i = 0; i < totalneighbors; i++) 
		sendcounts[ knn::lsh::idToHomeRank(tNeighbors[i].first, ppn, worldsize) ]++;
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_CALL(MPI_Alltoall(sendcounts, 1, MPI_INT, rcvcounts, 1, MPI_INT, MPI_COMM_WORLD));
	omp_par::scan(sendcounts, senddisp, worldsize);
	omp_par::scan(rcvcounts, rcvdisp, worldsize);
	int rcvneighs = rcvdisp[worldsize-1]+rcvcounts[worldsize-1];
	assert(rcvneighs == homepoints*k);
	triple<long, double, long> *homeneighbors = new triple<long, double, long>[rcvneighs];
	MPI_Datatype tripledata;
	MPI_Type_contiguous(sizeof(triple<long, double, long>), MPI_BYTE, &tripledata);
	MPI_Type_commit(&tripledata);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_CALL(MPI_Alltoallv(tNeighbors, sendcounts, senddisp, tripledata, homeneighbors,
			       rcvcounts, rcvdisp, tripledata, MPI_COMM_WORLD));
	
	delete[] tNeighbors;

	#if COMM_TIMING_VERBOSE
		Repartition_Query_T_ += omp_get_wtime() - start_t;
	#endif

	#if STAGE_OUTPUT_VERBOSE 
		if(worldrank == 0) cout<<"    > Query(a2a): repartition knn results done! "<<endl;
	#endif

	omp_par::merge_sort(homeneighbors, &(homeneighbors[rcvneighs]));
 	
	//Store results in output vectors.
	queryIDs->resize(homepoints);
	kNN->resize(k*homepoints) ;
	#pragma omp parallel for
	for(int i = 0; i < homepoints; i++) {
		(*queryIDs)[i]=homeneighbors[i*k].first;
		for( int j = 0; j < k; j ++ ) {
			(*kNN)[i*k+j].first = homeneighbors[i*k+j].second;
			(*kNN)[i*k+j].second = homeneighbors[i*k+j].third;
		}
	}

	MPI_Type_free(&tripledata);
	delete[] homeneighbors;  
	delete[] sendcounts;
	delete[] rcvcounts;
	delete[] senddisp;   
	delete[] rcvdisp;

	#if STAGE_OUTPUT_VERBOSE 
		if(worldrank == 0) cout<<"    > Query(a2a): store repartitioned results into output array done! "<<endl;
	#endif
}



void part_query_greedy(pbinData refData, pbinData queryData,
					   int k, int flag_r, int max_points, int max_tree_level,
					   vector<long>* queryIDs,
					   vector< pair<double, long> >*kNN, 
					   MPI_Comm comm)
{
	double start_t;

	int worldrank, worldsize;
	MPI_Comm_rank(comm, &worldrank);
	MPI_Comm_size(comm, &worldsize);

	int dim = refData->dim;
	int numof_ref_points = refData->X.size() / dim;
	int numof_query_points = queryData->X.size() / dim;
	
	int glb_numof_query_points;
	MPI_CALL(MPI_Allreduce(&numof_query_points, &glb_numof_query_points, 1, MPI_INT, MPI_SUM, comm));
	int ppn = glb_numof_query_points/worldsize; //Number of points per node
	int homepoints = (worldrank==worldsize-1) ? ppn+glb_numof_query_points%ppn : ppn; //Number of query points "owned" by each process

	int totalneighbors = k*numof_query_points;
	triple<long, double, long> *tNeighbors = new triple<long, double, long>[totalneighbors];

	int flag_stree_r = 0;
	if(flag_r == 2) flag_stree_r = 2;
	vector< pair<double, long> >*kneighbors = new vector< pair<double, long> >;
	if(numof_query_points > 0) {
		stTreeSearch_rkdt_me(refData, queryData, k, flag_stree_r, 
								 max_points, max_tree_level, kneighbors);
		// -> after build the tree, refData has already been deleted
		refData = NULL;
	}

	//Pack neighbors into array of (queryID, distance, refID) triples. 
	#pragma omp parallel for
	for(int i = 0; i < numof_query_points; i++) {
		for(int j = 0; j < k; j++) {
			triple<long, double, long> *currNeigh;
			currNeigh = &(tNeighbors[i*k+j]);
			currNeigh->first = queryData->gids[i];
			currNeigh->second = (*kneighbors)[i*k+j].first;
			currNeigh->third = (*kneighbors)[i*k+j].second;
		}
	}

	delete kneighbors;
 
	//Sort array of triples and transimit to appropriate home processes.
	if(totalneighbors > 0) 
		omp_par::merge_sort(tNeighbors, &(tNeighbors[totalneighbors]));
 

	#if COMM_TIMING_VERBOSE
		MPI_Barrier(MPI_COMM_WORLD);
		start_t = omp_get_wtime();
	#endif

	int *sendcounts = new int[worldsize];
	int *rcvcounts = new int[worldsize];
	int *senddisp = new int[worldsize];
	int *rcvdisp = new int[worldsize];
	for(int i = 0; i < worldsize; i++) 
		sendcounts[i] = 0;
	for(int i = 0; i < totalneighbors; i++) 
		sendcounts[ knn::lsh::idToHomeRank(tNeighbors[i].first, ppn, worldsize) ]++;
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_CALL(MPI_Alltoall(sendcounts, 1, MPI_INT, rcvcounts, 1, MPI_INT, MPI_COMM_WORLD));
	omp_par::scan(sendcounts, senddisp, worldsize);
	omp_par::scan(rcvcounts, rcvdisp, worldsize);
	int rcvneighs = rcvdisp[worldsize-1]+rcvcounts[worldsize-1];
	assert(rcvneighs == homepoints*k);
	triple<long, double, long> *homeneighbors = new triple<long, double, long>[rcvneighs];
	MPI_Datatype tripledata;
	MPI_Type_contiguous(sizeof(triple<long, double, long>), MPI_BYTE, &tripledata);
	MPI_Type_commit(&tripledata);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_CALL(MPI_Alltoallv(tNeighbors, sendcounts, senddisp, tripledata, homeneighbors,
			       rcvcounts, rcvdisp, tripledata, MPI_COMM_WORLD));
	
	delete[] tNeighbors;

	#if COMM_TIMING_VERBOSE
		Repartition_Query_T_ += omp_get_wtime() - start_t;
	#endif
  
	omp_par::merge_sort(homeneighbors, &(homeneighbors[rcvneighs]));
 	

	//Store results in output vectors.
	queryIDs->resize(homepoints);
	kNN->resize(k*homepoints) ;
	#pragma omp parallel for
	for(int i = 0; i < homepoints; i++) {
		(*queryIDs)[i]=homeneighbors[i*k].first;
		for( int j = 0; j < k; j ++ ) {
			(*kNN)[i*k+j].first = homeneighbors[i*k+j].second;
			(*kNN)[i*k+j].second = homeneighbors[i*k+j].third;
		}
	}

	MPI_Type_free(&tripledata);
	delete[] homeneighbors;  
	delete[] sendcounts;
	delete[] rcvcounts;
	delete[] senddisp;   
	delete[] rcvdisp;

}





void rkdt_a2a(pbinData refData, int k, int numof_iters,
					  int max_points, int max_tree_level, 
					  int flag_r, int flag_c,
					  // debug options
					  int verify_verbose,
					  // output
					  vector<long> &queryIDs,
					  vector< pair<double, long> >* &kNN,
					  MPI_Comm comm)
{

	double profile_t, stage_t;

	int worldrank;
	MPI_Comm_rank(comm, &worldrank);

	int dim = refData->dim;
	int numof_ref_points = refData->X.size() / dim;
	
	#if STAGE_OUTPUT_VERBOSE
		stage_t = omp_get_wtime();
	#endif

	// 0. prepare the verification info
	vector<long> sampleIDs;
	vector<double> globalKdist;
	vector<long> globalKid;
	if(verify_verbose) {
		get_sample_info( &(refData->X[0]), &(refData->X[0]), &(refData->gids[0]), &(refData->gids[0]),
						 numof_ref_points, numof_ref_points, dim, k, 
						 sampleIDs, globalKdist, globalKid);
	}

	#if STAGE_OUTPUT_VERBOSE
		if(worldrank == 0) cout<<"-> RKDT: sample verification points done ! - "<<omp_get_wtime()-stage_t<<endl;
		stage_t = omp_get_wtime();
	#endif

	for(int iter = 0; iter < numof_iters; iter++) {
		
	#if OVERALL_TREE_TIMING_VERBOSE
		//MPI_Barrier(comm);
		MPI_Barrier(MPI_COMM_WORLD);
		profile_t = omp_get_wtime();
	#endif

		// copy data
		pbinData refData_iter = new binData();
		refData_iter->Copy(refData);
		
		MPI_Collective_T_ = 0.0;

		// 1. build the tree
		pparTree tree = new parTree();
		tree->options.splitter = "rkdt";
		tree->options.flag_r = flag_r;
		tree->options.flag_c = flag_c;
		tree->build(refData_iter, max_points, max_tree_level, comm);
		// -> after build the tree, refData has already been deleted

	#if STAGE_OUTPUT_VERBOSE
		MPI_Barrier(comm);
		if(worldrank == 0) cout<<"-> RKDT: tree construction done ! - "<<omp_get_wtime()-stage_t<<endl;
		stage_t = omp_get_wtime();
	#endif

		// 1.1 repartion data
		refData_iter = new binData();
		tree->exchangeRefData(refData_iter, comm);
		int tree_depth = tree->depth;
		delete tree;

	#if OVERALL_TREE_TIMING_VERBOSE
		Tree_Const_T_ += omp_get_wtime() - profile_t;
		MPI_Collective_Const_T_ += MPI_Collective_T_;
		MPI_Barrier(comm);
		profile_t = omp_get_wtime();
	#endif

	#if STAGE_OUTPUT_VERBOSE
		//MPI_Barrier(comm);
		MPI_Barrier(MPI_COMM_WORLD);
		if(worldrank == 0) cout<<"-> RKDT: repartition ref points done ! - "<<omp_get_wtime()-stage_t<<endl;
		stage_t = omp_get_wtime();
	#endif

		MPI_Collective_T_ = 0.0;
	
		if(iter == 0) {
			part_query_greedy_a2a(refData_iter, k, flag_r, max_points,  max_tree_level-tree_depth,
								  &queryIDs, kNN, comm);
		}
		else {
			vector<long> queryIDs_iter;
			vector< pair<double, long> > *kNN_iter = new vector< pair<double, long> >;
			part_query_greedy_a2a(refData_iter, k, flag_r, max_points,  max_tree_level-tree_depth,
								  &queryIDs_iter, kNN_iter, comm);
			
			vector< pair<double, long> > *tmpkNN = new vector< pair<double, long> >;
			part_knn_merge((*kNN), (*kNN_iter), numof_ref_points, k, (*tmpkNN));

			delete kNN;
			delete kNN_iter;
			kNN = tmpkNN;
		}

	#if OVERALL_TREE_TIMING_VERBOSE
		MPI_Collective_Query_T_ += MPI_Collective_T_;
		Tree_Search_T_ += omp_get_wtime() - profile_t;
	#endif

	#if STAGE_OUTPUT_VERBOSE
		//MPI_Barrier(comm);
		MPI_Barrier(MPI_COMM_WORLD);
		if(worldrank == 0) cout<<"-> RKDT: find knn done ! - "<<omp_get_wtime()-stage_t<<endl;
		stage_t = omp_get_wtime();
	#endif

		if(verify_verbose) {
			double hit_rate = 0.0, relative_error = 0.09;
			int nmiss = 0;
			verify(sampleIDs, globalKdist, globalKid, queryIDs, *kNN, nmiss, hit_rate, relative_error);
			if(worldrank == 0) {
				cout<<"    + iter "<<iter
					<<": "<<sampleIDs.size()<<" samples -- hit rate "<< hit_rate << "%"
					<<"  relative error "<< relative_error << "%" << endl;
			}
		}

	}	// end for (iter)
}



void rkdt(pbinData refData, pbinData queryData,
		  int k, int numof_iters,
		  int max_points, int max_tree_level, 
		  int flag_r, int flag_c,
		  // debug options
		  int verify_verbose,
		  // output
		  vector<long> &queryIDs,
		  vector< pair<double, long> >* &kNN,
		  MPI_Comm comm)
{
	double stage_t;
	double profile_t;


	int worldrank;
	MPI_Comm_rank(comm, &worldrank);

	int dim = refData->dim;
	int numof_ref_points = refData->X.size() / dim;
	int numof_query_points = queryData->X.size() / dim;

	#if STAGE_OUTPUT_VERBOSE
		if(worldrank == 0) cout<<"-> RKDT: enter"<<endl;
		stage_t = omp_get_wtime();
	#endif


	// 0. prepare the verification info
	vector<long> sampleIDs;
	vector<double> globalKdist;
	vector<long> globalKid;
	if(verify_verbose) {
		get_sample_info( &(refData->X[0]), &(queryData->X[0]), &(refData->gids[0]), &(queryData->gids[0]),
						 numof_ref_points, numof_query_points, dim, k, 
						 sampleIDs, globalKdist, globalKid);
	}

	#if STAGE_OUTPUT_VERBOSE
		if(worldrank == 0) cout<<"-> RKDT: sample verification points done ! - "<<omp_get_wtime()-stage_t<<endl;
		stage_t = omp_get_wtime();
	#endif

	for(int iter = 0; iter < numof_iters; iter++) {
	
	#if OVERALL_TREE_TIMING_VERBOSE
		MPI_Barrier(comm);
		profile_t = omp_get_wtime();
	#endif
		
		MPI_Collective_T_ = 0.0;
		
		// copy data
		pbinData refData_iter = new binData();
		refData_iter->Copy(refData);
		pbinData queryData_iter = new binData();
		queryData_iter->Copy(queryData);

		// 1. build the tree
		pparTree tree = new parTree();
		tree->options.splitter = "rkdt";
		tree->options.flag_r = flag_r;
		tree->options.flag_c = flag_c;
		tree->build(refData_iter, max_points, max_tree_level, comm);
		// -> after build the tree, refData_iter has already been deleted
	
		#if STAGE_OUTPUT_VERBOSE
			if(worldrank == 0) cout<<"-> RKDT: build tree done ! - "<<omp_get_wtime()-stage_t<<endl;
			stage_t = omp_get_wtime();
		#endif

		// 1.1 repartition ref data
		refData_iter = new binData();
		tree->exchangeRefData(refData_iter, comm);
	
		#if OVERALL_TREE_TIMING_VERBOSE
			Tree_Const_T_ += omp_get_wtime() - profile_t;
			MPI_Collective_Const_T_ += MPI_Collective_T_;
			MPI_Barrier(comm);
			profile_t = omp_get_wtime();
		#endif
	
		#if STAGE_OUTPUT_VERBOSE
			if(worldrank == 0) cout<<"-> RKDT: repartition ref done ! - "<<omp_get_wtime()-stage_t<<endl;
			stage_t = omp_get_wtime();
		#endif
		
		MPI_Collective_T_ = 0.0;
		
		// 1.2 repartition query data
		tree->distributeQueryData(queryData_iter, comm);

		#if STAGE_OUTPUT_VERBOSE
			if(worldrank == 0) cout<<"-> RKDT: repartition query done ! - "<<omp_get_wtime()-stage_t<<endl;
			stage_t = omp_get_wtime();
		#endif

		int tree_depth = tree->depth;

		delete tree;
	
		if(iter == 0) {
			part_query_greedy(refData_iter, queryData_iter, k, flag_r, max_points,  max_tree_level-tree_depth,
							  &queryIDs, kNN, comm);
		}
		else {
			vector<long> queryIDs_iter;
			vector< pair<double, long> > *kNN_iter = new vector< pair<double, long> >;
			part_query_greedy(refData_iter, queryData_iter, k, flag_r, max_points,  max_tree_level-tree_depth,
							  &queryIDs_iter, kNN_iter, comm);
			
			vector< pair<double, long> > *tmpkNN = new vector< pair<double, long> >;
			part_knn_merge((*kNN), (*kNN_iter), numof_query_points, k, (*tmpkNN));

			delete kNN;
			delete kNN_iter;
			kNN = tmpkNN;
		}

		#if OVERALL_TREE_TIMING_VERBOSE
			MPI_Collective_Query_T_ += MPI_Collective_T_;
			Tree_Search_T_ += omp_get_wtime() - profile_t;
		#endif

		#if STAGE_OUTPUT_VERBOSE
			if(worldrank == 0) cout<<"-> RKDT: find knn done ! - "<<omp_get_wtime()-stage_t<<endl;
			stage_t = omp_get_wtime();
		#endif

		if(verify_verbose) {
			double hit_rate = 0.0, relative_error = 0.09;
			int nmiss = 0;
			verify(sampleIDs, globalKdist, globalKid, queryIDs, *kNN, nmiss, hit_rate, relative_error);
			if(worldrank == 0) {
				cout<<"    + iter "<<iter
					<<": "<<sampleIDs.size()<<" samples -- hit rate "<< hit_rate << "%"
					<<"  relative error "<< relative_error << "%" << endl;
			}
		}

		delete queryData_iter;

	}	// end for (iter)
}



