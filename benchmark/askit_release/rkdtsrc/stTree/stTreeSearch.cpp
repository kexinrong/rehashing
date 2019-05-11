#include <omp.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <ctime>
#include <float.h>
#include <algorithm>

#include "stTreeSearch.h"
#include "stTree.h"
#include "verbose.h"
#include "rotation.h"
#include "srkdt.h"
#include "eval.h"

#if LOAD_BALANCE_VERBOSE
	#include <mpi.h>
#endif

#define _SHARED_TREE_DEBUG_ 0
#define _SHARED_TREE_TIMING_ false


bool less_first(const pair<double, long> &a, const pair<double, long> &b)
{
    return (a.first < b.first);
}


bool less_second(const pair<double, long> &a, const pair<double, long> &b)
{
    return (a.second < b.second);
}


bool equal_second(const pair<double, long> &a, const pair<double, long> &b)
{
    return (a.second == b.second);
}


void single_merge(const pair<double, long> *A, const pair<double, long> *B, int k,
                    pair<double, long> *C, vector< pair<double, long> > &auxVec)
{
    if(auxVec.size() != 2*k)
        auxVec.resize(2*k);

    for(int i = 0; i < k; i++) auxVec[i] = A[i];
    for(int i = 0; i < k; i++) auxVec[k+i] = B[i];

    sort(auxVec.begin(), auxVec.end(), less_second);
    vector< pair<double, long> >::iterator it = unique(auxVec.begin(), auxVec.end(), equal_second);
    sort(auxVec.begin(), it, less_first);

    for(int i = 0; i < k; i++) C[i] = auxVec[i];
}


void knn_merge( vector< pair<double, long> > &A, vector< pair<double, long> > &B,
				int n, int k,
				vector< pair<double, long> > &result)
{
	result.resize(n*k);
	#pragma omp parallel
    {
        vector< pair<double, long> > auxVec(2*k);
        #pragma omp for
        for(int i = 0; i < n; i++) {
            single_merge( &(A[i*k]), &(B[i*k]), k, &(result[i*k]), auxVec);
        }
    }
}




//
// ======== specially optimized for the distribute tree, use less memory  ======== //
// =======  after call it, the data stored in the leaf->data will be deleted ===== //

// random rotation kd tree, special for all-to-all case, memory efficient
void stTreeSearch_rkdt_a2a_me( pbinData refData, int k, int flag_r,
							   int max_points, int max_tree_level,
								//output
								vector< pair<double, long> >* &kNN)
{
	double start_t = omp_get_wtime();

	int dim = refData->dim;
	int numof_ref_points = refData->X.size() / dim;

	start_t = omp_get_wtime();

	refData->lids.resize(numof_ref_points);
	#pragma omp parallel for
	for(int i = 0; i < numof_ref_points; i++)
		refData->lids[i] = (long)i;

	// 1. build the tree (splitter = rkdt, random kd tree)
	pstTree tree = new stTree();
	tree->options.splitter = "rkdt";
	tree->options.timing_verbose = 0;
	tree->options.debug_verbose = 0;
	tree->options.flag_r = flag_r;
	tree->build(refData, max_points, max_tree_level);
	// -> after build the tree, "refData" has already been deleted
	refData = NULL;

	#if OVERALL_TREE_TIMING_VERBOSE
		STree_Const_T_ += omp_get_wtime() - start_t;
	#endif

	// 2. search the tree
	start_t = omp_get_wtime();

	tree->queryGreedy_a2a(k, *kNN);

	delete tree;

	#if OVERALL_TREE_TIMING_VERBOSE
		STree_Search_T_ += omp_get_wtime() - start_t;
	#endif

}


// random rotation kd tree
void stTreeSearch_rkdt_me( pbinData refData, pbinData queryData,
						int k, int flag_r,
						int max_points, int max_tree_level,
						//output
						vector< pair<double, long> >* &kNN)
{
	double start_t = omp_get_wtime();

	int dim = refData->dim;
	int numof_ref_points = refData->X.size() / dim;
	int numof_query_points = queryData->X.size() / dim;

#if LOAD_BALANCE_VERBOSE
	int worldrank, worldsize;
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);

	int max_nq, max_nr, min_nq, min_nr, avg_nq, avg_nr;
	MPI_Reduce(&numof_ref_points, &max_nr, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&numof_ref_points, &min_nr, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&numof_ref_points, &avg_nr, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	MPI_Reduce(&numof_query_points, &max_nq, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&numof_query_points, &min_nq, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&numof_query_points, &avg_nq, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	if(worldrank == 0) {
		cout<<"      -- numof_ref_points (per proc): "
			<<"  min: "<<min_nr
			<<"  max: "<<max_nr
			<<"  avg: "<<avg_nr / (double)worldsize
			<<endl;
		cout<<"      -- numof_query_points (per proc): "
			<<"  min: "<<min_nq
			<<"  max: "<<max_nq
			<<"  avg: "<<avg_nq / (double)worldsize
			<<endl;
		}
#endif

	start_t = omp_get_wtime();

	// 1. build the tree (splitter = rkdt, random kd tree)
	pstTree tree = new stTree();
	tree->options.splitter = "rkdt";
	tree->options.timing_verbose = 0;
	tree->options.debug_verbose = 0;
	tree->options.flag_r = flag_r;
	tree->build(refData, max_points, max_tree_level);
	// -> after build the tree, "refData" has already been deleted
	refData = NULL;

//cout<<"\t\t\t(shared) build tree "<<omp_get_wtime()-start_t<<endl;

	#if OVERALL_TREE_TIMING_VERBOSE
		STree_Const_T_ += omp_get_wtime() - start_t;
	#endif

	// 2. search the tree
	start_t = omp_get_wtime();
	tree->queryGreedy(queryData, k, *kNN);
	delete tree;

//cout<<"\t\t\t(shared) search tree "<<omp_get_wtime()-start_t<<endl;

	#if OVERALL_TREE_TIMING_VERBOSE
		STree_Search_T_ += omp_get_wtime() - start_t;
	#endif

}




//
// ======== standalone shared search function ======== //
//

// random rotation kd tree, special for all-to-all case
void stTreeSearch_rkdt_a2a( double *ref, double *query,
						long *refids, long *queryids,
						int numof_ref_points, int numof_query_points, int dim,
						int max_points, int max_tree_level,
						int k, int numof_iterations,
						int flag_r,
						//output
						vector< pair<double, long> >* &kNN)
{
	double start_t = omp_get_wtime();

	// 0.II copy query data
	//pbinData queryData = new binData();
	//queryData->X.resize(numof_query_points*dim);
	//queryData->gids.resize(numof_query_points);
	//queryData->dim = dim;
	//#pragma omp parallel for
	//for(int i = 0; i < numof_query_points; i++) {
	//	for(int j = 0; j < dim; j++)
	//		queryData->X[i*dim+j] = query[i*dim+j];
	//	queryData->gids[i] = queryids[i];
	//}
	
#if LOAD_BALANCE_VERBOSE
	int worldrank, worldsize;
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);

	int max_nq, max_nr, min_nq, min_nr, avg_nq, avg_nr;
	MPI_Reduce(&numof_ref_points, &max_nr, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&numof_ref_points, &min_nr, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&numof_ref_points, &avg_nr, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	
	MPI_Reduce(&numof_query_points, &max_nq, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&numof_query_points, &min_nq, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&numof_query_points, &avg_nq, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	if(worldrank == 0) {
		cout<<"      -- numof_ref_points (per proc): "
			<<"  min: "<<min_nr
			<<"  max: "<<max_nr
			<<"  avg: "<<avg_nr / (double)worldsize
			<<endl;
		//cout<<"    + numof_query_points (per proc): "
		//	<<"  min: "<<min_nq
		//	<<"  max: "<<max_nq
		//	<<"  avg: "<<avg_nq / (double)worldsize
		//	<<endl;
		}
#endif


	for(int iter = 0; iter < numof_iterations; iter++) {
		
		start_t = omp_get_wtime();

		// 0.I copy ref data
		pbinData refData = new binData();
		refData->X.resize(numof_ref_points*dim);
		refData->gids.resize(numof_ref_points);
		refData->lids.resize(numof_ref_points);
		refData->dim = dim;
		#pragma omp parallel for
		for(int i = 0; i < numof_ref_points; i++) {
			for(int j = 0; j < dim; j++)
				refData->X[i*dim+j] = ref[i*dim+j];
			refData->gids[i] = refids[i];
			refData->lids[i] = (long)i;
		}
	
		// 1. build the tree (splitter = rkdt, random kd tree)
		pstTree tree = new stTree();
		tree->options.splitter = "rkdt";
		tree->options.timing_verbose = 0;
		tree->options.debug_verbose = 0;
		tree->options.flag_r = flag_r;
		tree->build(refData, max_points, max_tree_level);
		// -> after build the tree, "refData" has already been deleted
		refData = NULL;
		if(tree->options.timing_verbose) 
				cout<<" tree build ("<<iter<<") time: "<<omp_get_wtime() - start_t<<endl<<endl;


		#if OVERALL_TREE_TIMING_VERBOSE
			STree_Const_T_ += omp_get_wtime() - start_t;
		#endif

		// 2. search the tree
		
		start_t = omp_get_wtime();
	
		if(iter == 0) {
			// - 2.1 use nearest traverse strategy
			tree->queryGreedy_a2a(k, *kNN);
			
			if(tree->options.timing_verbose) 
				cout<<" greedy a2a ("<<iter<<") search time: "<<omp_get_wtime() - start_t<<endl<<endl;
		}
		else {
			// - 2.2 iterate several times
			vector< pair<double, long> > kNN_iter;
			tree->queryGreedy_a2a(k, kNN_iter);
			
			vector< pair<double, long> > *tmpkNN = new vector< pair<double, long> >;
			knn_merge(*kNN, kNN_iter, numof_query_points, k, *tmpkNN);

			delete kNN;
            kNN = tmpkNN;
			
			if(tree->options.timing_verbose) 
				cout<<" greedy a2a ("<<iter<<") search time: "<<omp_get_wtime() - start_t<<endl<<endl;

		} // end else
	
		delete tree;

		#if OVERALL_TREE_TIMING_VERBOSE
			STree_Search_T_ += omp_get_wtime() - start_t;
		#endif

	}  // end for(iter < numof_iterations)
	
	//delete queryData;

}



// random rotation kd tree
void stTreeSearch_rkdt( double *ref, double *query,
						long *refids, long *queryids,
						int numof_ref_points, int numof_query_points, int dim,
						int max_points, int max_tree_level,
						int k, int numof_iterations,
						int flag_r,
						//output
						vector< pair<double, long> >* &kNN)
{
	double start_t = omp_get_wtime();


/*
#if LOAD_BALANCE_VERBOSE
	int worldrank, worldsize;
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);

	int max_nq, max_nr, min_nq, min_nr, avg_nq, avg_nr;
	MPI_Reduce(&numof_ref_points, &max_nr, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&numof_ref_points, &min_nr, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&numof_ref_points, &avg_nr, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	MPI_Reduce(&numof_query_points, &max_nq, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&numof_query_points, &min_nq, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&numof_query_points, &avg_nq, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	if(worldrank == 0) {
		cout<<"      -- numof_ref_points (per proc): "
			<<"  min: "<<min_nr
			<<"  max: "<<max_nr
			<<"  avg: "<<avg_nr / (double)worldsize
			<<endl;
		cout<<"      -- numof_query_points (per proc): "
			<<"  min: "<<min_nq
			<<"  max: "<<max_nq
			<<"  avg: "<<avg_nq / (double)worldsize
			<<endl;
		}
#endif
*/

//cout<<"\t\t\t(shared) copy query data, niter =  "<<numof_iterations<<", k = "<<k<<", time "<<omp_get_wtime()-start_t<<endl;

    for(int iter = 0; iter < numof_iterations; iter++) {

		start_t = omp_get_wtime();

		// 0.I copy ref data
		pbinData refData = new binData();
		refData->X.resize(numof_ref_points*dim);
		refData->gids.resize(numof_ref_points);
		refData->dim = dim;
        refData->numof_points = numof_ref_points;
		#pragma omp parallel for
		for(int i = 0; i < numof_ref_points; i++) {
            memcpy(&(refData->X[i*dim]), ref+i*dim, sizeof(double)*dim);
			refData->gids[i] = refids[i];
		}

	    // 0.II copy query data
	    pbinData queryData = new binData();
	    queryData->X.resize(numof_query_points*dim);
	    queryData->gids.resize(numof_query_points);
	    queryData->dim = dim;
        queryData->numof_points = numof_query_points;
	    #pragma omp parallel for
	    for(int i = 0; i < numof_query_points; i++) {
            memcpy( &(queryData->X[i*dim]), query+i*dim, sizeof(double)*dim);
		    queryData->gids[i] = queryids[i];
	    }

//cout<<"\t\t\t(shared) copy ref data, iter =  "<<iter<<", time "<<omp_get_wtime()-start_t<<endl;

        #if _SHARED_TREE_DEBUG_
            cout<<"\t (share): ref data"<<endl;
            for(int i = 0; i < refData->numof_points; i++) {
                cout<<"\t\t "<<refData->gids[i]<<" ";
                for(int j = 0; j < dim; j++)
                    cout<<refData->X[i*dim+j]<<" ";
                cout<<endl;
            }
            cout<<"\t (share): query data"<<endl;
            for(int i = 0; i < queryData->numof_points; i++) {
                cout<<"\t\t "<<queryData->gids[i]<<" ";
                for(int j = 0; j < dim; j++)
                    cout<<queryData->X[i*dim+j]<<" ";
                cout<<endl;
            }
        #endif


        // 1. build the tree (splitter = rkdt, random kd tree)
		pstTree tree = new stTree();
		tree->options.splitter = "rkdt";
		tree->options.timing_verbose = 0;
		tree->options.debug_verbose = 0;
		tree->options.flag_r = flag_r;
		tree->build(refData, max_points, max_tree_level);
		// -> after build the tree, "refData" has already been deleted
		if(tree->options.timing_verbose)
				cout<<" tree build ("<<iter<<") time: "<<omp_get_wtime() - start_t<<endl<<endl;

//cout<<"\t\t\t(shared) build tree, iter =  "<<iter<<", time "<<omp_get_wtime()-start_t<<endl;

        #if OVERALL_TREE_TIMING_VERBOSE
			STree_Const_T_ += omp_get_wtime() - start_t;
		#endif

        //cout<<"\t\tshtree iter "<<iter<<": build tree done"<<endl;


        // 2. search the tree
		start_t = omp_get_wtime();

		if(iter == 0) {
			// - 2.1 use nearest traverse strategy
			tree->queryGreedy(queryData, k, *kNN);
            //cout<<"\t\tshtree iter "<<iter<<": query greedy done"<<endl;
            if(tree->options.timing_verbose)
				cout<<" greedy standard ("<<iter<<") search time: "<<omp_get_wtime() - start_t<<endl<<endl;
		}
		else {
			// - 2.2 iterate several times
			vector< pair<double, long> > kNN_iter;
			tree->queryGreedy(queryData, k, kNN_iter);

            //cout<<"\t\tshtree iter "<<iter<<": query greedy done"<<endl;

            vector< pair<double, long> > *tmpkNN = new vector< pair<double, long> >;
			knn_merge(*kNN, kNN_iter, numof_query_points, k, *tmpkNN);

            //cout<<"\t\tshtree iter "<<iter<<": query merge done"<<endl;

            delete kNN;
            kNN = tmpkNN;

			if(tree->options.timing_verbose)
				cout<<" greedy standard ("<<iter<<") search time: "<<omp_get_wtime() - start_t<<endl<<endl;
		} // end else
		delete tree;
        delete queryData;
//cout<<"\t\t\t(shared) search tree, iter =  "<<iter<<", time "<<omp_get_wtime()-start_t<<endl;

        #if OVERALL_TREE_TIMING_VERBOSE
			STree_Search_T_ += omp_get_wtime() - start_t;
		#endif

	}  // end for(iter < numof_iterations)

}



// random sampling metric tree
void stTreeSearch_rsmt( double *ref, double *query,
						long *refids, long *queryids,
						int numof_ref_points, int numof_query_points, int dim,
						int max_points, int max_tree_level,
						int k, int numof_iterations,
						//output
						vector< pair<double, long> > &kNN)
{
	double start_t = omp_get_wtime();

	// 0.I copy ref data
	pbinData refData = new binData();
	refData->X.resize(numof_ref_points*dim);
	refData->gids.resize(numof_ref_points);
	refData->dim = dim;
	#pragma omp parallel for
	for(int i = 0; i < numof_ref_points; i++) {
		for(int j = 0; j < dim; j++)
			refData->X[i*dim+j] = ref[i*dim+j];
		refData->gids[i] = refids[i];
	}
	// 0.II copy query data
	pbinData queryData = new binData();
	queryData->X.resize(numof_query_points*dim);
	queryData->gids.resize(numof_query_points);
	queryData->dim = dim;
	#pragma omp parallel for
	for(int i = 0; i < numof_query_points; i++) {
		for(int j = 0; j < dim; j++)
			queryData->X[i*dim+j] = query[i*dim+j];
		queryData->gids[i] = queryids[i];
	}
		
	// 1. build the tree (mtree)
	pstTree tree = new stTree();
	tree->options.splitter = "rsmt";
	tree->options.timing_verbose = 0;
	tree->options.debug_verbose = 0;
	tree->build(refData, max_points, max_tree_level);
	// -> after build the tree, "refData" has already been deleted. 

	if(tree->options.timing_verbose) {
		cout<<" build tree time: "<<omp_get_wtime() - start_t<<endl<<endl;
		start_t = omp_get_wtime();
	}

	// 2. traverse the tree several times by sample the nearest kid, and merge the results
	// - 2.1 first, do a deterministic search, traverse_type = 0
	tree->queryGreedy(queryData, k, kNN);
	if(tree->options.timing_verbose) {
		cout<<" deterministic search time: "<<omp_get_wtime() - start_t<<endl<<endl;
		start_t = omp_get_wtime();
	}


	// - 2.2 do numof_iterations random sampling traverse, traverse_type = 1
	for(int iter = 1; iter < numof_iterations; iter++) {
		vector< pair<double, long> > kNN_iter;
		tree->querySampling(queryData, k, kNN_iter);

		vector< pair<double, long> > tmpkNN;
		knn_merge(kNN, kNN_iter, numof_query_points, k, tmpkNN);
		#pragma omp parallel for
		for(int t = 0; t < tmpkNN.size(); t++)
			kNN[t] = tmpkNN[t];
	
		if(tree->options.timing_verbose) {
			cout<<" random sampling ("<<iter<<") search time: "<<omp_get_wtime() - start_t<<endl<<endl;
			start_t = omp_get_wtime();
		}
	
	}  // end for(iter < numof_iterations)

	delete queryData;
	delete tree;
}



// new random rotation kd tree, save memory
void stTreeSearch_rkdt( binData *&refData, binData *queryData, int k,
						int max_points, int max_tree_level, int numof_iterations, int flag_r, bool flag_rotate_back,
						//output
						vector< pair<double, long> >* &kNN)
{
	double start_t = omp_get_wtime();
    double tmp_t = 0.0;
    int numof_query_points = queryData->numof_points;
    int numof_ref_points = refData->numof_points;
    int dim = refData->dim;

    for(int iter = 0; iter < numof_iterations; iter++) {

        #if _SHARED_TREE_DEBUG_
            cout<<"\t (share): iter #"<<iter<<" ref data (before)"<<endl;
            for(int i = 0; i < refData->numof_points; i++) {
                cout<<"\t\t "<<refData->gids[i]<<" ";
                for(int j = 0; j < dim; j++)
                    cout<<refData->X[i*dim+j]<<" ";
                cout<<endl;
            }
            cout<<"\t (share): iter #"<<iter<<" query data (before)"<<endl;
            for(int i = 0; i < queryData->numof_points; i++) {
                cout<<"\t\t "<<queryData->gids[i]<<" ";
                for(int j = 0; j < dim; j++)
                    cout<<queryData->X[i*dim+j]<<" ";
                cout<<endl;
            }
        #endif

		start_t = omp_get_wtime();

        // 1. build the tree (splitter = rkdt, random kd tree)
		pstTree tree = new stTree();
		tree->options.splitter = "rkdt";
		tree->options.timing_verbose = 0;
		tree->options.debug_verbose = 0;
		tree->options.flag_r = flag_r;
		tree->build(refData, max_points, max_tree_level);
		// -> after build the tree, "refData" has already been deleted

#if _SHARED_TREE_TIMING_
cout<<endl<<"\t(shared) build tree, iter #"<<iter<<", time "<<omp_get_wtime()-start_t<<endl;
#endif

        // 2. search the tree
		start_t = omp_get_wtime();

		if(iter == 0) {
			// - 2.1 use nearest traverse strategy
            tree->queryGreedy(queryData, k, *kNN);
		}
		else {
			// - 2.2 iterate several times
			vector< pair<double, long> > kNN_iter;
			tree->queryGreedy(queryData, k, kNN_iter);

            vector< pair<double, long> > *tmpkNN = new vector< pair<double, long> >;
			knn_merge(*kNN, kNN_iter, numof_query_points, k, *tmpkNN);
            delete kNN;
            kNN = tmpkNN;
		} // end else

#if _SHARED_TREE_TIMING_
cout<<"\t(shared) search tree, iter #"<<iter<<", time "<<omp_get_wtime()-start_t<<endl;
#endif

        start_t = omp_get_wtime();
        // 3. copy ref data back (rotate back)
        refData = new binData();
		refData->X.resize(numof_ref_points*dim);
		refData->gids.resize(numof_ref_points);
		refData->dim = dim;
        refData->numof_points = numof_ref_points;
	    for(int i = 0; i < tree->leafRefArr.size(); i++) {
            for(int j = 0; j < tree->leafRefArr[i]->numof_points; j++) {
                int pos = tree->leafRefArr[i]->gids[j];
                memcpy( &(refData->X[pos*dim]), &(tree->leafRefArr[i]->X[j*dim]), sizeof(double)*dim );
                refData->gids[pos] = tree->leafRefArr[i]->gids[j];
            }
        }

#if _SHARED_TREE_TIMING_
cout<<"\t(shared) copy ref data, iter #"<<iter<<", time "<<omp_get_wtime()-start_t<<endl;
#endif

        #if _SHARED_TREE_DEBUG_
            cout<<"\t (share): iter #"<<iter<<" ref data (after)"<<endl;
            for(int i = 0; i < refData->numof_points; i++) {
                cout<<"\t\t "<<refData->gids[i]<<" ";
                for(int j = 0; j < dim; j++)
                    cout<<refData->X[i*dim+j]<<" ";
                cout<<endl;
            }
            cout<<"\t (share): iter #"<<iter<<" query data (after)"<<endl;
            for(int i = 0; i < queryData->numof_points; i++) {
                cout<<"\t\t "<<queryData->gids[i]<<" ";
                for(int j = 0; j < dim; j++)
                    cout<<queryData->X[i*dim+j]<<" ";
                cout<<endl;
            }
            cout<<endl;
        #endif

        start_t = omp_get_wtime();

        if(flag_rotate_back && tree->depth > 0) {
            newInverseRotatePoints( &(refData->X[0]), refData->numof_points, dim, tree->root->rw);
            newInverseRotatePoints( &(queryData->X[0]), queryData->numof_points, dim, tree->root->rw);
        }

	    delete tree;

#if _SHARED_TREE_TIMING_
cout<<"\t(shared) rotate point, iter #"<<iter<<", time "<<omp_get_wtime()-start_t<<endl;
#endif

    }  // end for(iter < numof_iterations)

}


void find_knn_srkdt( binData *&refData, binData *queryData, int k,
					 int max_points, int max_tree_level, int numof_iterations,
					 vector< pair<double, long> >* &kNN)
{
	double start_t = omp_get_wtime();
    int numof_query_points = queryData->numof_points;
    int numof_ref_points = refData->numof_points;
    int dim = refData->dim;

#if _SHARED_TREE_TIMING_
cout<<endl<<"\t(srkdt) maxp = "<<max_points<<" maxl = "<<max_tree_level<<" niter = "<<numof_iterations<<endl;
#endif

    for(int iter = 0; iter < numof_iterations; iter++) {

		start_t = omp_get_wtime();
        // 1. build the tree (splitter = rkdt, random kd tree)
		srkdt *tree = new srkdt();
		tree->build(refData, max_points, max_tree_level);

#if _SHARED_TREE_TIMING_
cout<<endl<<"\t(srkdt) build tree, iter #"<<iter<<", time "<<omp_get_wtime()-start_t<<endl;
#endif

        // 2. search the tree
		start_t = omp_get_wtime();

        if(iter == 0) {
			// - 2.1 use nearest traverse strategy
            tree->queryGreedy(queryData, k, *kNN);
		}
		else {
			// - 2.2 iterate several times
            tree->queryGreedyandMerge(queryData, k, *kNN);

            //vector< pair<double, long> > kNN_iter;
			//tree->queryGreedy(queryData, k, kNN_iter);
            //vector< pair<double, long> > *tmpkNN = new vector< pair<double, long> >;
		    //knn_merge(*kNN, kNN_iter, numof_query_points, k, *tmpkNN);
            //delete kNN;
            //kNN = tmpkNN;
		} // end else

	    delete tree;

#if _SHARED_TREE_TIMING_
cout<<"\t(srkdt) search tree, iter #"<<iter<<", time "<<omp_get_wtime()-start_t<<endl;
#endif

    }  // end for(iter < numof_iterations)

}


void find_knn_srkdt( binData *&refData, binData *queryData, int k,
					 int max_points, int max_tree_level, int numof_iterations,
					 vector< pair<double, long> >* &kNN,
                     vector<long> &queryIDs,
                     vector<long> &sampleIDs, vector<double> &globalKdist, vector<long> &globalKid)
{
	double start_t = omp_get_wtime();
    int numof_query_points = queryData->numof_points;
    int numof_ref_points = refData->numof_points;
    int dim = refData->dim;

#if _SHARED_TREE_TIMING_
cout<<endl<<"\t(srkdt) maxp = "<<max_points<<" maxl = "<<max_tree_level<<" niter = "<<numof_iterations<<endl;
#endif

    cout<<"************** total iters "<<numof_iterations<<" ***************"<<endl;

    for(int iter = 0; iter < numof_iterations; iter++) {

        double iter_t = omp_get_wtime();

		start_t = omp_get_wtime();
        // 1. build the tree (splitter = rkdt, random kd tree)
		srkdt *tree = new srkdt();
		tree->build(refData, max_points, max_tree_level);

#if _SHARED_TREE_TIMING_
cout<<endl<<"\t(srkdt) build tree, iter #"<<iter<<", time "<<omp_get_wtime()-start_t<<endl;
#endif

        // 2. search the tree
		start_t = omp_get_wtime();

        if(iter == 0) {
			// - 2.1 use nearest traverse strategy
            tree->queryGreedy(queryData, k, *kNN);
		}
        else {
			// - 2.2 iterate several times
            tree->queryGreedyandMerge(queryData, k, *kNN);

            //vector< pair<double, long> > kNN_iter;
			//tree->queryGreedy(queryData, k, kNN_iter);
            //vector< pair<double, long> > *tmpkNN = new vector< pair<double, long> >;
		    //knn_merge(*kNN, kNN_iter, numof_query_points, k, *tmpkNN);
            //delete kNN;
            //kNN = tmpkNN;
		} // end else

	    delete tree;

#if _SHARED_TREE_TIMING_
cout<<"\t(srkdt) search tree, iter #"<<iter<<", time "<<omp_get_wtime()-start_t<<endl;
#endif

		double hit_rate = 0.0, relative_error = 0.0;
		int nmiss = 0;
        double total_true_nn_dist = 0.0, total_estl_nn_dist = 0.0;
		verify(sampleIDs, globalKdist, globalKid,
                    queryIDs, *kNN, nmiss, hit_rate, relative_error,
                    total_true_nn_dist, total_estl_nn_dist);
		cout<<"    + shared memory knn search "
			<<": "<<sampleIDs.size()
            <<" samples -- hit rate "<< hit_rate << "%"
			<<"  relative error "<< relative_error << "%"
            <<"  elapsed time "<<omp_get_wtime()-iter_t
            <<endl;
            //<<"      total_true_nn_dist = "<<total_true_nn_dist
            //<<"  total_estl_nn_dist = "<<total_estl_nn_dist << endl;

    }  // end for(iter < numof_iterations)

}


void find_knn_srkdt_a2a( binData *&refData, int k,
					 int max_points, int max_tree_level, int numof_iterations,
					 vector< pair<double, long> >* &kNN)
{
	double start_t = omp_get_wtime();
    int numof_ref_points = refData->numof_points;
    int dim = refData->dim;

    for(int iter = 0; iter < numof_iterations; iter++) {
		srkdt *tree = new srkdt();
		tree->build(refData, max_points, max_tree_level);

        if(iter == 0) {
            tree->queryGreedy_a2a(k, *kNN);
		} else {
            tree->queryGreedyandMerge_a2a(k, *kNN);
		}

	    delete tree;
    }  // end for(iter < numof_iterations)

}


void find_knn_srkdt_a2a( binData *&refData, int k,
					 int max_points, int max_tree_level, int numof_iterations,
					 vector< pair<double, long> >* &kNN,
                     vector<long> &queryIDs,
                     vector<long> &sampleIDs, vector<double> &globalKdist, vector<long> &globalKid)
{
	double start_t = omp_get_wtime();
    int numof_ref_points = refData->numof_points;
    int dim = refData->dim;

    for(int iter = 0; iter < numof_iterations; iter++) {
		srkdt *tree = new srkdt();
		tree->build(refData, max_points, max_tree_level);

        if(iter == 0) {
            tree->queryGreedy_a2a(k, *kNN);
		} else {
            tree->queryGreedyandMerge_a2a(k, *kNN);
		}

	    delete tree;

		double hit_rate = 0.0, relative_error = 0.0;
		int nmiss = 0;
        double total_true_nn_dist = 0.0, total_estl_nn_dist = 0.0;
		verify(sampleIDs, globalKdist, globalKid,
                    queryIDs, *kNN, nmiss, hit_rate, relative_error,
                    total_true_nn_dist, total_estl_nn_dist);
		cout<<"    + shared memory knn search "
			<<": "<<sampleIDs.size()
            <<" samples -- hit rate "<< hit_rate << "%"
			<<"  relative error "<< relative_error << "%" << endl
            <<"      total_true_nn_dist = "<<total_true_nn_dist
            <<"  total_estl_nn_dist = "<<total_estl_nn_dist << endl;

    }  // end for(iter < numof_iterations)

}


