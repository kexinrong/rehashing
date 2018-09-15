#include <mpi.h>
#include <blas.h>
#include <omp.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <stdio.h>
#include <ompUtils.h>
#include <mpi.h>
#include <float.h>

#include "parTree.h"
#include "direct_knn.h"
#include "generator.h"
#include "verbose.h"
#include "rotation.h"
#include "repartition.h"
#include "mpitree.h"


using namespace std;
using namespace knn;


void parTree::destroy_tree(pparNode node)
{
	if(node != NULL) {
		destroy_tree(node->leftNode);
		destroy_tree(node->rightNode);
		delete node;
		node = NULL;
	}
}


parTree::~parTree()
{
	destroy_tree(root);

	for(int i = 0; i < leafRefArr.size(); i++) {
		if(leafRefArr[i] != NULL) {
			delete leafRefArr[i];
			leafRefArr[i] = NULL;
		}
	}
}


void parTree::build(pbinData inData, int minp, int maxlev, MPI_Comm comm)
{
	int nproc; 
	MPI_Comm_size(comm, &nproc);

	int numof_points = inData->X.size() / inData->dim;
	double logp = ceil( log((double)nproc) / log(2.0) );
	depth = min((int)logp, maxlev);
	int nleaf = (int)pow(2.0, (double)depth);
	numof_ref_points_in_tree = numof_points;
	
	leafRefArr.resize(nleaf);
	for(int i = 0; i < nleaf; i++)
		leafRefArr[i] = NULL;
	
	int nrcoord = 2 * inData->dim;
	rcoordArr.resize( nrcoord );
	for(int i = 0; i < nrcoord; i++) {
		double ri = (double)rand() / (double)RAND_MAX;
		int mvind = (int)(ri*(double)inData->dim);
		rcoordArr[i] = mvind;
	}
	MPI_CALL(MPI_Bcast(&(rcoordArr[0]), nrcoord, MPI_INT, 0, comm));
	
	root = new parNode();
	insert(NULL, root, inData, minp, maxlev);

	leafArrFlag.resize(nleaf);
	for(int i = 0; i < nleaf; i++) {
		if(NULL == leafRefArr[i]) {
			leafArrFlag[i] = -1;
		}
		else {
			leafArrFlag[i] = 1;
		}
	}

}

// leafRefArr will be released
void parTree::exchangeRefData(pbinData outData, MPI_Comm comm)
{
	double stage_t;
	double profile_t;

	#if STAGE_DTREE_OUTPUT_VERBOSE
		stage_t = omp_get_wtime();
	#endif

	int rank, size;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	int dim = leafRefArr[0]->dim;
	int *send_count = new int [size];
	double *raX = new double [numof_ref_points_in_tree*dim];
	long *ragids = new long [numof_ref_points_in_tree];

	int nleaf = leafRefArr.size();
	int p = 0;
	int sum = 0;
	for(int i = 0; i < nleaf; i++) {
		if( NULL != leafRefArr[i] ) {
			send_count[p] = leafRefArr[i]->gids.size();
			sum += send_count[p];
			p++;
		}
	}

	assert(sum == numof_ref_points_in_tree);

	int curpos = 0;
	for(int i = 0; i < nleaf; i++) {
		if(NULL != leafRefArr[i]) {
			int n = leafRefArr[i]->gids.size();
			memcpy(raX+curpos*dim, &(leafRefArr[i]->X[0]), sizeof(double)*n*dim);
			memcpy(ragids+curpos, &(leafRefArr[i]->gids[0]), sizeof(long)*n);
			curpos += n;
		}
	}

	// release data stored in leafRefArr to save memory
	for(int i = 0; i < nleaf; i++) {
		if(NULL != leafRefArr[i]) {
			delete leafRefArr[i];
			leafRefArr[i] = NULL;
		}
	}

	#if STAGE_DTREE_OUTPUT_VERBOSE
		MPI_Barrier(comm);
		if(rank == 0) { 
			cout<<"    > REPART: Local Rearrange Ref Done: sum(send_count) = "<<sum
			    <<" ; numof_points in tree = "<<numof_ref_points_in_tree
				<<" -> "<<omp_get_wtime() - stage_t<<endl;
		}
		stage_t = omp_get_wtime();
	#endif
	
	#if COMM_TIMING_VERBOSE
		profile_t = omp_get_wtime();
	#endif

	double *newX;
	long *newgids;
	long newN;
	knn::repartition::repartition(ragids, raX, (long)numof_ref_points_in_tree, send_count, dim,
								  &newgids, &newX, &newN, comm);
		
	#if COMM_TIMING_VERBOSE
		Repartition_Tree_Build_T_ += omp_get_wtime() - profile_t;
	#endif

	#if STAGE_DTREE_OUTPUT_VERBOSE
		MPI_Barrier(comm);
		if(rank == 0) cout<<"    > REPART: Repartition Ref Points Done! - "<<omp_get_wtime() - stage_t<<endl;
		stage_t = omp_get_wtime();
	#endif
	
	delete [] raX;
	delete [] ragids;
	delete [] send_count;

	outData->X.resize(newN*dim);
	outData->gids.resize(newN);
	outData->dim = dim;
	int tnsize = newN*dim;
	#pragma omp parallel for
	for(int i = 0; i < tnsize; i++)
		outData->X[i] = newX[i];
	#pragma omp parallel for
	for(int i = 0; i < newN; i++)
		outData->gids[i] = newgids[i];

	delete [] newX;
	delete [] newgids;

	#if STAGE_DTREE_OUTPUT_VERBOSE
		MPI_Barrier(comm);
		if(rank == 0) cout<<"    > REPART: Copyting Ref Points Done! - "<<omp_get_wtime() - stage_t<<endl;
		stage_t = omp_get_wtime();
	#endif

}


// "inData" will be deleted inside this function
void parTree::insert(pparNode in_parent, pparNode inNode, pbinData inData, int minp, int maxlev)
{
	
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	double stage_t;
	
	#if STAGE_DTREE_OUTPUT_VERBOSE
		stage_t = omp_get_wtime();
	#endif

	// input checks
	assert( minp > 1 );
	assert( maxlev >= 0 );
	
	int numof_kids = 2;
	int dim = inData->dim;
	int numof_points = inData->X.size()/dim;
	int glb_numof_points = 0;
	
	// Initializations
	if (in_parent!=NULL)  { 
		inNode->level = in_parent->level + 1; 
		inNode->parent = in_parent; 
	}
	else {
		inNode->commsize = size;
		MPI_CALL(MPI_Allreduce(&numof_points, &glb_numof_points, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
		inNode->glbN = glb_numof_points;
	}
	
	vector<double> &X = inData->X;
	vector<long> &gids= inData->gids;
	glb_numof_points = inNode->glbN;

	#if STAGE_DTREE_OUTPUT_VERBOSE
		MPI_Barrier(MPI_COMM_WORLD);
		if(rank == 0) cout<<"    > Insert: level "<<inNode->level<<" initialization done - numof_points = "<<numof_points<<endl;
		stage_t = omp_get_wtime();
	#endif

	// BASE CASE TO TERMINATE RECURSION
	if (   inNode->level == maxlev ||              // reached max level
			glb_numof_points <= minp || // not enough points to further partition
			inNode->commsize <= 1
		) { 
		int refArrInd = inNode->lnid;
		if(inNode->level < depth) refArrInd = inNode->lnid*2+0;
		leafRefArr[refArrInd] = new binData();
		leafRefArr[refArrInd]->Copy(inData);
		delete inData; inData = NULL; 
	
		#if STAGE_DTREE_OUTPUT_VERBOSE
			MPI_Barrier(MPI_COMM_WORLD);
			if(rank == 0) cout<<"    > Insert: level "<<inNode->level<<" base case done - "<<omp_get_wtime() - stage_t<<endl;
			stage_t = omp_get_wtime();
		#endif

		return;
	}// end of base case 

	pbinData leftData = new binData;
	pbinData rightData = new binData;
	int glb_n_left = 0, glb_n_right = 0;
	vector<int> kid_size;
	{	// scope all vector

		vector<double> px(numof_points);
		// 1. rotate points
		vector<double> tmpX;
		vector<double> rw;
		if(inNode->level == 0 && options.flag_r == 1
			&& 0 == strcmp(options.splitter.c_str(), "rkdt") ) {		// if root level
			tmpX.resize(X.size());
			generateRotation( dim, inNode->rw, MPI_COMM_WORLD );
			memcpy( &(tmpX[0]), &(X[0]), sizeof(double)*X.size() );
			rotatePoints( &(tmpX[0]), numof_points, dim, inNode->rw, &(X[0]) );
			tmpX.clear();
		}
		else if( options.flag_r == 2
				&& 0 == strcmp(options.splitter.c_str(), "rkdt") ) {	// rotate on each level
			tmpX.resize(X.size());
			generateRotation( dim, rw, MPI_COMM_WORLD );
			rotatePoints( &(X[0]), numof_points, dim, rw, &(tmpX[0]) );
		}
	
		#if STAGE_DTREE_OUTPUT_VERBOSE
			MPI_Barrier(MPI_COMM_WORLD);
			if(rank == 0) cout<<"    > Insert: level "<<inNode->level<<" rotate points done - "<<omp_get_wtime() - stage_t<<endl;
			stage_t = omp_get_wtime();
		#endif

		// 2. choose projection direction and project values
		if(0 == strcmp(options.splitter.c_str(), "rkdt")) {
			if(options.flag_r == 2) {
				int gnid = (int)pow(2.0, (double)inNode->level) - 1 + inNode->lnid;
				int pos = gnid % rcoordArr.size();
				inNode->coord_mv = rcoordArr[pos];
				maxvarProjection(&(tmpX[0]), numof_points, glb_numof_points,
								dim, inNode->coord_mv, &(px[0]), MPI_COMM_WORLD);
				inNode->proj.resize(dim);
				vector<double> e; e.resize(dim);
				e[inNode->coord_mv] = 1.0;
				RROT_INV(&(e[0]), &(inNode->proj[0]), &(rw[0]));
			}
			else {
				int gnid = (int)pow(2.0, (double)inNode->level) - 1 + inNode->lnid;
				int pos = gnid % rcoordArr.size();
				inNode->coord_mv = rcoordArr[pos];
				maxvarProjection(&(X[0]), numof_points, glb_numof_points,
								dim, inNode->coord_mv, &(px[0]), MPI_COMM_WORLD);
			}
		}
	
	
		#if STAGE_DTREE_OUTPUT_VERBOSE
			MPI_Barrier(MPI_COMM_WORLD);
			if(rank == 0) cout<<"    > Insert: level "<<inNode->level<<" choose direction done - "<<omp_get_wtime() - stage_t<<endl;
			stage_t = omp_get_wtime();
		#endif

		// 3. find median
		inNode->median = distSelect(px, glb_numof_points/2, MPI_COMM_WORLD);
	
		#if STAGE_DTREE_OUTPUT_VERBOSE
			MPI_Barrier(MPI_COMM_WORLD);
			if(rank == 0) cout<<"    > Insert: level "<<inNode->level<<" find median done - "<<omp_get_wtime() - stage_t<<endl;
			stage_t = omp_get_wtime();
		#endif
	
		// 4. assign membership
		int *point_to_kid_membership = new int [numof_points];
		vector<int> local_numof_points_per_kid(numof_kids) ;
		vector<int> glb_numof_points_per_kid(numof_kids);
		assignMembership(&(px[0]), numof_points, glb_numof_points, inNode->median,
						 point_to_kid_membership, &(local_numof_points_per_kid[0]), 
						 &(glb_numof_points_per_kid[0]),
						 MPI_COMM_WORLD);
		glb_n_left = glb_numof_points_per_kid[0];
		glb_n_right = glb_numof_points_per_kid[1];


		#if STAGE_DTREE_OUTPUT_VERBOSE
			MPI_Barrier(MPI_COMM_WORLD);
			if(rank == 0) {
				cout<<"    > Insert: level "<<inNode->level<<" assign membership done - "
					<<" glb_numof_points_per_kid: "<<glb_numof_points_per_kid[0]<<" "<<glb_numof_points_per_kid[1]
					<<" - "<<omp_get_wtime() - stage_t<<endl;
			}
			stage_t = omp_get_wtime();
		#endif
	

		// 5. assign comm size
		work_partition(glb_numof_points_per_kid, inNode->commsize, kid_size);
	
		#if STAGE_DTREE_OUTPUT_VERBOSE
			MPI_Barrier(MPI_COMM_WORLD);
			if(rank == 0) {
				cout<<"    > Insert: level "<<inNode->level<<" assign comm size done - "
					<<" kid_size (comm): "<<kid_size[0]<<" "<<kid_size[1]
					<<" - "<<omp_get_wtime() - stage_t<<endl;
			}
			stage_t = omp_get_wtime();
		
			if( abs(glb_numof_points_per_kid[0] - glb_numof_points_per_kid[1]) > 2 ) {
				cout<<"!!!! imbalance at level "<<inNode->level<<" worldrank - "<<rank
				<<" glb_numof_points_per_kid: "<<glb_numof_points_per_kid[0]<<" "<<glb_numof_points_per_kid[1]
				<<endl;	
			}
		#endif

		// 6. copy data
		vector<int> leftkid_membership;
		vector<int> rightkid_membership;
		leftkid_membership.reserve(local_numof_points_per_kid[0]);
		rightkid_membership.reserve(local_numof_points_per_kid[1]);
		for(int i = 0; i < numof_points; i++) {
			if(point_to_kid_membership[i] == 0) {
				leftkid_membership.push_back(i);
			}
			else {
				rightkid_membership.push_back(i);
			}
		}
		copyData(inData, leftkid_membership, leftData);
		copyData(inData, rightkid_membership, rightData);
	
		delete [] point_to_kid_membership;
		delete inData; inData = NULL;
	
		#if STAGE_DTREE_OUTPUT_VERBOSE
			MPI_Barrier(MPI_COMM_WORLD);
			if(rank == 0) cout<<"    > Insert: level "<<inNode->level<<" copying data done - "<<omp_get_wtime() - stage_t<<endl;
		#endif


	}	// end scope all vectors
	
	// 6. recursively build the tree
	inNode->leftNode = new parNode(2*inNode->lnid+0, kid_size[0], glb_n_left);
	inNode->rightNode = new parNode(2*inNode->lnid+1, kid_size[1], glb_n_right);
	insert(inNode, inNode->leftNode, leftData, minp, maxlev);
	insert(inNode, inNode->rightNode, rightData, minp, maxlev);
};



void parTree::assignMembership(double *px, int numof_points, int glb_numof_points, double median,
							   // output
								int *point_to_kid_membership,
								int *local_numof_points_per_kid,
								int *glb_numof_points_per_kid,
								MPI_Comm comm)
{

	local_numof_points_per_kid[0] = 0;
	local_numof_points_per_kid[1] = 0;
	int npkid0 = 0, npkid1 = 0, npsame = 0;
	
	int *local_arr = new int [3];
	int *glb_arr = new int [3];

	#pragma omp parallel if (numof_points > 1000)
	{
		#pragma omp for reduction(+:npkid0,npkid1,npsame)
		for(int i = 0; i < numof_points; i++) {
			double diff = fabs((px[i]-median)/median);
			//if(diff < 2*DBL_EPSILON) {
			if(diff < 1.0e-6) {
				npsame++;
				point_to_kid_membership[i] = -1;
			}
			else {
				if(px[i] < median) {
					point_to_kid_membership[i] = 0;
					npkid0++;
				}
				else {
					point_to_kid_membership[i] = 1;
					npkid1++;
				}
			}
		}// end for
	}
	
	local_arr[0] = npkid0;
	local_arr[1] = npkid1;
	local_arr[2] = npsame;

	MPI_CALL(MPI_Allreduce(local_arr, glb_arr, 3, MPI_INT, MPI_SUM, comm));
	int glb_npsame = glb_arr[2];
	if(glb_npsame > 0) {
		
		int scan_nsame;
		MPI_Scan(&npsame, &scan_nsame, 1, MPI_INT, MPI_SUM, comm);
		int glb_n_move_left = glb_numof_points/2 - glb_arr[0];
		int glb_n_move_right = glb_npsame - glb_n_move_left;
		int pmove = 0, local_n_move_left = 0, local_n_move_right = 0;
		if(scan_nsame < glb_n_move_left) {
			local_n_move_left = npsame;
			local_n_move_right = 0;
		}
		else {
			int excl_scan = scan_nsame - npsame;
			local_n_move_left = glb_n_move_left - excl_scan;
			if(local_n_move_left < 0) local_n_move_left = 0;
			local_n_move_right = npsame - local_n_move_left;
		}

		for(int i = 0; i < numof_points; i++) {
			if(point_to_kid_membership[i] == -1) {
				if(pmove < local_n_move_left) {
					point_to_kid_membership[i] = 0;
					local_arr[0]++;
				}
				else {
					point_to_kid_membership[i] = 1;
					local_arr[1]++;
				}
				pmove++;
			}
		}

		/*int glb_n_move_left = glb_numof_points/2 - glb_arr[0];
		int glb_n_move_right = glb_npsame - glb_n_move_left;
		double n_tmp_right = (double)glb_n_move_right*( (double)npsame/(double)glb_npsame );
		n_tmp_right = floor(n_tmp_right+0.5);
		int local_n_move_right = (int)(n_tmp_right);
		local_n_move_right = min(local_n_move_right, npsame);
		int local_n_move_left = npsame - local_n_move_right;
		int pmove = 0;
		for(int i = 0; i < numof_points; i++) {
			if(point_to_kid_membership[i] == -1) {
				if(pmove < local_n_move_left) {
					point_to_kid_membership[i] = 0;
					local_arr[0]++;
				}
				else {
					point_to_kid_membership[i] = 1;
					local_arr[1]++;
				}
				pmove++;
			}
		}*/

		MPI_CALL(MPI_Allreduce(local_arr, glb_arr, 3, MPI_INT, MPI_SUM, comm));
	} // end if (glb_npsame > 0)
	
	local_numof_points_per_kid[0] = local_arr[0];
	local_numof_points_per_kid[1] = local_arr[1];
	glb_numof_points_per_kid[0] = glb_arr[0];
	glb_numof_points_per_kid[1] = glb_arr[1];

	delete [] local_arr;
	delete [] glb_arr;

}


void parTree::maxvarProjection(double *points, int numof_points, int glb_numof_points, int dim,
							 // output
							 int &mvind, double *pv,
							 MPI_Comm comm)
{
	if(options.flag_c != 0) {
		// 1.1 glb_mu
		double *glb_mu = new double [dim];
		glb_mean(points, numof_points, glb_numof_points, dim, glb_mu, comm);
		// 1.2 local_var
		vector<double> local_var(dim);
		vector<double> glb_var(dim);
		if(numof_points > 0) parvar(points, numof_points, dim, glb_mu, &(local_var[0]));
		delete [] glb_mu;
		// 1.3 glb_var
		MPI_CALL(MPI_Allreduce(&local_var[0], &glb_var[0], dim, MPI_DOUBLE, MPI_SUM, comm));
		// 2. coord with max var 
		vector<double>::iterator it = max_element(glb_var.begin(), glb_var.end());
		mvind = it - glb_var.begin();
	}

	// 3. pv
	#pragma omp parallel if(numof_points > 1000)
	{
		#pragma omp for
		for(int i = 0; i < numof_points; i++)
			pv[i] = points[i*dim+mvind];
	}
}



void parTree::parvar(double *points, int numof_points, int dim, double *mean, double *var) 
{
	int stride = dim + 128 / sizeof(double); //Make sure two threads don't write to same cache line.
	int maxt = omp_get_max_threads();

	vector<double> threadvar( stride * maxt );
	for(int i = 0; i < dim; i++) var[i] = 0.0;
	#pragma omp parallel if(numof_points > 1500)
	{
		int t = omp_get_thread_num();
		double *localvar = &(threadvar[t*stride]);

		for(int i = 0; i < dim; i++) localvar[i] = 0.0;
        int npdim = numof_points*dim;
       	register int j;
        #pragma omp for schedule(dynamic,25)
        for(int i = 0; i < numof_points; i++) {
           	register int idim = i*dim;
           	#pragma vector
           	for(j = 0; j < dim; j++) {
				double diff = points[idim+j] - mean[j];
				localvar[j] += diff*diff;
           	}
		}
	}
	for(int t = 0; t < maxt; t++) {
		double *localvar = &(threadvar[t*stride]);
		for(int i = 0; i < dim; i++)
			var[i] += localvar[i];
	}
}


void parTree::glb_mean(// input
						double *points, int numof_points, int glb_numof_points, int dim,
						// output
						double *glb_mu, 
						MPI_Comm comm)
{

	double *mu = new double [dim];
	for(int i = 0; i < dim; i++) mu[i] = 0.0;

	int stride = dim + 128 / sizeof(double); //Make sure two threads don't write to same cache line.
	int maxt = omp_get_max_threads();

	vector<double> threadmu( stride * maxt );
	#pragma omp parallel if(numof_points > 2000)
	{
		int t = omp_get_thread_num();
		double *localmu = &(threadmu[t*stride]);
		for(int i = 0; i < dim; i++) localmu[i] = 0.0;
		int npdim = numof_points * dim;

		register int idim;
		register int j;
		#pragma omp for schedule(dynamic,50)
		for(int i = 0; i < numof_points; i++) {
			idim = i*dim;
			#pragma vector
			for(j = 0; j < dim; j++)
				localmu[j] += points[idim+j];
		}

	}

	for(int t = 0; t < maxt; t++) { 
		double *localmu = &(threadmu[t*stride]);
		for(int i = 0; i < dim; i++)
			mu[i] += localmu[i];
	}

	MPI_CALL(MPI_Allreduce(mu, glb_mu, dim, MPI_DOUBLE, MPI_SUM, comm));

	for(int i = 0; i < dim; i++) 
		glb_mu[i] /= (double)glb_numof_points;

	delete [] mu;

}


// select the kth smallest element in arr
// for median, ks = glb_N / 2
double parTree::distSelect(vector<double> &arr, int ks, MPI_Comm comm)
{
	vector<double> S_less;
	vector<double> S_great;
	S_less.reserve(arr.size());
	S_great.reserve(arr.size());
	
	int N = arr.size();
	int glb_N;
	MPI_CALL(MPI_Allreduce(&N, &glb_N, 1, MPI_INT, MPI_SUM, comm));
	
	assert(glb_N > 0);

	double mean;
	glb_mean(&(arr[0]), N, glb_N, 1, &mean, comm);
	
	for(int i = 0; i < arr.size(); i++) {
		if(arr[i] > mean) S_great.push_back(arr[i]);
		else S_less.push_back(arr[i]);
	}

	int N_less, N_great, glb_N_less, glb_N_great;
	N_less = S_less.size();
	N_great = S_great.size();
	int * local_tmp = new int [2];
	int * glb_tmp = new int [2];
	local_tmp[0] = N_less;
	local_tmp[1] = N_great;
	MPI_CALL(MPI_Allreduce(local_tmp, glb_tmp, 2, MPI_INT, MPI_SUM, comm));
	glb_N_less = glb_tmp[0];
	glb_N_great = glb_tmp[1];
	delete [] local_tmp;
	delete [] glb_tmp;


	if( glb_N_less == ks || glb_N == 1 || glb_N == glb_N_less || glb_N == glb_N_great ) return mean;
	else if(glb_N_less > ks) {
		return distSelect(S_less, ks, comm);
	}
	else {
		return distSelect(S_great, ks-glb_N_less, comm);
	}

}


void parTree::copyData(pbinData inData, vector<int> & membership, pbinData outData)
{
	int dim = inData->dim;
	outData->X.resize(membership.size()*dim);
	outData->gids.resize(membership.size());
	outData->dim = dim;
	outData->numof_points = membership.size();
	int membsize = membership.size();

	#pragma omp parallel if(membsize > 1000)
	{
		int idim;
		int membi;
		int membidim;
		long *ingids = &(inData->gids[0]);
		long *outgids = &(outData->gids[0]);
		double *inX =  &(inData->X[0]);
		double *outX =  &(outData->X[0]);
		int i, j;
		#pragma omp for //schedule(dynamic,50)
		for(i = 0; i < membsize; i++) {
			idim = i*dim;
			membi = membership[i];
			membidim = membi*dim;
			outgids[i] = ingids[ membi ];
			#pragma vector
			#pragma ivdep
			for(j = 0; j < dim; j++)
				outX[idim+j] = inX[membidim+j];
		}
	}

	if(inData->lids.size() > 0) {
		outData->lids.resize(membership.size());
		#pragma omp parallel if(membsize > 1000)
		{
			int membi;
			int membidim;
			long *inlids = &(inData->lids[0]);
			long *outlids = &(outData->lids[0]);
			int i, j;
			#pragma omp for //schedule(dynamic,50)
			for(i = 0; i < membsize; i++) {
				outlids[i] = inlids[ membership[i] ];
			}
		}
	}	// end if
}




int parTree::visitGreedy(double *point, int dim, pparNode node)
{
	if(NULL == node->leftNode && NULL == node->rightNode) {		// leaf node
		int QueryArrInd = node->lnid;
		if(node->level < depth) QueryArrInd = node->lnid*2+0;
		return QueryArrInd;
	}
	else {
		double py = 0.0;
		if( 0 == strcmp(options.splitter.c_str(), "rkdt") 
				&& options.flag_r != 2) {
			py = point[node->coord_mv];
		}
		else {
			int ONE = 1;
			py = ddot(&dim, &(node->proj[0]), &ONE, &(point[0]), &ONE);
		}

		if( py < node->median )
			return visitGreedy(point, dim, node->leftNode);
		else
			return visitGreedy(point, dim, node->rightNode);
	}
}



void parTree::distributeQueryData(pbinData queryData, MPI_Comm comm)
{
	double stage_t;
	double profile_t;

	int rank, size;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	#if STAGE_DTREE_OUTPUT_VERBOSE
		if(rank == 0) cout<<"    > DistributeQ: Enter distributeQueryData()"<<endl;
		stage_t = omp_get_wtime();
	#endif

	int dim = queryData->dim;
	int numof_queries = queryData->X.size() / dim;

	// if "rkdt, flag_r = 1", rotate query points once
	if( options.flag_r == 1 && 0 == strcmp(options.splitter.c_str(), "rkdt") ) {
		vector<double> tmpX(queryData->X.size());
		memcpy(&(tmpX[0]), &(queryData->X[0]), sizeof(double)*(queryData->X.size()));
		rotatePoints( &(tmpX[0]), numof_queries, dim, root->rw, &(queryData->X[0]) );
	}

	#if STAGE_DTREE_OUTPUT_VERBOSE
		if(rank == 0) cout<<"    > DistributeQ: Rotate Query Points Done! - "<<omp_get_wtime() - stage_t<<endl;
		stage_t = omp_get_wtime();
	#endif

	vector<int> membership(numof_queries);
	//int nleaf = leafRefArr.size();
	int nleaf = leafArrFlag.size();
	vector<int> numof_queries_per_leaf;
	numof_queries_per_leaf.resize(nleaf);
    int membi;
    int *nqpl;
	#pragma omp parallel for private(membi,nqpl)
	for(int i = 0; i < numof_queries; i++) {
		membership[i] = visitGreedy(&(queryData->X[0])+i*dim, dim, root);
	}
	for(int i = 0; i < numof_queries; i++)
		numof_queries_per_leaf[membership[i]]++;

	#if STAGE_DTREE_OUTPUT_VERBOSE
		if(rank == 0) cout<<"    > DistributeQ: Assign membership Done! - "<<omp_get_wtime() - stage_t<<endl;
		stage_t = omp_get_wtime();
	#endif

	// allocate memory for query data.
	vector<pbinData> leafQueryArr(nleaf);
	for(int i = 0; i < nleaf; i++) 
		leafQueryArr[i] = NULL;
	for(int i = 0; i < nleaf; i++) {
		if(leafArrFlag[i] != -1) {
			leafQueryArr[i] = new binData();
			leafQueryArr[i]->X.reserve( numof_queries_per_leaf[i]*dim );
			leafQueryArr[i]->gids.reserve( numof_queries_per_leaf[i] );
			leafQueryArr[i]->lids.reserve( numof_queries_per_leaf[i] );
			leafQueryArr[i]->dim = dim;
		}
	}

	#if STAGE_DTREE_OUTPUT_VERBOSE
		if(rank == 0) cout<<"    > DistributeQ: Allocate memory for leafQueryArr Done! - "<<omp_get_wtime() - stage_t<<endl;
		stage_t = omp_get_wtime();
	#endif

	// redistribute query data to leafQueryArr[]
	for(int i = 0; i < numof_queries; i++) {
		for(int j = 0; j < dim; j++) 
			leafQueryArr[membership[i]]->X.push_back( queryData->X[i*dim+j] );
		leafQueryArr[membership[i]]->gids.push_back( queryData->gids[i] );
		leafQueryArr[membership[i]]->lids.push_back( (long)i ); 
	}

	#if STAGE_DTREE_OUTPUT_VERBOSE
		if(rank == 0) cout<<"    > DistributeQ: Copying points to LeafQueryArr Done! - "<<omp_get_wtime() - stage_t<<endl;
		stage_t = omp_get_wtime();
	#endif

	int *send_count = new int [size];
	int p = 0;
	for(int i = 0; i < nleaf; i++) {
		if(NULL != leafQueryArr[i]) {
			send_count[p] = leafQueryArr[i]->gids.size();
			p++;
		}
	}

	int curpos = 0;
	for(int i = 0; i < nleaf; i++) {
		if(NULL != leafQueryArr[i]) {
			int n = leafQueryArr[i]->gids.size();
			memcpy( &(queryData->X[curpos*dim]), &(leafQueryArr[i]->X[0]), sizeof(double)*n*dim );
			memcpy( &(queryData->gids[curpos]), &(leafQueryArr[i]->gids[0]), sizeof(long)*n );
			curpos += n;
		}
	}

	for(int i = 0; i < nleaf; i++) {
		if(NULL != leafQueryArr[i]) {
			delete leafQueryArr[i];
			leafQueryArr[i] = NULL;
		}
	}

	#if STAGE_DTREE_OUTPUT_VERBOSE
		if(rank == 0) cout<<"    > DistributeQ: Local Rearrange Query Points Done! - "<<omp_get_wtime() - stage_t<<endl;
		stage_t = omp_get_wtime();
	#endif
	
	#if COMM_TIMING_VERBOSE
		MPI_Barrier(comm);
		profile_t = omp_get_wtime();
	#endif

	double *newX;
	long *newgids;
	long newN;
	knn::repartition::repartition(&(queryData->gids[0]), &(queryData->X[0]), (long)numof_queries,
								send_count, dim, &newgids, &newX, &newN, comm);
	delete [] send_count;

	#if COMM_TIMING_VERBOSE
		Repartition_Query_T_ += omp_get_wtime() - profile_t;
	#endif


	#if STAGE_DTREE_OUTPUT_VERBOSE
		if(rank == 0) cout<<"    > DistributeQ: Repartition Query Points Done! - "<<omp_get_wtime() - stage_t<<endl;
		stage_t = omp_get_wtime();
	#endif
	
	queryData->X.resize(newN*dim);
	queryData->gids.resize(newN);
	int tnsize = newN*dim;
	#pragma omp parallel for
	for(int i = 0; i < tnsize; i++)
		queryData->X[i] = newX[i];
	#pragma omp parallel for
	for(int i = 0; i < newN; i++)
		queryData->gids[i] = newgids[i];
	
	delete [] newX;
	delete [] newgids;

	#if STAGE_DTREE_OUTPUT_VERBOSE
		if(rank == 0) cout<<"    > DistributeQ: Copying Repartitioned Query Points Done! - "<<omp_get_wtime() - stage_t<<endl;
		stage_t = omp_get_wtime();
	#endif

}




// select the kth smallest element in arr
// for median, ks = glb_N / 2
double parTree::select(vector<double> &arr, int ks)
{
	vector<double> S_less;
	vector<double> S_great;
	S_less.reserve(arr.size());
	S_great.reserve(arr.size());
	
	double mu;
	mean(&(arr[0]), arr.size(), 1, &mu);
	
	for(int i = 0; i < arr.size(); i++) {
		if(arr[i] > mu) S_great.push_back(arr[i]);
		else S_less.push_back(arr[i]);
	}

	if( S_less.size() == ks || arr.size() == 1 || arr.size() == S_less.size() || arr.size() == S_great.size() ) return mu;
	else if(S_less.size() > ks) {
		return select(S_less, ks);
	}
	else {
		return select(S_great, ks-S_less.size());
	}

}


void parTree::mean(// input
				  double *points, int numof_points, int dim,
				  // output
				  double *mu)
{
	int stride = dim + 128 / sizeof(double); //Make sure two threads don't write to same cache line.
	int maxt = omp_get_max_threads();

	vector<double> threadmu( stride * maxt );
	for(int i = 0; i < dim; i++) mu[i] = 0.0;
	#pragma omp parallel if(numof_points > 2000)
	{
		int t = omp_get_thread_num();
		double *localmu = &(threadmu[t*stride]);
		for(int i = 0; i < dim; i++) localmu[i] = 0.0;
		int npdim = numof_points * dim;

		register int idim;
		register int j;
		#pragma omp for schedule(dynamic,50)
		for(int i = 0; i < numof_points; i++) {
			idim = i*dim;
			#pragma vector
			for(j = 0; j < dim; j++)
				localmu[j] += points[idim+j];
		}

	}

	for(int t = 0; t < maxt; t++) { 
		double *localmu = &(threadmu[t*stride]);
		for(int i = 0; i < dim; i++)
			mu[i] += localmu[i];
	}

	for(int i = 0; i < dim; i++) 
		mu[i] /= (double)numof_points;

}


