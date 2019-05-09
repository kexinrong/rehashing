#include <mpi.h>
#include <blas.h>
#include <omp.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <stdio.h>
#include <ompUtils.h>
#include <float.h>
#include <cstring>

#include "stTree.h"
#include "direct_knn.h"
#include "generator.h"
#include "verbose.h"
#include "rotation.h"

//#if LOAD_BALANCE_VERBOSE
#if STAGE_STREE_OUTPUT_VERBOSE
	#include <mpi.h>
#endif

#define _STTREE_DEBUG_ false

using namespace std;
using namespace knn;


void stTree::destroy_tree(pstNode node)
{
	if(node != NULL) {
		destroy_tree(node->leftNode);
		destroy_tree(node->rightNode);
		delete node;
		node = NULL;
	}
}


stTree::~stTree()
{
	destroy_tree(root);

	for(int i = 0; i < leafRefArr.size(); i++) {
		delete leafRefArr[i];
		leafRefArr[i] = NULL;
	}
}


void stTree::build(pbinData inData, int minp, int maxlev)
{
	int numof_points = inData->X.size() / inData->dim;
	double least_numof_nodes = (double)numof_points / (double)minp;
	double d = ceil(log(least_numof_nodes) / log(2.0));
	depth = (int)d;
	depth = min(maxlev, depth);
	int nleaf = (int)pow(2.0, (double)depth);
	numof_ref_points_in_tree = numof_points;

	if(options.debug_verbose) cout<<"<<< depth: "<<depth<<" nleaf: "<<nleaf<<" >>>"<<endl;

	leafRefArr.reserve(nleaf);
	root = new stNode();

	insert(NULL, root, inData, minp, maxlev);
}


void stTree::recoverData(pbinData outData)
{
	int dim = leafRefArr[0]->dim;
	outData->X.resize(numof_ref_points_in_tree*dim);
	outData->gids.resize(numof_ref_points_in_tree);
    outData->numof_points = numof_ref_points_in_tree;

	int nleaf = leafRefArr.size();
	int curpos = 0;
    for(int i = 0; i < nleaf; i++) {
		int n = leafRefArr[i]->gids.size();
		memcpy(&(outData->X[curpos*dim]), &(leafRefArr[i]->X[0]), sizeof(double)*n*dim);
		memcpy(&(outData->gids[curpos]), &(leafRefArr[i]->gids[0]), sizeof(long)*n);
		curpos += n;
	}
}


// "inData" will be deleted inside this function
void stTree::insert(pstNode in_parent, pstNode inNode, pbinData inData, int minp, int maxlev)
{

#if STAGE_STREE_OUTPUT_VERBOSE
	int worldrank, worldsize;
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
	double stage_t = omp_get_wtime();
#endif

	// input checks
	assert( minp > 1 );
	assert( maxlev >= 0 );

	// Initializations
    if (in_parent!=NULL)  {
		inNode->level = in_parent->level + 1;
		inNode->parent = in_parent;
	}

	int dim = inData->dim;
	vector<double> &X = inData->X;
	vector<long> &gids= inData->gids;
	int numof_points = X.size()/dim;

	if(options.debug_verbose == 1) cout<<endl<<"level: "<<inNode->level<<" lnid: "<<inNode->lnid<<endl;

	// BASE CASE TO TERMINATE RECURSION
	if ( inNode->level == maxlev ||              // reached max level
		 numof_points <= minp // not enough points to further partition
       ) {
		//leafRefArr[inNode->lnid] = new binData();
		//leafRefArr[inNode->lnid]->Copy(inData);
		//delete inData;
		//leafRefArr[inNode->lnid] = inData;

		leafRefArr.push_back(inData);
		if(options.debug_verbose == 1) cout<<"base case done"<<endl;
		#if STAGE_STREE_OUTPUT_VERBOSE
		if(worldrank == worldsize-1) {
			cout<<"    >> Stree: Build: level "<<inNode->level
				<<": copy points into leafArr -> "<<omp_get_wtime()-stage_t<<endl;
		}
		#endif
		return;
	}// end of base case

	pbinData leftData = new binData;
	pbinData rightData = new binData;
	{	// scope all vectors

		vector<double> px(numof_points);
		vector<double> tmpX;
		vector<double> rw;

		// 1 rotate points
		if(inNode->level == 0 && options.flag_r == 1
			&& 0 == strcmp(options.splitter.c_str(), "rkdt") ) {		// if root level
			generateRotation( dim, inNode->rw );
			newRotatePoints( &(X[0]), numof_points, dim, inNode->rw);

            //tmpX.resize(X.size());
			//generateRotation( dim, inNode->rw );
			//memcpy( &(tmpX[0]), &(X[0]), sizeof(double)*X.size() );
			//rotatePoints( &(tmpX[0]), numof_points, dim, inNode->rw, &(X[0]) );
			//tmpX.clear();
		}
		else if( options.flag_r == 2
				&& 0 == strcmp(options.splitter.c_str(), "rkdt") ) {		// rotate on each level
			tmpX.resize(X.size());
			generateRotation(dim, rw);
			rotatePoints( &(X[0]), numof_points, dim, rw, &(tmpX[0]) );
		}

		#if STAGE_STREE_OUTPUT_VERBOSE
		if(worldrank == worldsize-1) { //&& inNode->level < 3) {
			cout<<"    >> Stree: Build: level "<<inNode->level
				<<": rotate points done! -> "<<omp_get_wtime()-stage_t<<endl;
			stage_t = omp_get_wtime();
		}
		#endif

		// 2. choose projection direction and project values
		//inNode->proj.resize(dim);
		if(0 == strcmp(options.splitter.c_str(), "rsmt")) {
			inNode->proj.resize(dim);
			inNode->coord_mv = -1;
			mtreeProjection(&(X[0]), numof_points, dim, &(inNode->proj[0]), &(px[0]));
		}
		if(0 == strcmp(options.splitter.c_str(), "rkdt")) {
			if(options.flag_r == 2) {
				inNode->proj.resize(dim);
				maxvarProjection(&(tmpX[0]), numof_points, dim, inNode->coord_mv, &(px[0]));
				//for(int i = 0; i < dim; i++)
				//	inNode->proj[i] = R[inNode->coord_mv*dim+i];
				vector<double> e; e.resize(dim);
				e[inNode->coord_mv] = 1.0;
				RROT_INV(&(e[0]), &(inNode->proj[0]), &(rw[0]));
			}
			else {
				maxvarProjection(&(X[0]), numof_points, dim, inNode->coord_mv, &(px[0]));
				//inNode->proj[ inNode->coord_mv ] = 1.0;
			}
		}

		#if STAGE_STREE_OUTPUT_VERBOSE
			if(worldrank == worldsize-1) { // && inNode->level < 3) {
				cout<<"    >> Stree: Build: level "<<inNode->level
					<<": projection done! -> "<<omp_get_wtime()-stage_t<<endl;
				stage_t = omp_get_wtime();
			}
		#endif

		// 3. find median
		inNode->median = select(px, px.size()/2);

		#if STAGE_STREE_OUTPUT_VERBOSE
			if(worldrank == worldsize-1) { // && inNode->level < 3) {
				cout<<"    >> Stree: Build: level "<<inNode->level
					<<": find median done! -> "<<omp_get_wtime()-stage_t<<endl;
				stage_t = omp_get_wtime();
			}
		#endif

		// 4. assign membership
		//pbinData leftData = new binData;
		//pbinData rightData = new binData;
		vector<int> leftkid_membership;
		vector<int> rightkid_membership;
		assignMembership(px, inNode->median, leftkid_membership, rightkid_membership);

		#if STAGE_STREE_OUTPUT_VERBOSE
			if(worldrank == worldsize-1) { // && inNode->level < 3) {
				cout<<"    >> Stree: Build: level "<<inNode->level
					<<": assign membership done! - kid size: "
					<<leftkid_membership.size()<<" "<<rightkid_membership.size()
					<<" -> "<<omp_get_wtime()-stage_t<<endl;
				stage_t = omp_get_wtime();
			}
		#endif

		// 5. copy data
		copyData(inData, leftkid_membership, leftData);
		copyData(inData, rightkid_membership, rightData);
		delete inData;

		#if STAGE_STREE_OUTPUT_VERBOSE
			if(worldrank == worldsize-1) { // && inNode->level < 3) {
				cout<<"    >> Stree: Build: level "<<inNode->level
					<<": copying data done! -> "<<omp_get_wtime()-stage_t<<endl;
				stage_t = omp_get_wtime();
			}
		#endif

	}	// scope all vectors

	/*
	// 2. find median
	//inNode->median = select(px, px.size()/2);

	#if STAGE_STREE_OUTPUT_VERBOSE
		if(rank == 0) { // && inNode->level < 3) {
			cout<<"    >> Stree: Build: level "<<inNode->level
				<<": find median done! -> "<<omp_get_wtime()-stage_t<<endl;
			stage_t = omp_get_wtime();
		}
	#endif

	// 4. assign membership
	pbinData leftData = new binData;
	pbinData rightData = new binData;
	{	// scope leftkid_membership and rightkid_membership
		vector<int> leftkid_membership;
		vector<int> rightkid_membership;
		assignMembership(px, inNode->median, leftkid_membership, rightkid_membership);
		if(options.debug_verbose == 2) {
			cout<<"  + left kid: ";
			for(int i = 0; i < leftkid_membership.size(); i++)
				cout<<leftkid_membership[i]<<" ";
			cout<<endl;
			cout<<"  + right kid: ";
			for(int i = 0; i < rightkid_membership.size(); i++)
				cout<<rightkid_membership[i]<<" ";
			cout<<endl;
			cout<<"assign membership done!"<<endl;
		}
		if(options.debug_verbose == 1) cout<<"assign membership done!"<<endl;

		#if STAGE_STREE_OUTPUT_VERBOSE
		if(rank == 0) { // && inNode->level < 3) {
			cout<<"    >> Stree: Build: level "<<inNode->level
				<<": assign membership done! -> "<<omp_get_wtime()-stage_t<<endl;
			stage_t = omp_get_wtime();
		}
		#endif

		// 5. copy data
		//pbinData leftData = new binData;
		//pbinData rightData = new binData;
		copyData(inData, leftkid_membership, leftData);
		copyData(inData, rightkid_membership, rightData);
		delete inData;
		if(options.debug_verbose == 2) {
				cout<<"  + left kid: ";
			for(int i = 0; i < leftData->gids.size(); i++)
				cout<<leftData->gids[i]<<" ";
			cout<<endl;
			cout<<"  + right kid: ";
			for(int i = 0; i < rightData->gids.size(); i++)
				cout<<rightData->gids[i]<<" ";
			cout<<endl;
		}
		if(options.debug_verbose == 1) cout<<"copy data done!"<<endl;
	}	// end scope left and right kid membership
	 */


#if STAGE_STREE_OUTPUT_VERBOSE
	if(worldrank == worldsize) { // && inNode->level < 3) {
		cout<<"    >> Stree: Build: level "<<inNode->level
			<<": copy data done! -> "<<omp_get_wtime()-stage_t<<endl;
		stage_t = omp_get_wtime();
	}
#endif

	// 6. recursively build the tree
	inNode->leftNode = new stNode(2*inNode->lnid+0);
	inNode->rightNode = new stNode(2*inNode->lnid+1);
	if(options.debug_verbose == 1) cout<<"start next level insert .."<<endl;
	insert(inNode, inNode->leftNode, leftData, minp, maxlev);
	insert(inNode, inNode->rightNode, rightData, minp, maxlev);

};


/*
void stTree::assignMembership(const vector<double>& px, double median,
							  vector<int>& leftkid_membership, vector<int>& rightkid_membership)
{
	int numof_points = px.size();
	leftkid_membership.reserve(numof_points/2);
	rightkid_membership.reserve(numof_points/2);
	for(int i = 0; i < numof_points; i++) {
		if(px[i] < median) {
			leftkid_membership.push_back(i);
		}
		else {
			rightkid_membership.push_back(i);
		}
	}
	// in case all points are the same
	if(rightkid_membership.size() == numof_points) {
		for(int i = numof_points/2; i < numof_points; i++)
			leftkid_membership.push_back(i);
		rightkid_membership.resize(numof_points - numof_points/2);
	}

	int threshold = numof_points / 5;
	int nleft = leftkid_membership.size();
	int nright = rightkid_membership.size();
	if( std::abs(nleft - nright) > threshold) { // severe load balance
		if(nright > nleft) {
			int nmove = nright - nleft - threshold;
			for(int i = 0; i < nmove; i++) 
				leftkid_membership.push_back( rightkid_membership[nright-1-i] );
			rightkid_membership.resize(nright-nmove);
		}
		else {
			int nmove = nleft - nright - threshold;
			for(int i = 0; i < nmove; i++)
				rightkid_membership.push_back( leftkid_membership[nleft-1-i] );
			leftkid_membership.resize(nleft-nmove);
		}
	}

}
*/


/*
void stTree::assignMembership(const vector<double>& px, 
							 // output
							 double &median,
							 vector<int>& leftkid_membership,
							 vector<int>& rightkid_membership)
{
	int numof_points = px.size();
	pair<double, int> *pobj_px = new pair<double, int> [numof_points];
	#pragma omp parallel if(numof_points > 2000)
	{
		#pragma omp parallel for
		for(int i = 0; i < numof_points; i++) {
			pobj_px[i].first = px[i];
			pobj_px[i].second = i;
		}
	}
	omp_par::merge_sort(&(pobj_px[0]), &(pobj_px[numof_points]));
	
	if(numof_points % 2 == 0) {
		median = (pobj_px[numof_points/2].first + pobj_px[numof_points/2-1].first) / 2.0;
	}
	else {
		median = pobj_px[numof_points/2].first;
	}
	
	int nleft = numof_points / 2;
	int nright = numof_points - nleft;
	leftkid_membership.resize(nleft);
	rightkid_membership.resize(nright);
	#pragma omp parallel if(numof_points > 2000)
	{
		#pragma omp parallel for
		for(int i = 0; i < nleft; i++)
			leftkid_membership[i] = pobj_px[i].second;

		#pragma omp parallel for
		for(int i = 0; i < nright; i++)
			rightkid_membership[i] = pobj_px[nleft+i].second;
	}
	delete [] pobj_px;

}
*/


void stTree::assignMembership(const vector<double>& px, double median,
							  vector<int>& leftkid_membership, vector<int>& rightkid_membership)
{
	int numof_points = px.size();
	leftkid_membership.reserve(numof_points/2);
	rightkid_membership.reserve(numof_points/2);
	vector<int> same_membership;
	same_membership.reserve(numof_points/2);

	for(int i = 0; i < numof_points; i++) {
		double diff = fabs((px[i]-median)/median);
		if(diff < 1.0e-6) {
			same_membership.push_back(i);
		}
		else {
			if(px[i] < median) {
				leftkid_membership.push_back(i);
			}
			else {
				rightkid_membership.push_back(i);
			}
		}
	}
	
	// if there is load imbalance
	if(same_membership.size() > 0) {

/*	
		#if LOAD_BALANCE_VERBOSE
			int rank;
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);
			cout<<"load imbalance found at rank "<<rank<<" npsame: "<<same_membership.size()<<endl;
		#endif
*/		
		int n_move_left = numof_points/2 - leftkid_membership.size();
		int n_move_right = same_membership.size() - n_move_left;
		for(int i = 0; i < n_move_left; i++)
			leftkid_membership.push_back( same_membership[i] );
		for(int i = 0; i < n_move_right; i++)
			rightkid_membership.push_back( same_membership[n_move_left+i] );
	}

}



void stTree::copyData(pbinData inData, vector<int> & membership, pbinData outData)
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

if( dim >= 512 ) {
		int pointSize = dim*sizeof(double);
		#pragma omp for //schedule(dynamic,50)
		for(i = 0; i < membsize; i++) {
			idim = i*dim;
			membi = membership[i];
			membidim = membi*dim;
			outgids[i] = ingids[ membi ];
			memcpy( (void*)&(outX[idim]), (void*)&(inX[membidim]), pointSize );
		}

} else {
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



void stTree::mtreeProjection(double * points, int numof_points, int dim,
						   // output
						   double * proj, double *pv)
{
	// 1. choose projection direction
	vector<double> p1(dim);
	vector<double> p2(dim);
	vector<double> mu(dim);

	mean(points, numof_points, dim, &(mu[0]));
	furthestPoint(points, numof_points, dim, &(mu[0]), &(p1[0]));
	furthestPoint(points, numof_points, dim, &(p1[0]), &(p2[0]));

	for(int i = 0; i < dim; i++)
		proj[i] = p1[i] - p2[i];
	int ONE = 1;
	double norm = ddot(&dim, &(proj[0]), &ONE, &(proj[0]), &ONE);
	norm = sqrt(norm);
	for(int i = 0; i < dim; i++)
		proj[i] /= norm;
	
	#pragma omp parallel if(numof_points > 1000)
	{
		#pragma omp for
		for(int i = 0; i < numof_points; i++)
			pv[i] = ddot(&dim, &(proj[0]), &ONE, &(points[i*dim]), &ONE);
	}

}


void stTree::maxvarProjection(double *points, int numof_points, int dim,
							 // output
							 int &mvind, double *pv)
{
	// 1. compute var for each coord
	vector<double> var(dim);
	vector<double> mu(dim);
	mean(points, numof_points, dim, &(mu[0]));
	parvar(points, numof_points, dim, &(mu[0]), &(var[0]));
	

	// 2. coord with max var 
	vector<double>::iterator it = max_element(var.begin(), var.end());
	mvind = it - var.begin();
	if(options.debug_verbose) cout<<"max coord"<<endl;
	
	// 3. pv
	#pragma omp parallel if(numof_points > 1000)
	{
		#pragma omp for
		for(int i = 0; i < numof_points; i++)
			pv[i] = points[i*dim+mvind];
	}
	if(options.debug_verbose) cout<<"pv"<<endl;
}


void stTree::parvar(double *points, int numof_points, int dim, double *mean, double *var) 
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


void stTree::furthestPoint(// input
							double *points, int numof_points, int dim, double *query,
							// output
							double *furP)
{
	double * dist = new double [numof_points];
	knn::compute_distances(points, query, numof_points, 1, dim, dist);
	double * pdmax = max_element(dist, dist+numof_points);
	int idmax = pdmax - dist;
	for(int i = 0; i < dim; i++)
		furP[i] = points[idmax*dim+i];
	delete [] dist;
}


void stTree::mean(// input
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


// select the kth smallest element in arr
// for median, ks = glb_N / 2
double stTree::select(vector<double> &arr, int ks)
{
/*
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

	if( S_less.size() == ks || arr.size() == 1 || arr.size() == S_less.size() ) return mu;
	else if(S_less.size() > ks) {
		return select(S_less, ks);
	}
	else {
		return select(S_great, ks-S_less.size());
	}
*/
	vector<double> sorted = arr;
	omp_par::merge_sort(sorted.begin(), sorted.end());
	return(sorted[ks]);
}


int stTree::visitGreedy(double *point, int dim, pstNode node)
{
	if(NULL == node->leftNode && NULL == node->rightNode) {		// leaf node
		return node->lnid;
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


// all-2-all case
void stTree::queryGreedy_a2a(int k, vector< pair<double, long> > &results)
{
	double start_t = omp_get_wtime();
	double profile_t = omp_get_wtime();

#if STAGE_STREE_OUTPUT_VERBOSE
	int rank, nproc;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	double stage_t = omp_get_wtime();
#endif

	int dim = leafRefArr[0]->dim;
	int nleaf = leafRefArr.size();

	vector<int> numof_points_per_leaf(nleaf);
    for(int i = 0; i < nleaf; i++) {
		numof_points_per_leaf[i] = leafRefArr[i]->X.size() / dim;
	}


#if LOAD_BALANCE_VERBOSE
	int worldrank, worldsize;
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);

	int min_lnr, max_lnr, avg_lnr = 0;
	max_lnr = *max_element(numof_points_per_leaf.begin(), numof_points_per_leaf.end());
	min_lnr = *min_element(numof_points_per_leaf.begin(), numof_points_per_leaf.end());
	for(int i = 0; i < nleaf; i++)
		avg_lnr += numof_points_per_leaf[i];
	avg_lnr = avg_lnr / nleaf;

	int max_nr, min_nr, avg_nr;
	MPI_Reduce(&max_lnr, &max_nr, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&min_lnr, &min_nr, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&avg_lnr, &avg_nr, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if(worldrank == 0) {
		cout<<"    -- numof_ref_points (per leaf): "
			<<"  min: "<<min_nr
			<<"  max: "<<max_nr
			<<"  avg: "<<avg_nr / (double)worldsize
			<<endl;
		//cout<<"    + numof_query_points (per leaf): "
		//	<<"  min: "<<min_nr
		//	<<"  max: "<<max_nr
		//	<<"  avg: "<<avg_nr / (double)worldsize
		//	<<endl;
	}
#endif


	#if OVERALL_TREE_TIMING_VERBOSE
		profile_t = omp_get_wtime();
	#endif

	// find knn on each leaf
	results.resize(numof_ref_points_in_tree*k);

	int maxdistsize = 0, max_query_size = 0;
    for(int i = 0; i < nleaf; i++) {
		int numR = leafRefArr[i]->X.size() / dim;
		int numQ = leafRefArr[i]->X.size() / dim;
		int dasize = getBlockSize(numR, numQ)*numR;
		if(dasize > maxdistsize) maxdistsize = dasize;
		if(numQ > max_query_size) max_query_size = numQ;
	}
	double *tmpdist = new double [maxdistsize];
	double *snormr = new double [max_query_size];
	double *snormq = new double [max_query_size];

	pair<double, long> *kmin = new pair<double, long>[max_query_size*k];

	for(int i = 0; i < nleaf; i++) {

		int numof_ref_points = leafRefArr[i]->X.size() / dim;
		int numof_query_points = leafRefArr[i]->X.size() / dim;

		if(numof_query_points > 0) {

            directKQuery_small_a2a( &(leafRefArr[i]->X[0]), numof_ref_points, dim, k, kmin, tmpdist, snormr, snormq );
			//directKQueryLowMem(&(leafRefArr[i]->X[0]), &(leafRefArr[i]->X[0]), numof_ref_points, numof_query_points, k, dim, kmin, tmpdist, snormr, snormq);

			#pragma omp parallel if(numof_query_points > 500)
			{
				#pragma omp for
				for(int j = 0; j < numof_query_points; j++) {
					register int lidsjk = leafRefArr[i]->lids[j]*k;
					register int jk = j*k;
					for(int t = 0; t < k; t++) {
						results[ lidsjk+t ].first = kmin[jk+t].first;
						results[ lidsjk+t ].second = ( (kmin[jk+t].second==-1) ? -1 : leafRefArr[i]->gids[kmin[jk+t].second] );
					}
				}
			}

		}	// end if

	}		// end for (i < nleaf)

	delete [] tmpdist;
	delete [] snormr;
	delete [] snormq;

	delete [] kmin;
	
	#if OVERALL_TREE_TIMING_VERBOSE
		Direct_Kernel_T_ += omp_get_wtime() - profile_t;
	#endif


#if STAGE_STREE_OUTPUT_VERBOSE
	if(rank == nproc-1) {
		cout<<"    >> Stree: Query: find knn done! -> "<<omp_get_wtime()-stage_t<<endl;
		stage_t = omp_get_wtime();
	}
#endif

	if(options.debug_verbose == 3) cout<<endl<<"queryGreedy done!"<<endl;

}


void stTree::queryGreedy(pbinData queryData, int k, vector< pair<double, long> > &results)
{
	double start_t = omp_get_wtime();
	double profile_t = omp_get_wtime();

#if STAGE_STREE_OUTPUT_VERBOSE
	int rank, nproc;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	double stage_t = omp_get_wtime();
#endif

	int dim = queryData->dim;
	int numof_queries = queryData->X.size() / dim;

	// if "rkdt, flag_r = 1", rotate query points once
	if( options.flag_r == 1 && 0 == strcmp(options.splitter.c_str(), "rkdt") && depth > 0) {
		newRotatePoints(&(queryData->X[0]), numof_queries, dim, root->rw);

        //vector<double> tmpX(queryData->X.size());
		//memcpy(&(tmpX[0]), &(queryData->X[0]), sizeof(double)*(queryData->X.size()));
		//rotatePoints( &(tmpX[0]), numof_queries, dim, root->rw, &(queryData->X[0]) );
		//tmpX.clear();
	}

#if _STTREE_DEBUG_
cout<<"\t\t\t(queryGreedy) rotate points done "<<omp_get_wtime()-start_t<<endl;
start_t = omp_get_wtime();
#endif

	vector<int> membership(numof_queries);
	int nleaf = leafRefArr.size();
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


#if STAGE_STREE_OUTPUT_VERBOSE
	if(rank == nproc-1) {
		cout<<"    >> Stree: Query: assgin membership done! -> "<<omp_get_wtime()-stage_t<<endl;
		stage_t = omp_get_wtime();
	}
#endif


	if(options.debug_verbose == 2) {
		cout<<endl<<"  + query membership: ";
		for(int i = 0; i < numof_queries; i++)
			cout<<membership[i]<<" ";
		cout<<endl;
		cout<<endl<<"  + query size: ";
		for(int i = 0; i < numof_queries_per_leaf.size(); i++)
			cout<<numof_queries_per_leaf[i]<<" ";
		cout<<endl;
	}
	if(options.debug_verbose == 3) cout<<endl<<"membership done!"<<endl;

	if(options.timing_verbose) {
		cout<<"   -> visitGreedy time: "<<omp_get_wtime()-start_t<<endl;
		start_t = omp_get_wtime();
	}


#if _STTREE_DEBUG_
cout<<"\t\t\t(queryGreedy) membership done "<<omp_get_wtime()-start_t<<endl;
start_t = omp_get_wtime();
#endif

	// allocate memory for query data.
	vector<pbinData> leafQueryArr(nleaf);
	for(int i = 0; i < nleaf; i++) {
		leafQueryArr[i] = new binData();
		leafQueryArr[i]->X.reserve( numof_queries_per_leaf[i]*dim );
		leafQueryArr[i]->gids.reserve( numof_queries_per_leaf[i] );
		leafQueryArr[i]->lids.reserve( numof_queries_per_leaf[i] );
		leafQueryArr[i]->dim = dim;
	}
	if(options.debug_verbose == 3) cout<<endl<<"allocate memory done!"<<endl;

	// redistribute query data to leafQueryArr[]
	for(int i = 0; i < numof_queries; i++) {
		for(int j = 0; j < dim; j++)
			leafQueryArr[membership[i]]->X.push_back( queryData->X[i*dim+j] );
		leafQueryArr[membership[i]]->gids.push_back( queryData->gids[i] );
		leafQueryArr[membership[i]]->lids.push_back( (long)i );
	}
	if(options.debug_verbose == 3) cout<<endl<<"redistribute data done!"<<endl;

	if(options.timing_verbose) {
		cout<<"   -> redistribute data time: "<<omp_get_wtime()-start_t<<endl;
		start_t = omp_get_wtime();
	}


#if _STTREE_DEBUG_
cout<<"\t\t\t(queryGreedy) redistribute data done "<<omp_get_wtime()-start_t<<endl;
start_t = omp_get_wtime();
#endif

#if STAGE_STREE_OUTPUT_VERBOSE
	if(rank == nproc-1) {
		cout<<"    >> Stree: Query: distribute query to leaf done! -> "<<omp_get_wtime()-stage_t<<endl;
		stage_t = omp_get_wtime();
	}
#endif

#if LOAD_BALANCE_VERBOSE
	int worldrank, worldsize;
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);

	vector<int> numof_points_per_leaf(nleaf);
	for(int i = 0; i < nleaf; i++) {
		numof_points_per_leaf[i] = leafRefArr[i]->X.size() / dim;
	}
	int min_lnr, max_lnr, avg_lnr = 0;
	max_lnr = *max_element(numof_points_per_leaf.begin(), numof_points_per_leaf.end());
	min_lnr = *min_element(numof_points_per_leaf.begin(), numof_points_per_leaf.end());
	for(int i = 0; i < nleaf; i++)
		avg_lnr += numof_points_per_leaf[i];
	avg_lnr = avg_lnr / nleaf;
	int max_nr, min_nr, avg_nr;
	MPI_Reduce(&max_lnr, &max_nr, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&min_lnr, &min_nr, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&avg_lnr, &avg_nr, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	int min_lnq, max_lnq, avg_lnq = 0;
	max_lnq = *max_element(numof_queries_per_leaf.begin(), numof_queries_per_leaf.end());
	min_lnq = *min_element(numof_queries_per_leaf.begin(), numof_queries_per_leaf.end());
	for(int i = 0; i < nleaf; i++)
		avg_lnq += numof_queries_per_leaf[i];
	avg_lnq = avg_lnq / nleaf;
	int max_nq, min_nq, avg_nq;
	MPI_Reduce(&max_lnq, &max_nq, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&min_lnq, &min_nq, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&avg_lnq, &avg_nq, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	if(worldrank == 0) {
		cout<<"      -- numof_ref_points (per leaf): "
			<<"  min: "<<min_nr
			<<"  max: "<<max_nr
			<<"  avg: "<<avg_nr / (double)worldsize
			<<endl;
		cout<<"      -- numof_query_points (per leaf): "
			<<"  min: "<<min_nq
			<<"  max: "<<max_nq
			<<"  avg: "<<avg_nq / (double)worldsize
			<<endl;
	}
#endif

	#if OVERALL_TREE_TIMING_VERBOSE
		profile_t = omp_get_wtime();
	#endif

	// find knn on each leaf
	results.resize(numof_queries*k);

	int maxdistsize = 0, max_ref_size = 0, max_query_size = 0;
	for(int i = 0; i < nleaf; i++) {
		int numR = leafRefArr[i]->X.size() / dim;
		int numQ = leafQueryArr[i]->X.size() / dim;
		int dasize = getBlockSize(numR, numQ)*numR;
		if(dasize > maxdistsize) maxdistsize = dasize;
		if(numR > max_ref_size) max_ref_size = numR;
		if(numQ > max_query_size) max_query_size = numQ;
	}
	double *tmpdist = new double [maxdistsize];
	double *snormr = new double [max_ref_size];
	double *snormq = new double [max_query_size];

	pair<double, long> *kmin = new pair<double, long>[max_query_size*k];

	for(int i = 0; i < nleaf; i++) {

		double knn_t = omp_get_wtime();

		int numof_ref_points = leafRefArr[i]->X.size() / dim;
		int numof_query_points = leafQueryArr[i]->X.size() / dim;
		if(numof_query_points > 0) {
			directKQueryLowMem(&(leafRefArr[i]->X[0]), &(leafQueryArr[i]->X[0]), numof_ref_points, numof_query_points, k, dim, kmin, tmpdist, snormr, snormq);
			#pragma omp parallel if(numof_query_points > 500)
			{
				#pragma omp for
				for(int j = 0; j < numof_query_points; j++) {
					register int lidsjk = leafQueryArr[i]->lids[j]*k;
					register int jk = j*k;
					for(int t = 0; t < k; t++) {
						results[ lidsjk+t ].first = kmin[jk+t].first;
						results[ lidsjk+t ].second = ( (kmin[jk+t].second == -1) ? -1 : leafRefArr[i]->gids[kmin[jk+t].second] );
					}
				}
			}
		}	// end if
		if(options.timing_verbose > 1) {
			cout<<"      -> on leaf node ("<<i<<"), knn find time: "<<omp_get_wtime()-knn_t<<endl;
		}

	}	// end for (i < nleaf)


#if _STTREE_DEBUG_
cout<<"\t\t\t(queryGreedy) search each leaf "<<omp_get_wtime()-start_t<<endl;
#endif

	delete [] tmpdist;
	delete [] snormr;
	delete [] snormq;
	delete [] kmin;
	for(int i = 0; i < nleaf; i++) {
		delete leafQueryArr[i];
		leafQueryArr[i] = NULL;
	}

	#if OVERALL_TREE_TIMING_VERBOSE
		Direct_Kernel_T_ += omp_get_wtime() - profile_t;
	#endif

#if STAGE_STREE_OUTPUT_VERBOSE
	if(rank == nproc-1) {
		cout<<"    >> Stree: Query: find knn done! -> "<<omp_get_wtime()-stage_t<<endl;
		stage_t = omp_get_wtime();
	}
#endif

	if(options.debug_verbose == 3) cout<<endl<<"queryGreedy done!"<<endl;

}



void stTree::randperm(int m, int N, vector<int>& arr)
{

	if(m > N) { 
		cerr<<" m must <= N"<<endl;
		return;
	}

	arr.resize(m);
	for(int i = 0; i <arr.size(); i++) {
		double tmp = floor( (double)N*(double)rand()/(double)RAND_MAX );
		arr[i] = (int)tmp;
	}
	sort(arr.begin(), arr.end());
	vector<int>::iterator it = unique(arr.begin(), arr.end());
	arr.resize(it - arr.begin());

	int pp = m;
	while(arr.size() < m) {
		pp++;
		double tmp = floor( (double)N*(double)rand()/(double)RAND_MAX );
		arr.push_back((int)tmp);
		sort(arr.begin(), arr.end());
		vector<int>::iterator it = unique(arr.begin(), arr.end());
		arr.resize(it - arr.begin());
	}
}


void stTree::sampleNode(pstNode node, vector<double> &samples)
{
	int dim = leafRefArr[0]->dim;
	int nleaf = leafRefArr.size();
	int numof_nodes_this_level = (int)pow(2.0, (double)(node->level));
	int numof_buckets_per_node = leafRefArr.size() / numof_nodes_this_level;
	int left = node->lnid * numof_buckets_per_node;
	int right = left + numof_buckets_per_node - 1;

	vector<int> inScan(right-left+1+1);	// inclusive scan of number of points per buckets, first is 0
	inScan[0] = 0;
	for(int i = 0; i < inScan.size()-1; i++) 
		inScan[i+1] = inScan[i] + leafRefArr[left+i]->X.size() / dim;
	int numof_points = inScan[inScan.size()-1];
	int numof_samples = ceil( sqrt((double)dim)*log10((double)numof_points)/log10(2.0) );
	numof_samples = max(numof_samples, 1);
	numof_samples = min(numof_samples, numof_points);
	vector<int> sampleIDs(numof_samples);
	randperm(numof_samples, numof_points, sampleIDs);

	if(options.debug_verbose == 2) {
		cout<<"  + sample ids: ";
		for(int i = 0; i < sampleIDs.size(); i++)
			cout<<sampleIDs[i]<<" ";
		cout<<endl;
	}

	samples.resize(numof_samples*dim);
	for(int i = 0; i < numof_samples; i++) {
		int glb_bucket_id = -1;
		int local_bucket_id = -1;
		for(int j = 1; j < inScan.size(); j++) {
			if(sampleIDs[i] < inScan[j]) {
				local_bucket_id = j - 1;
				glb_bucket_id = left + local_bucket_id;
				break;
			}
		}
		int local_arr_id = sampleIDs[i] - inScan[local_bucket_id+1-1];

		if(options.debug_verbose == 2) {
			cout<<"  - i: "<<i<<" glb_bucket_id: "<<glb_bucket_id<<" local_arr_id: "<<local_arr_id<<endl;
		}

		for(int t = 0; t < dim; t++)
			samples[i*dim+t] = leafRefArr[glb_bucket_id]->X[local_arr_id*dim+t];
	}	// end for(i < numof_samples)
}


// "data" will be deleted inside this function
void stTree::visitSampling(pbinData data, pstNode node, int *membership)
{
	int dim = data->dim;
	int numof_points = data->X.size() / dim;

	if(options.debug_verbose == 2) cout<<"  + level: "<<node->level<<" lnid: "<<node->lnid<<endl;

	if(NULL == node->leftNode && NULL == node->rightNode) {
		for(int i = 0; i < numof_points; i++) 
			membership[ data->lids[i] ] = node->lnid;
		delete data;
	}
	else {
		// 1. sample two kids
		vector<double> leftSamples;
		vector<double> rightSamples;
		sampleNode(node->leftNode, leftSamples);
		sampleNode(node->rightNode, rightSamples);
		int numof_left_points = leftSamples.size() / dim;
		int numof_right_points = rightSamples.size() / dim;
		if(options.debug_verbose == 2) {
			cout<<"  + left samples: ";
			for(int i = 0; i < leftSamples.size(); i++)
				cout<<leftSamples[i]<<" ";
			cout<<endl;
			cout<<"  + right samples: ";
			for(int i = 0; i < rightSamples.size(); i++)
				cout<<rightSamples[i]<<" ";
			cout<<endl;
		}
		if(options.debug_verbose == 3) cout<<endl<<"  + sample two kids done!"<<endl;

		// 2. compute distance
		//vector<double> leftDist(numof_points*numof_left_points);
		//vector<double> rightDist(numof_points*numof_right_points);
		//compute_distances(&(leftSamples[0]), &(data->X[0]), numof_left_points, numof_points, dim, &(leftDist[0]));
		//compute_distances(&(rightSamples[0]), &(data->X[0]), numof_right_points, numof_points, dim, &(rightDist[0]));
		pair<double, long> *haus0 = new pair<double, long> [numof_points];
		pair<double, long> *haus1 = new pair<double, long> [numof_points];
		directKQueryLowMem(&(leftSamples[0]), &(data->X[0]), numof_left_points, numof_points, 1, dim, haus0);
		directKQueryLowMem(&(rightSamples[0]), &(data->X[0]), numof_right_points, numof_points, 1, dim, haus1);

		if(options.debug_verbose == 3) cout<<"  + compute distance done!"<<endl;

		// 3. assign points to nearest kid
		int numof_kids = 2;
		vector<int> * members_in_kid = new vector<int> [numof_kids];
		for(int i = 0; i < numof_kids; i++)
			members_in_kid[i].reserve(numof_points/numof_kids);

		for(int i = 0; i < numof_points; i++) {
			//double leftHaus = *min_element( leftDist.begin()+i*numof_left_points, leftDist.begin()+(i+1)*numof_left_points );
			//double rightHaus = *min_element( rightDist.begin()+i*numof_right_points, rightDist.begin()+(i+1)*numof_right_points );
			double leftHaus = haus0[i].first;
			double rightHaus = haus1[i].first;
			if(leftHaus < rightHaus) {
				members_in_kid[ 0 ].push_back(i);
			}
			else {
				members_in_kid[ 1 ].push_back(i);
			}
			if(options.debug_verbose == 2) {
				cout<<"  + i: "<<i<<" leftHaus: "<<leftHaus<<" rightHaus: "<<rightHaus<<endl;
			}
		}
		if(options.debug_verbose == 3) cout<<"  + assign points to kid done!"<<endl;

		delete [] haus0;
		delete [] haus1;

		// 4. rearrange data
		pbinData leftData = new binData();
		leftData->dim = dim;
		leftData->X.resize( members_in_kid[0].size() * dim);
		leftData->lids.resize( members_in_kid[0].size() );
		for(int i = 0; i < members_in_kid[0].size(); i++) {
			leftData->lids[i] = data->lids[ members_in_kid[0].at(i) ];
			for(int j = 0; j < dim; j++)
				leftData->X[i*dim+j] = data->X[ members_in_kid[0].at(i)*dim+j ];
		}
	
		pbinData rightData = new binData();
		rightData->dim = dim;
		rightData->X.resize( members_in_kid[1].size() * dim);
		rightData->lids.resize( members_in_kid[1].size() );
		for(int i = 0; i < members_in_kid[1].size(); i++) {
			rightData->lids[i] = data->lids[ members_in_kid[1].at(i) ];
			for(int j = 0; j < dim; j++)
				rightData->X[i*dim+j] = data->X[ members_in_kid[1].at(i)*dim+j ];
		}
		delete data;
		if(options.debug_verbose == 3) cout<<"  + rearrange data done!"<<endl;

		// 5. recursively assign membership
		visitSampling(leftData, node->leftNode, membership);
		visitSampling(rightData, node->rightNode, membership);
	}	// end else
}


void stTree::querySampling(pbinData queryData, int k, vector< pair<double, long> > &results)
{

	double start_t = omp_get_wtime();

	int dim = queryData->dim;
	int numof_queries = queryData->X.size() / dim;
	
	vector<int> membership(numof_queries);
	int nleaf = leafRefArr.size();
	vector<int> numof_queries_per_leaf;
	numof_queries_per_leaf.resize(nleaf);

	pbinData queryData_clone = new binData;
	queryData_clone->Copy(queryData);
	queryData_clone->lids.resize(numof_queries);
	#pragma omp parallel for
	for(int i = 0; i < numof_queries; i++)
		queryData_clone->lids[i] = i;

	
	visitSampling(queryData_clone, root, &(membership[0]));
	// -> after call visitSampling(), queryData_clone is deleted
	
	if(options.timing_verbose) {
		cout<<"   -> visitSampling time: "<<omp_get_wtime()-start_t<<endl;
		start_t = omp_get_wtime();
	}

	for(int i = 0; i < numof_queries; i++) 
		numof_queries_per_leaf[membership[i]]++;
	
	if(options.debug_verbose == 3) cout<<endl<<"querySampling membership done!"<<endl;
	if(options.debug_verbose == 2) {
		cout<<endl<<"  + query membership: ";
		for(int i = 0; i < numof_queries; i++)
			cout<<membership[i]<<" ";
		cout<<endl;
		cout<<endl<<"  + query size: ";
		for(int i = 0; i < numof_queries_per_leaf.size(); i++)
			cout<<numof_queries_per_leaf[i]<<" ";
		cout<<endl;
	}
	
	if(STTREE_LOAD_BALANCE_VERBOSE) {
		int max_n = *max_element(numof_queries_per_leaf.begin(), numof_queries_per_leaf.end());
		int min_n = *min_element(numof_queries_per_leaf.begin(), numof_queries_per_leaf.end());
		double avg_n = 0.0;
		for(int i = 0; i < numof_queries_per_leaf.size(); i++)
			avg_n += (double)numof_queries_per_leaf[i];
		avg_n /= (double)nleaf;
		cout<<"  + min query size: "<<min_n
			<<"  max query size: "<<max_n
			<<"  avg query size: "<<avg_n
			<<endl;
	}



	// allocate memory for query data.
	vector<pbinData> leafQueryArr(nleaf);
	for(int i = 0; i < nleaf; i++) {
		leafQueryArr[i] = new binData();
		leafQueryArr[i]->X.reserve( numof_queries_per_leaf[i]*dim );
		leafQueryArr[i]->gids.reserve( numof_queries_per_leaf[i] );
		leafQueryArr[i]->lids.reserve( numof_queries_per_leaf[i] );
		leafQueryArr[i]->dim = dim;
	}
	if(options.debug_verbose == 3) cout<<endl<<"allocate memory done!"<<endl;
	
	// redistribute query data to leafQueryArr[]
	for(int i = 0; i < numof_queries; i++) {
		for(int j = 0; j < dim; j++)
			leafQueryArr[membership[i]]->X.push_back( queryData->X[i*dim+j] );
		leafQueryArr[membership[i]]->gids.push_back( queryData->gids[i] );
		leafQueryArr[membership[i]]->lids.push_back( i ); 
	}
	if(options.debug_verbose == 3) cout<<endl<<"redistribute data done!"<<endl;
	
	if(options.timing_verbose) {
		cout<<"   -> redistribute data time: "<<omp_get_wtime()-start_t<<endl;
		start_t = omp_get_wtime();
	}


	// find knn on each leaf
	results.resize(numof_queries*k);
	int max_query_size = *max_element(numof_queries_per_leaf.begin(), numof_queries_per_leaf.end());
	pair<double, long> *kmin = new pair<double, long>[max_query_size*k];

	//vector<int> numof_ref_per_leaf(nleaf);
	//for(int i = 0; i < nleaf; i++)
	//	numof_ref_per_leaf[i] = leafRefArr[i]->X.size() / dim;
	//int max_ref_size = *max_element(numof_ref_per_leaf.begin(), numof_ref_per_leaf.end());
	//vector<double> tmp_dist(max_ref_size * max_query_size);
	//vector<double> tmp_sqnormr(max_ref_size);
	//vector<double> tmp_sqnormq(max_query_size);

	for(int i = 0; i < nleaf; i++) {
		
		double knn_t = omp_get_wtime();
		
		int numof_ref_points = leafRefArr[i]->X.size() / dim;
		int numof_query_points = leafQueryArr[i]->X.size() / dim;
		if(numof_query_points > 0) {
			directKQueryLowMem(&(leafRefArr[i]->X[0]), &(leafQueryArr[i]->X[0]), numof_ref_points, numof_query_points, k, dim, kmin);
			//directKQueryLowMem(&(leafRefArr[i]->X[0]), &(leafQueryArr[i]->X[0]), numof_ref_points, numof_query_points, k, dim, kmin, &(tmp_dist[0]), &(tmp_sqnormr[0]), &(tmp_sqnormq[0]));
			
			#pragma omp parallel for
			for(int j = 0; j < numof_query_points; j++) {
				for(int t = 0; t < k; t++) {
					results[ leafQueryArr[i]->lids[j]*k+t ].first = kmin[j*k+t].first;
					results[ leafQueryArr[i]->lids[j]*k+t ].second = leafRefArr[i]->gids[kmin[j*k+t].second];
				}
			}
		} // enf if

		if(options.timing_verbose > 1) {
			cout<<"      -> on leaf node ("<<i<<"), knn find time: "<<omp_get_wtime()-knn_t<<endl;
		}


	}
	if(options.debug_verbose == 3) cout<<endl<<"knn search done!"<<endl;
	
	if(options.timing_verbose) {
		cout<<"   -> find knn time: "<<omp_get_wtime()-start_t<<endl;
	}


	delete [] kmin;
	for(int i = 0; i < nleaf; i++) {
		delete leafQueryArr[i];
		leafQueryArr[i] = NULL;
	}
	if(options.debug_verbose == 3) cout<<endl<<"querySampling done!"<<endl;

}


