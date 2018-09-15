#include <blas.h>
#include <omp.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <stdio.h>
#include <ompUtils.h>
#include <float.h>
#include <cstring>

#include "fks_ompNode.h"

using namespace std;

#define _ompnode_debug_ true

void print(fksData *data, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
	MPI_Barrier(comm);

    int dim = data->dim;

    if(data->charges.size() == data->numof_points && data->mortons.size() == 0) {
        for(int k = 0; k < size; k++) {
            if(k == rank) {
	            for(int i = 0; i < data->numof_points; i++) {
		            cout<<"(rank "<<rank<<") "<<data->gids[i]<<": ("<<data->charges[i]<<") ";
		            for(int j = 0; j < dim; j++)
			            cout<<data->X[i*dim+j]<<" ";
		            cout<<endl;
	            }
                cout.flush();
            }
	        MPI_Barrier(comm);
        }
    }
    else if(data->charges.size() == 0 && data->mortons.size() == data->numof_points) {
        for(int k = 0; k < size; k++) {
            if(k == rank) {
	            for(int i = 0; i < data->numof_points; i++) {
		            cout<<"(rank "<<rank<<") "<<data->gids[i]<<": ("<<data->mortons[i]<<") ";
		            for(int j = 0; j < dim; j++)
			            cout<<data->X[i*dim+j]<<" ";
		            cout<<endl;
	            }
                cout.flush();
            }
	        MPI_Barrier(comm);
        }
    }
    else if(data->charges.size() == data->numof_points && data->mortons.size() == data->numof_points) {
        for(int k = 0; k < size; k++) {
            if(k == rank) {
	            for(int i = 0; i < data->numof_points; i++) {
		            cout<<"(rank "<<rank<<") "<<data->gids[i]
                        <<": ("<<data->mortons[i]<<" - "<<data->charges[i]<<") ";
		            for(int j = 0; j < dim; j++)
			            cout<<data->X[i*dim+j]<<" ";
		            cout<<endl;
	            }
                cout.flush();
            }
	        MPI_Barrier(comm);
        }
    }
    else {
        for(int k = 0; k < size; k++) {
            if(k == rank) {
	            for(int i = 0; i < data->numof_points; i++) {
		            cout<<"(rank "<<rank<<") "<<data->gids[i]<<": ";
		            for(int j = 0; j < dim; j++)
			            cout<<data->X[i*dim+j]<<" ";
		            cout<<endl;
	            }
                cout.flush();
            }
	        MPI_Barrier(comm);
        }
    }
    if(rank == 0) cout<<endl;
}


fks_ompNode::~fks_ompNode()
{
    if(leftNode != NULL) {
        delete leftNode;
        leftNode = NULL;
    }
    if(rightNode != NULL) {
        delete rightNode;
        rightNode = NULL;
    }
    if(skeletons != NULL) {
        delete skeletons;
        skeletons = NULL;
    }
 }

void fks_ompNode::insert(fks_ompNode *in_parent, fksData *inData,
                         vector<long> *&active_set, long morton_offset,
                         int mppn, int maxlevel)
{
	// input checks
	assert( mppn > 1 );
	assert( maxlevel >= 0 );

	// .0 initializations
    if (in_parent!=NULL)  {
		level = in_parent->level + 1;
		parent = in_parent;
	}
    else {      // if root
        if(inData->mortons.size() == 0) {
            inData->mortons.resize(inData->numof_points);
            memset( &(inData->mortons[0]), 0, sizeof(long)*inData->numof_points );
        }
    }

	int dim = inData->dim;
	int numof_points = active_set->size();

    cout<<"@level: "<<level<<", leaf node id: "<<lnid
                <<", level = "<<level<<", mlevel = "<<maxlevel
                <<", numof_points = "<<numof_points<<", mppn = "<<mppn
                <<endl;

	// base case to terminate recursion
	if( level == maxlevel || numof_points <= mppn )
    {

        if(_ompnode_debug_) {
            cout<<"@level: "<<level<<", leaf node id: "<<lnid
                <<"level = "<<level<<", mlevel = "<<maxlevel
                <<"numof_points = "<<numof_points<<", mppn = "<<mppn
                <<endl;
            cout<<"member: ";
            for(int i = 0; i < active_set->size(); i++) {
                cout<<"("<<(*active_set)[i]<<", "
                    <<inData->mortons[ (*active_set)[i] ]<<") ";
            }
            cout<<endl;
        }

        leaf_point_local_id.resize(numof_points);
        for(int i = 0; i < numof_points; i++)
            leaf_point_local_id[i] = (*active_set)[i];
        delete active_set; active_set = NULL;
		return;
    }

    // .1 generate projection direction and project points
    vector<double> px(numof_points);
    proj.resize(dim);
    mtreeProjection(&(inData->X[0]), numof_points, dim, &(proj[0]), &(px[0]));

    // .2 find median
    //median = select(px, px.size()/2);
    median = getMedian(px);

    // .3 assign membership and set morton id
	vector<int> leftkid_membership;
	vector<int> rightkid_membership;
	assignMembership(px, median, leftkid_membership, rightkid_membership);
    vector<long> *left_active_set = new vector<long>();
    vector<long> *right_active_set = new vector<long>();
    left_active_set->resize(leftkid_membership.size());
    right_active_set->resize(rightkid_membership.size());
    #pragma omp parallel if (leftkid_membership.size() > 5000)
    {
        #pragma omp for
        for(int i = 0; i < leftkid_membership.size(); i++) {
            long idx = (*active_set)[leftkid_membership[i]];
            (*left_active_set)[i] = idx;
            inData->mortons[idx] = setBitZero(inData->mortons[idx], level+1+morton_offset);
        }
    }
    #pragma omp parallel if (rightkid_membership.size() > 5000)
    {
        #pragma omp for
        for(int i = 0; i < rightkid_membership.size(); i++) {
            long idx = (*active_set)[rightkid_membership[i]];
            (*right_active_set)[i] = idx;
            inData->mortons[idx] = setBitOne(inData->mortons[idx], level+1+morton_offset);
        }
    }
    delete active_set; active_set=NULL;

    if(_ompnode_debug_) {
        cout<<"@level: "<<level<<", node id: "<<lnid<<endl;
        cout<<"left_member: ";
        for(int i = 0; i < left_active_set->size(); i++) {
            cout<<"("<<(*left_active_set)[i]<<", "
                <<px[ (*left_active_set)[i] ]<<", "
                <<inData->mortons[ (*left_active_set)[i] ]<<") ";
        }
        cout<<endl;
        cout<<"right_member: ";
        for(int i = 0; i < right_active_set->size(); i++) {
            cout<<"("<<(*right_active_set)[i]<<", "
                <<px[ (*right_active_set)[i] ]<<", "
                <<inData->mortons[ (*right_active_set)[i] ]<<") ";
        }
        cout<<endl;
    }

	// .4 recursively build the tree
	leftNode = new fks_ompNode(2*lnid+0);
    leftNode->node_morton = setBitZero(node_morton, morton_offset+1+level);
	rightNode = new fks_ompNode(2*lnid+1);
    rightNode->node_morton = setBitOne(node_morton, morton_offset+1+level);
	leftNode->insert(this, inData, left_active_set, morton_offset, mppn, maxlevel);
	rightNode->insert(this, inData, right_active_set, morton_offset, mppn, maxlevel);
};




void fks_ompNode::furthestPoint(double *points, int numof_points, int dim, double *query, double *furP)
{
	double * dist = new double [numof_points];
	knn::compute_distances(points, query, numof_points, 1, dim, dist);
	double * pdmax = max_element(dist, dist+numof_points);
	int idmax = pdmax - dist;
	for(int i = 0; i < dim; i++)
		furP[i] = points[idmax*dim+i];
	delete [] dist;
}


void fks_ompNode::mean(double *points, int numof_points, int dim, double *mu)
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


void fks_ompNode::mtreeProjection(double *points, int numof_points, int dim,
                                    double *proj, double *pv)
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


void fks_ompNode::assignMembership(const vector<double>& px, double median,
							 vector<int>& leftkid_membership, vector<int>& rightkid_membership)
{
	int numof_points = px.size();
	leftkid_membership.reserve(numof_points/2);
	rightkid_membership.reserve(numof_points/2);
	vector<int> same_membership;
	same_membership.reserve(numof_points/2);

    for(int i = 0; i < numof_points; i++) {
		double diff = fabs((px[i]-median)/median);
        if(diff < 1.0e-6 || isnan(diff) || isinf(diff) ) {   // isnan(diff) when median = 0
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
		int n_move_left = numof_points/2 - leftkid_membership.size();
		int n_move_right = same_membership.size() - n_move_left;
		for(int i = 0; i < n_move_left; i++)
			leftkid_membership.push_back( same_membership[i] );
		for(int i = 0; i < n_move_right; i++)
			rightkid_membership.push_back( same_membership[n_move_left+i] );
	}

}


double fks_ompNode::select(vector<double> &arr, int ks)
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


double fks_ompNode::getMedian(vector<double> &arr)
{
    vector<double> sorted = arr;
	omp_par::merge_sort(sorted.begin(), sorted.end());

    int n = arr.size();
    if(n%2 == 0) {  // even
        return ( (sorted[n/2-1]+sorted[n/2])/2.0 );
    }
    else {
        return ( sorted[n/2] );
    }
}

void fks_ompNode::getLeafData(fksData *inData, fksData *outData)
{
	int dim = inData->dim;
	outData->X.resize(leaf_point_local_id.size()*dim);
	outData->gids.resize(leaf_point_local_id.size());
    outData->dim = dim;
	outData->numof_points = leaf_point_local_id.size();
	int membsize = leaf_point_local_id.size();

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
			    membi = leaf_point_local_id[i];
			    membidim = membi*dim;
			    outgids[i] = ingids[ membi ];
			    memcpy( (void*)&(outX[idim]), (void*)&(inX[membidim]), pointSize );
		    }

        } else {
		    #pragma omp for //schedule(dynamic,50)
		    for(i = 0; i < membsize; i++) {
			    idim = i*dim;
			    membi = leaf_point_local_id[i];
			    membidim = membi*dim;
			    outgids[i] = ingids[ membi ];
			    #pragma vector
			    #pragma ivdep
			    for(j = 0; j < dim; j++)
				    outX[idim+j] = inX[membidim+j];
		    }

	    }
    }

    if(inData->charges.size() > 0) {
		outData->charges.resize(leaf_point_local_id.size());
		#pragma omp parallel if(membsize > 1000)
		{
			int membi;
			int membidim;
			double *incharges = &(inData->charges[0]);
			double *outcharges = &(outData->charges[0]);
			int i, j;
			#pragma omp for //schedule(dynamic,50)
			for(i = 0; i < membsize; i++) {
				outcharges[i] = incharges[ leaf_point_local_id[i] ];
			}
		}
	}	// end if

    if(inData->mortons.size() > 0) {
		outData->mortons.resize(leaf_point_local_id.size());
		#pragma omp parallel if(membsize > 1000)
		{
			int membi;
			int membidim;
			long *inmortons = &(inData->mortons[0]);
			long *outmortons = &(outData->mortons[0]);
			int i, j;
			#pragma omp for //schedule(dynamic,50)
			for(i = 0; i < membsize; i++) {
				outmortons[i] = inmortons[ leaf_point_local_id[i] ];
			}
		}
	}	// end if

    if(inData->lids.size() > 0) {
		outData->lids.resize(leaf_point_local_id.size());
		#pragma omp parallel if(membsize > 1000)
		{
			int membi;
			int membidim;
			long *inlids = &(inData->lids[0]);
			long *outlids = &(outData->lids[0]);
			int i, j;
			#pragma omp for //schedule(dynamic,50)
			for(i = 0; i < membsize; i++) {
				outlids[i] = inlids[ leaf_point_local_id[i] ];
			}
		}
	}	// end if
}


long fks_ompNode::setBitZero(long input, long bitpos)
{
    long output = ( input & (~(1 << bitpos)) );
    return output;
}


long fks_ompNode::setBitOne(long input, long bitpos)
{
    long output = ( input | (1 << bitpos) );
    return output;
}







/*
void fks_ompNode::insert(fks_ompNode *in_parent, fksData *inData,
                            vector<long> *&active_set, int mppn, int maxlevel)
{
    double start_t = omp_get_wtime();

	// input checks
	assert( mppn > 1 );
	assert( maxlevel >= 0 );

	// .0 initializations
    if (in_parent!=NULL)  {
		level = in_parent->level + 1;
		parent = in_parent;
	}

	int dim = inData->dim;
	int numof_points = active_set->size();

	// base case to terminate recursion
	if( level == maxlevel || numof_points <= mppn )
    {
        leaf_point_local_id.resize(numof_points);
        for(int i = 0; i < numof_points; i++)
            leaf_point_local_id[i] = (*active_set)[i];
        delete active_set; active_set = NULL;
		return;
    }

    // .1 generate projection direction and project points
    vector<double> px(numof_points);
    proj.resize(dim);
    mtreeProjection(&(inData->X[0]), numof_points, dim, &(proj[0]), &(px[0]));

    // .2 find median
    //median = select(px, px.size()/2);
    median = getMedian(px);

    // .3 assign membership
	vector<int> leftkid_membership;
	vector<int> rightkid_membership;
	assignMembership(px, median, leftkid_membership, rightkid_membership);
    vector<long> *left_active_set = new vector<long>();
    vector<long> *right_active_set = new vector<long>();
    left_active_set->resize(leftkid_membership.size());
    right_active_set->resize(rightkid_membership.size());
    #pragma omp parallel if (leftkid_membership.size() > 5000)
    {
        #pragma omp for
        for(int i = 0; i < leftkid_membership.size(); i++)
            (*left_active_set)[i] = (*active_set)[leftkid_membership[i]];
    }
    #pragma omp parallel if (rightkid_membership.size() > 5000)
    {
        #pragma omp for
        for(int i = 0; i < rightkid_membership.size(); i++)
            (*right_active_set)[i] = (*active_set)[rightkid_membership[i]];
    }
    delete active_set; active_set=NULL;

    if(_ompnode_debug_) {
        cout<<"@level: "<<level<<", node id: "<<lnid<<endl;
        cout<<"left_member: ";
        for(int i = 0; i < left_active_set->size(); i++)
            cout<<(*left_active_set)[i]<<" ";
        cout<<endl;
        cout<<"right_member: ";
        for(int i = 0; i < right_active_set->size(); i++)
            cout<<(*right_active_set)[i]<<" ";
        cout<<endl;

    }



	// .4 recursively build the tree
	leftNode = new fks_ompNode(2*lnid+0);
	rightNode = new fks_ompNode(2*lnid+1);
	leftNode->insert(this, inData, left_active_set, mppn, maxlevel);
	rightNode->insert(this, inData, right_active_set, mppn, maxlevel);
};
*/








