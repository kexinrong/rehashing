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

#include "srkdt.h"
#include "direct_knn.h"
#include "generator.h"
#include "verbose.h"
#include "rotation.h"
#include "stTreeSearch.h"

using namespace std;
using namespace knn;

#define _SRKDT_DEBUG_ false
#define _SRKDT_TIMER_ false

void srkdt::build(pbinData inData, int minp, int maxlev)
{
	int numof_points = inData->X.size() / inData->dim;
	double least_numof_nodes = (double)numof_points / (double)minp;
	double d = ceil(log(least_numof_nodes) / log(2.0));
	depth = (int)d;
	depth = min(maxlev, depth);
	int nleaf = (int)pow(2.0, (double)depth);
	numof_ref_points_in_tree = numof_points;

	generateRotation(inData->dim, rw);

    //leafRefIDArr.reserve(nleaf);
    leafRefIDArr.resize(nleaf);
	root = new snode();

    ptrData = inData;

    vector<long> *pids = new vector<long>();
    pids->resize(inData->numof_points);
    for(int i = 0; i < inData->numof_points; i++) {
        (*pids)[i] = (long)i;
        //(*pids)[i] = inData->gids[i];
    }
	insert(NULL, root, inData, minp, maxlev, pids);
    //if(pids != NULL) cerr<<"pids is not deleted "<<endl;
    // pids will be deleted inside insert
}

void srkdt::destroy_tree(psnode node)
{
	if(node != NULL) {
		destroy_tree(node->leftNode);
		destroy_tree(node->rightNode);
		delete node;
		node = NULL;
	}
}

srkdt::~srkdt()
{
	destroy_tree(root);
	for(int i = 0; i < leafRefIDArr.size(); i++) {
		delete leafRefIDArr[i];
		leafRefIDArr[i] = NULL;
	}
    ptrData = NULL;
}


// "inData" will be deleted inside this function
void srkdt::insert(psnode in_parent, psnode inNode, pbinData inData, int minp, int maxlev, vector<long> *&pids)
{
    double start_t = omp_get_wtime();

	// input checks
	assert( minp > 1 );
	assert( maxlev >= 0 );

	// .0 initializations
    if (in_parent!=NULL)  {
		inNode->level = in_parent->level + 1;
		inNode->parent = in_parent;
	}

	int dim = inData->dim;
	int numof_points = pids->size();

	// base case to terminate recursion
	if( inNode->level == maxlev || numof_points <= minp )
    {
        //leafRefIDArr.push_back(pids);
        leafRefIDArr[inNode->lnid] = pids;
		return;
    }

    // .1 generate projection direction
    // -> random choose one coordinate
    int coid = rand() % dim;
    inNode->proj.resize(dim);
    memset(&(inNode->proj[0]), 0, sizeof(double)*dim);
    inNode->proj[coid] = 1.0;
    newRotatePoints( &(inNode->proj[0]), 1, dim, rw);

#if _SRKDT_DEBUG_
cout<<"\ngenerate projection direction level "<<inNode->level<<" "<<omp_get_wtime()-start_t<<endl;
start_t = omp_get_wtime();
#endif

    // .2 project points
    vector<double> px;
    px.resize(numof_points);
    int ONE = 1;
    #pragma omp parallel for
    for(int i = 0; i < numof_points; i++) {
		px[i] = ddot(&dim, &(inNode->proj[0]), &ONE, &(inData->X[(*pids)[i]*dim]), &ONE);
    }

#if _SRKDT_DEBUG_
cout<<"project points "<<omp_get_wtime()-start_t<<", level = "<<inNode->level<<endl;
start_t = omp_get_wtime();
#endif

    // .3 find median
    inNode->median = select(px, px.size()/2);

#if _SRKDT_DEBUG_
cout<<"find median "<<omp_get_wtime()-start_t<<", level = "<<inNode->level<<endl;
start_t = omp_get_wtime();
#endif

    // .4 assign membership
	vector<int> leftkid_membership;
	vector<int> rightkid_membership;
	assignMembership(px, inNode->median, leftkid_membership, rightkid_membership);
    vector<long> *leftpids = new vector<long>();
    vector<long> *rightpids = new vector<long>();
    leftpids->resize(leftkid_membership.size());
    rightpids->resize(rightkid_membership.size());
    #pragma omp parallel if (leftkid_membership.size() > 5000)
    {
        #pragma omp for
        for(int i = 0; i < leftkid_membership.size(); i++)
            (*leftpids)[i] = (*pids)[leftkid_membership[i]];
    }
    #pragma omp parallel if (rightkid_membership.size() > 5000)
    {
        #pragma omp for
        for(int i = 0; i < rightkid_membership.size(); i++)
            (*rightpids)[i] = (*pids)[rightkid_membership[i]];
    }
    delete pids; pids=NULL;

#if _SRKDT_DEBUG_
cout<<"assign membership "<<omp_get_wtime()-start_t<<", level = "<<inNode->level<<", left = "<<leftpids->size()<<", right = "<<rightpids->size()
    <<" leftkid.membership.size = "<<leftkid_membership.size()<<", rightkid.membership.size = "<<rightkid_membership.size()
    <<endl;
start_t = omp_get_wtime();
#endif

	// .5 recursively build the tree
	inNode->leftNode = new snode(2*inNode->lnid+0);
	inNode->rightNode = new snode(2*inNode->lnid+1);
	insert(inNode, inNode->leftNode, inData, minp, maxlev, leftpids);
	insert(inNode, inNode->rightNode, inData, minp, maxlev, rightpids);
};


void srkdt::assignMembership(const vector<double>& px, double median,
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

void srkdt::copyData(pbinData inData, vector<long> &membership, pbinData outData)
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

double srkdt::select(vector<double> &arr, int ks)
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


int srkdt::visitGreedy(double *point, int dim, psnode node)
{
	if(NULL == node->leftNode && NULL == node->rightNode) {		// leaf node
		return node->lnid;
	}
	else {
        int ONE = 1;
		double py = ddot(&dim, &(node->proj[0]), &ONE, &(point[0]), &ONE);
		if(py < node->median)
			return visitGreedy(point, dim, node->leftNode);
		else
			return visitGreedy(point, dim, node->rightNode);
	}
}


void srkdt::queryGreedy(pbinData queryData, int k, vector< pair<double, long> > &results)
{
	double start_t = omp_get_wtime();

	int dim = queryData->dim;
	int numof_queries = queryData->X.size() / dim;

	vector<int> membership(numof_queries);
	int nleaf = leafRefIDArr.size();
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

    vector<vector<long> *> leafQueryIDArr(nleaf);
    for(int i = 0; i < nleaf; i++) {
        leafQueryIDArr[i] = new vector<long>();
        leafQueryIDArr[i]->reserve(numof_queries_per_leaf[i]);
    }
    for(int i = 0; i < numof_queries; i++) {
        leafQueryIDArr[membership[i]]->push_back((long)i);
    }

    long nk = (long)numof_queries*(long)k;

    //std::cout << "Storing " << nk << " results\n";
    results.resize(nk);

    //for(int i = 0; i < numof_queries; i++) {
    //    cout<<"query: "<<i<<" ";
    //    for(int j = 0; j < k; j++) {
    //        cout<<"("<<results[i*k+j].second<<", "<<results[i*k+j].first<<")  ";
    //    }
    //    cout<<endl;
    //}

	int maxdistsize = 0, max_ref_size = 0, max_query_size = 0;
    for(int i = 0; i < nleaf; i++) {
		int numR = leafRefIDArr[i]->size();
		int numQ = leafQueryIDArr[i]->size();
		int dasize = getBlockSize(numR, numQ)*numR;
		if(dasize > maxdistsize) maxdistsize = dasize;
		if(numR > max_ref_size) max_ref_size = numR;
		if(numQ > max_query_size) max_query_size = numQ;
	}
	double *tmpdist = new double [maxdistsize];
	double *snormr = new double [max_ref_size];
	double *snormq = new double [max_query_size];

	pair<double, long> *kmin = new pair<double, long>[max_query_size*k];

    pbinData ref = new binData();
    pbinData query = new binData();

	for(int i = 0; i < nleaf; i++) {

        copyData(ptrData, (*(leafRefIDArr[i])), ref);
        copyData(queryData, (*(leafQueryIDArr[i])), query);

		int numof_ref_points = leafRefIDArr[i]->size();
		int numof_query_points = leafQueryIDArr[i]->size();
		if(numof_query_points > 0) {
            double tmp_t = omp_get_wtime();
			directKQueryLowMem(&(ref->X[0]), &(query->X[0]), numof_ref_points, numof_query_points, k, dim, kmin, tmpdist, snormr, snormq);
            STree_Search_T_ += omp_get_wtime()-tmp_t;
            #pragma omp parallel if(numof_query_points > 500)
			{
				#pragma omp for
				for(int j = 0; j < numof_query_points; j++) {
					register long lidsjk = (long)((*leafQueryIDArr[i])[j]) * (long)k;
					register int jk = j*k;
					for(int t = 0; t < k; t++) {
						results[ lidsjk+t ].first = kmin[jk+t].first;
						results[ lidsjk+t ].second = ( (kmin[jk+t].second==-1) ? -1 : ref->gids[kmin[jk+t].second] );
					}
				} // end for j
			}
		}	// end if
	}	// end for (i < nleaf)

    delete ref;
    delete query;

	delete [] tmpdist;
	delete [] snormr;
	delete [] snormq;
	delete [] kmin;
	for(int i = 0; i < nleaf; i++) {
		delete leafQueryIDArr[i];
		leafQueryIDArr[i] = NULL;
	}


}


void srkdt::queryGreedyandMerge(pbinData queryData, int k, vector< pair<double, long> > &results)
{
	double start_t = omp_get_wtime();

	int dim = queryData->dim;
	int numof_queries = queryData->X.size() / dim;

	vector<int> membership(numof_queries);
	int nleaf = leafRefIDArr.size();
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

    vector<vector<long> *> leafQueryIDArr(nleaf);
    for(int i = 0; i < nleaf; i++) {
        leafQueryIDArr[i] = new vector<long>();
        leafQueryIDArr[i]->reserve(numof_queries_per_leaf[i]);
    }
    for(int i = 0; i < numof_queries; i++) {
        leafQueryIDArr[membership[i]]->push_back((long)i);
    }

    long nk = (long)numof_queries * (long)k;
    
    results.resize(nk);

    //cout<<"before merge: "<<endl;
    //for(int i = 0; i < numof_queries; i++) {
    //    cout<<"query: "<<i<<" ";
    //    for(int j = 0; j < k; j++) {
    //        cout<<"("<<results[i*k+j].second<<", "<<results[i*k+j].first<<")  ";
    //    }
    //    cout<<endl;
    //}


	int maxdistsize = 0, max_ref_size = 0, max_query_size = 0;
    for(int i = 0; i < nleaf; i++) {
		int numR = leafRefIDArr[i]->size();
		int numQ = leafQueryIDArr[i]->size();
		int dasize = getBlockSize(numR, numQ)*numR;
		if(dasize > maxdistsize) maxdistsize = dasize;
		if(numR > max_ref_size) max_ref_size = numR;
		if(numQ > max_query_size) max_query_size = numQ;
	}
	double *tmpdist = new double [maxdistsize];
	double *snormr = new double [max_ref_size];
	double *snormq = new double [max_query_size];

	pair<double, long> *kmin = new pair<double, long> [max_query_size*k];
    pbinData ref = new binData();
    pbinData query = new binData();

    for(int i = 0; i < nleaf; i++) {

        copyData(ptrData, (*(leafRefIDArr[i])), ref);
        copyData(queryData, (*(leafQueryIDArr[i])), query);

		int numof_ref_points = leafRefIDArr[i]->size();
		int numof_query_points = leafQueryIDArr[i]->size();
        if(numof_query_points > 0) {
            double tmp_t = omp_get_wtime();
			directKQueryLowMem(&(ref->X[0]), &(query->X[0]), numof_ref_points, numof_query_points,
                                        k, dim, kmin, tmpdist, snormr, snormq);
            STree_Search_T_ += omp_get_wtime()-tmp_t;

            #pragma omp parallel if(numof_query_points > 500)
			{
                pair<double, long> *A = new pair<double, long> [k];
                pair<double, long> *B = new pair<double, long> [k];
                pair<double, long> *C = new pair<double, long> [k];

                vector< pair<double, long> > auxVec(2*k);
				#pragma omp for
				for(int j = 0; j < numof_query_points; j++) {
		            register long lidsjk = (long)((*leafQueryIDArr[i])[j]) * (long)k;
					register int jk = j*k;
                    for(int t = 0; t < k; t++) {
                        A[t] = results[ lidsjk+t ];
                        B[t].first = kmin[jk+t].first;
                        B[t].second = ( (kmin[jk+t].second == -1) ? -1 : ref->gids[kmin[jk+t].second]);
                    }

                    //cout<<"\t"<<(*leafQueryIDArr[i])[j]<<" old: ";
                    //for(int t = 0; t < k; t++)
                    //    cout<<"("<<A[t].second<<", "<<A[t].first<<")  ";
                    //cout<<endl;

                    //cout<<"\t"<<(*leafQueryIDArr[i])[j]<<" iter: ";
                    //for(int t = 0; t < k; t++)
                    //    cout<<"("<<B[t].second<<", "<<B[t].first<<")  ";
                    //cout<<endl;

                    single_merge(A, B, k, C, auxVec);
                    //merge_one(A, B, C, k);

                    //cout<<"\t"<<(*leafQueryIDArr[i])[j]<<" merged: ";
                    //for(int t = 0; t < k; t++)
                    //    cout<<"("<<C[t].second<<", "<<C[t].first<<")  ";
                    //cout<<endl;

                    for(int t = 0; t < k; t++) {
                        results[ lidsjk+t ] = C[t];
                    }
				} // end for j
			    delete [] A;
                delete [] B;
                delete [] C;
            }
		}	// end if pragma
	}	// end for (i < nleaf)


    //cout<<endl<<"after merge: "<<endl;
    //for(int i = 0; i < numof_queries; i++) {
    //    cout<<"query: "<<i<<" ";
    //    for(int j = 0; j < k; j++) {
    //        cout<<"("<<results[i*k+j].second<<", "<<results[i*k+j].first<<")  ";
    //    }
    //    cout<<endl;
    //}


    delete ref;
    delete query;

	delete [] tmpdist;
	delete [] snormr;
	delete [] snormq;
	delete [] kmin;
    //delete [] A;
    //delete [] B;
    //delete [] C;

	for(int i = 0; i < nleaf; i++) {
		delete leafQueryIDArr[i];
		leafQueryIDArr[i] = NULL;
	}
}


void srkdt::queryGreedy_a2a(int k, vector< pair<double, long> > &results)
{
    int dim = ptrData->dim;
	int numof_queries = ptrData->X.size() / dim;
	int nleaf = leafRefIDArr.size();

    long nk = (long)numof_queries * (long)k;
    results.resize(nk);

    int maxdistsize = 0, max_ref_size = 0, max_query_size = 0;
    for(int i = 0; i < nleaf; i++) {
		int numR = leafRefIDArr[i]->size();
		int numQ = leafRefIDArr[i]->size();
		int dasize = getBlockSize(numR, numQ)*numR;
		if(dasize > maxdistsize) maxdistsize = dasize;
		if(numR > max_ref_size) max_ref_size = numR;
		if(numQ > max_query_size) max_query_size = numQ;
	}
	double *tmpdist = new double [maxdistsize];
	double *snormr = new double [max_ref_size];
	double *snormq = new double [max_query_size];

	pair<double, long> *kmin = new pair<double, long>[max_query_size*k];
    pbinData ref = new binData();

    for(int i = 0; i < nleaf; i++) {
        copyData(ptrData, (*(leafRefIDArr[i])), ref);

		int numof_ref_points = leafRefIDArr[i]->size();
		int numof_query_points = leafRefIDArr[i]->size();
        if(numof_query_points > 0) {
            double tmp_t = omp_get_wtime();
			directKQueryLowMem(&(ref->X[0]), &(ref->X[0]), numof_ref_points, numof_query_points, k, dim, kmin, tmpdist, snormr, snormq);
            STree_Search_T_ += omp_get_wtime()-tmp_t;

            #pragma omp parallel if(numof_query_points > 500)
			{
				#pragma omp for
				for(int j = 0; j < numof_query_points; j++) {
					register long lidsjk = (long)((*leafRefIDArr[i])[j]) * (long)k;
					register int jk = j*k;
					for(int t = 0; t < k; t++) {
						results[ lidsjk+t ].first = kmin[jk+t].first;
						results[ lidsjk+t ].second = ( (kmin[jk+t].second==-1) ? -1 : ref->gids[kmin[jk+t].second] );
					}
				} // end for j
			}
		}	// end if
	}	// end for (i < nleaf)

    delete ref;

	delete [] tmpdist;
	delete [] snormr;
	delete [] snormq;
	delete [] kmin;

}


void srkdt::queryGreedyandMerge_a2a(int k, vector< pair<double, long> > &results)
{
	double start_t = omp_get_wtime();

	int dim = ptrData->dim;
	int numof_queries = ptrData->X.size() / dim;
	int nleaf = leafRefIDArr.size();

    long nk = (long)numof_queries * (long)k;
    results.resize(nk);

	int maxdistsize = 0, max_ref_size = 0, max_query_size = 0;
    for(int i = 0; i < nleaf; i++) {
		int numR = leafRefIDArr[i]->size();
		int numQ = leafRefIDArr[i]->size();
		int dasize = getBlockSize(numR, numQ)*numR;
		if(dasize > maxdistsize) maxdistsize = dasize;
		if(numR > max_ref_size) max_ref_size = numR;
		if(numQ > max_query_size) max_query_size = numQ;
	}
	double *tmpdist = new double [maxdistsize];
	double *snormr = new double [max_ref_size];
	double *snormq = new double [max_query_size];

	pair<double, long> *kmin = new pair<double, long> [max_query_size*k];
    pbinData ref = new binData();

    for(int i = 0; i < nleaf; i++) {

        copyData(ptrData, (*(leafRefIDArr[i])), ref);

		int numof_ref_points = leafRefIDArr[i]->size();
		int numof_query_points = leafRefIDArr[i]->size();
        if(numof_query_points > 0) {
            double tmp_t = omp_get_wtime();
			directKQueryLowMem(&(ref->X[0]), &(ref->X[0]),
                               numof_ref_points, numof_query_points,
                               k, dim, kmin, tmpdist, snormr, snormq);
            STree_Search_T_ += omp_get_wtime()-tmp_t;

            #pragma omp parallel if(numof_query_points > 500)
			{
                pair<double, long> *A = new pair<double, long> [k];
                pair<double, long> *B = new pair<double, long> [k];
                pair<double, long> *C = new pair<double, long> [k];

                vector< pair<double, long> > auxVec(2*k);
				#pragma omp for
				for(int j = 0; j < numof_query_points; j++) {
		            register long lidsjk = (long)((*leafRefIDArr[i])[j]) * (long)k;
					register int jk = j*k;
                    for(int t = 0; t < k; t++) {
                        A[t] = results[ lidsjk+t ];
                        B[t].first = kmin[jk+t].first;
                        B[t].second = ( (kmin[jk+t].second == -1) ? -1 : ref->gids[kmin[jk+t].second]);
                    }

                    //cout<<"\t"<<(*leafQueryIDArr[i])[j]<<" old: ";
                    //for(int t = 0; t < k; t++)
                    //    cout<<"("<<A[t].second<<", "<<A[t].first<<")  ";
                    //cout<<endl;

                    //cout<<"\t"<<(*leafQueryIDArr[i])[j]<<" iter: ";
                    //for(int t = 0; t < k; t++)
                    //    cout<<"("<<B[t].second<<", "<<B[t].first<<")  ";
                    //cout<<endl;

                    single_merge(A, B, k, C, auxVec);
                    //merge_one(A, B, C, k);

                    //cout<<"\t"<<(*leafQueryIDArr[i])[j]<<" merged: ";
                    //for(int t = 0; t < k; t++)
                    //    cout<<"("<<C[t].second<<", "<<C[t].first<<")  ";
                    //cout<<endl;

                    for(int t = 0; t < k; t++) {
                        results[ lidsjk+t ] = C[t];
                    }
				} // end for j
			    delete [] A;
                delete [] B;
                delete [] C;
            }
		}	// end if pragma
	}	// end for (i < nleaf)


    //cout<<endl<<"after merge: "<<endl;
    //for(int i = 0; i < numof_queries; i++) {
    //    cout<<"query: "<<i<<" ";
    //    for(int j = 0; j < k; j++) {
    //        cout<<"("<<results[i*k+j].second<<", "<<results[i*k+j].first<<")  ";
    //    }
    //    cout<<endl;
    //}

    delete ref;
	delete [] tmpdist;
	delete [] snormr;
	delete [] snormq;
	delete [] kmin;
    //delete [] A;
    //delete [] B;
    //delete [] C;
}


void srkdt::merge_one(pair<double, long> *A, pair<double, long> *B, pair<double, long> *tmp, int k)
{
    int aloc = 0;
    int bloc = 0;
    int tmploc = 0;
    for(int i = 0; i < k; i++) {
	    if( (A[aloc].second == B[bloc].second) && (bloc == (k-1)) )
            B[bloc] = make_pair(DBL_MAX, -1);
		if( (A[aloc].second == B[bloc].second) && (bloc < (k-1)) )
            bloc++;
        if( A[aloc].first <= B[bloc].first ) {
			tmp[tmploc++] = A[aloc++];
        }
		else {
			tmp[tmploc++] = B[bloc++];
        }
	}
}





