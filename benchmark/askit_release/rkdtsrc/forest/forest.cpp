#include <vector>
#include <map>
#include <cfloat>
#include <queue>
#include <set>
#include <algorithm>
#include "forest.h"
#include "rotation.h"
#include "stTree.h"
#include "stTreeSearch.h"
#include "distributeToLeaf.h"
#include "srkdt.h"

#define _FOREST_DEBUG_ 0

int _K_FOR_REDUCE_;

void forest::plant(binData *inData, int numof_trees)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    shared_tree_iters = numof_trees;

    if(datapool == NULL) {
        datapool = new binData();
        datapool->numof_points = 0;
        datapool->dim = inData->dim;
    }

    long local_n = inData->numof_points;
    long glb_n = 0;
    MPI_Allreduce(&local_n, &glb_n, 1, MPI_LONG, MPI_SUM, comm);
    glb_numof_points = glb_n;
    gid2lid.resize(glb_numof_points);
    #pragma omp parallel for
    for(long i = 0; i < glb_numof_points; i++)
        gid2lid[i] = -1;

    maxp = maxp < (glb_numof_points/size+1) ? maxp : (glb_numof_points/size+1);

    if(size == 1) numof_trees = 1;
    trees.resize(numof_trees);
    int p = 0;
    for(int i = 0; i < numof_trees; i++) {
        binNode *root = new binNode();
        root->options.flag_r = 1;
        //root->options.flag_c = 1;
        root->InsertInMemory(NULL, maxp, maxLevel, minCommSize, comm, inData, datapool, gid2lid);
        trees[i] = root;
    } // end for
}


forest::~forest()
{
    for(int i = 0; i < trees.size(); i++) {
        delete trees[i];
    }
    delete datapool;
    datapool = NULL;
}


// this function assume every process have all query data
void forest::find_knn_in_tree(binData *allQueryData, int k, binNode *root,
                              vector< pair<double, long> > *&kneighbors,
                              vector<long> *queryIDs)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if(size == 1) {
        find_knn_srkdt(datapool, allQueryData, k, maxp, maxLevel, shared_tree_iters, kneighbors);

        queryIDs->resize(allQueryData->numof_points);
        for(int i = 0; i < allQueryData->numof_points; i ++)
            (*queryIDs)[i] = allQueryData->gids[i];
    }
    else {
        // 1. get local query
        int dim = allQueryData->dim;
        int glb_nq = allQueryData->numof_points;
        int divd = glb_nq / size;
        int rem = glb_nq % size;
        int local_nq = rank < rem ? divd+1 : divd;
        int offset = rank < rem ? rank*(divd+1) : rem*(divd+1)+(rank-rem)*divd;
        binData *localQuery = new binData;
        localQuery->dim = allQueryData->dim;
        localQuery->numof_points = local_nq;
        localQuery->gids.resize(local_nq);
        localQuery->X.resize(local_nq*dim);
        #pragma omp parallel for
        for(int i = 0; i < local_nq; i++) {
            localQuery->gids[i] = allQueryData->gids[offset+i];
            memcpy( &(localQuery->X[i*dim]), &(allQueryData->X[(offset+i)*dim]), dim*sizeof(double));
        }

#if _FOREST_DEBUG_
cout<<"rank "<<rank<<" get local query done"<<endl;
#endif

        // 2. pass down tree
        binData *redistQuery;
        binNode *leaf;
        bintree::GoToNearestLeafbyMedian(localQuery, root, &redistQuery, &leaf);
        int numof_query_points = redistQuery->numof_points;

#if _FOREST_DEBUG_
cout<<"rank "<<rank<<" query pass down tree done"<<endl;
#endif

        // 3. copy ref data
        binData *localRef = new binData();
        int local_nr = leaf->data->gids.size();
	    localRef->dim = dim;
        localRef->numof_points = leaf->data->gids.size();
        localRef->gids.resize(local_nr);
        localRef->X.resize(local_nr*dim);
        #pragma omp parallel for
        for(int i = 0; i < local_nr; i++) {
            localRef->gids[i] = leaf->data->gids[i];
            int lid = getLocalID(leaf->data->gids[i]);
            if(lid != -1) {
                memcpy( &(localRef->X[i*dim]), &(datapool->X[lid*dim]), dim*sizeof(double) );
            }
        }

#if _FOREST_DEBUG_
cout<<"rank "<<rank<<" get local ref done"<<endl;
#endif

        // 4. find knn
        if(numof_query_points > 0) {

#if _FOREST_DEBUG_
cout<<"rank "<<rank<<" ref->n = "<<localRef->numof_points<<" ref->dim = "<<localRef->dim
    <<" ref->X.size = "<<localRef->X.size()<<" ref->gids.size = "<<localRef->gids.size()
    <<" query->n = "<<redistQuery->numof_points<<" query->dim = "<<redistQuery->dim
    <<" query->X.size = "<<redistQuery->X.size()<<" query->gids.size = "<<redistQuery->gids.size()
    <<endl;
#endif

            find_knn_srkdt(localRef, redistQuery, k, maxp, maxLevel-leaf->level, 1, kneighbors);
            delete localRef;
            localRef = NULL;
            //stTreeSearch_rkdt_me(localRef, redistQuery, k, 0, maxp, maxLevel-leaf->level, kneighbors);
            //localRef = NULL;
        }
        else {
            delete localRef;
            localRef = NULL;
        }

        // 5. copy query id
        queryIDs->resize(numof_query_points);
        for(int i = 0; i < numof_query_points; i ++)
            (*queryIDs)[i] = redistQuery->gids[i];

        delete localQuery;
        delete redistQuery;
    }

}


void forest::find_knn(binData *allQueryData, int k,
                      vector< pair<double, long> * > &knn,
                      vector<int> &ksize)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double start_t;
    int glb_nq = allQueryData->numof_points;
    ksize.resize(glb_nq);
    if(knn.empty()) {
        knn.resize(glb_nq);
        for(int i = 0; i < glb_nq; i++)
            knn[i] = new pair<double, long> [k];
    }

    pair<double, long> *all_local_knn = new pair<double, long> [glb_nq*k];
    #pragma omp parallel for
    for(int i = 0; i < glb_nq*k; i++)
        all_local_knn[i] = make_pair<double, long>(DBL_MAX, -1);

    start_t = omp_get_wtime();
    for(int t = 0; t < trees.size(); t++) {
        vector<pair<double, long> > *kneighbors = new vector<pair<double, long> >();
        vector<long> *queryIDs = new vector<long>();
        find_knn_in_tree(allQueryData, k, trees[t], kneighbors, queryIDs);

        // merge local results
        #pragma omp parallel
        {
            pair<double, long> *tmp = new pair<double, long> [k];
            #pragma omp for
            for(int i = 0; i < queryIDs->size(); i++) {
                int qid = (*queryIDs)[i];
                int aloc = 0;
                int bloc = 0;
                for(int j = 0; j < k; j++) {
                    if( all_local_knn[qid*k+aloc].second == (*kneighbors)[i*k+bloc].second && bloc == k-1 )
                        (*kneighbors)[i*k+bloc] = make_pair<double, long>(DBL_MAX, -1);
                    if( all_local_knn[qid*k+aloc].second == (*kneighbors)[i*k+bloc].second && bloc < k-1 ) bloc++;
                    if( all_local_knn[qid*k+aloc].first <= (*kneighbors)[i*k+bloc].first ) {
                        tmp[j] = all_local_knn[qid*k+(aloc++)];
                    }
                    else {
                        tmp[j] = (*kneighbors)[i*k+(bloc++)];
                    }
                }   // end for j
                for(int j = 0; j < k; j++)
                    all_local_knn[qid*k+j] = tmp[j];
            }
            delete [] tmp;
        }
        delete kneighbors;
        delete queryIDs;
    } // end for t

    #if _FOREST_DEBUG_
    /*
    for(int r = 0; r < size; r++) {
        if(rank == r) {
            if(rank == 0) cout<<"local tree search time "<<omp_get_wtime()-start_t<<endl;
            for(int i = 0; i < glb_nq; i++) {
                cout<<"(local merge) ID "<<allQueryData->gids[i]<<":  ";
                for(int j = 0; j < k; j++) {
                    cout<<"("<<all_local_knn[i*k+j].second
                        <<","<<all_local_knn[i*k+j].first<<")  ";
                }
                cout<<endl;
            }
            cout<<endl;
        }
        cout.flush();
        MPI_Barrier(comm);
    }*/
    //if(rank == 0) cout<<"local tree search time "<<omp_get_wtime()-start_t<<endl;
    #endif

    // global merge
    start_t = omp_get_wtime();
    pair<double, long> *all_glb_knn = new pair<double, long> [glb_nq*k];
    globalMerge(all_local_knn, all_glb_knn, glb_nq, k);
    delete [] all_local_knn;

    #if _FOREST_DEBUG_
    /*for(int r = 0; r < size; r++) {
        if(rank == r) {
            if(rank == 0) cout<<"global merge time "<<omp_get_wtime()-start_t<<endl;
            for(int i = 0; i < glb_nq; i++) {
                cout<<"(global merge) ID "<<allQueryData->gids[i]<<":  ";
                for(int j = 0; j < k; j++) {
                    cout<<"("<<all_glb_knn[i*k+j].second
                        <<","<<all_glb_knn[i*k+j].first<<")  ";
                }
                cout<<endl;
            }
            cout<<endl;
        }
        cout.flush();
        MPI_Barrier(comm);
    }*/
    //if(rank == 0) cout<<"global merge time "<<omp_get_wtime()-start_t<<endl;
    #endif

    // packing results into knn and ksize
    // --------------------------------------------
    // It is better to use hash table,
    // but unordered_set is in boost::tr1 or C++11
    // I do not have hash_table header on hand,
    // for this time being, I just use vector<int>,
    // each bit in the int represent true or false
    // as long as glb_numof_points is not too large, it is fine
    // --------------------------------------------
    binNode *leaf = trees[0];
    while(NULL != leaf->kid) leaf = leaf->kid;
    bitvec table;
    table.create(glb_numof_points);
    for(int i = 0; i < leaf->data->gids.size(); i++)
        table.set(leaf->data->gids[i]);
    for(int i = 0; i < glb_nq; i++) {
        int local_k = 0;
        for(int j = 0; j < k; j++) {
            long refid = all_glb_knn[i*k+j].second;
            if(table.get(refid)) local_k++;
        }
        ksize[i] = local_k;
        int idx = 0;
        for(int j = 0; j < k; j++) {
            long refid = all_glb_knn[i*k+j].second;
            if(table.get(refid)) knn[i][idx++] = all_glb_knn[i*k+j];
        }
    }
    delete [] all_glb_knn;

    #if _FOREST_DEBUG_
    /*for(int r = 0; r < size; r++) {
        if(rank == r) {
            for(int i = 0; i < ksize.size(); i++) {
                cout<<"(result) ID "<<allQueryData->gids[i]<<": local k = "<<ksize[i]<<" ";
                for(int j = 0; j < ksize[i]; j++) {
                    cout<<"("<<knn[i][j].second
                        <<","<<knn[i][j].first<<")  ";
                }
                cout<<endl;
            }
            cout<<endl;
        }
        cout.flush();
        MPI_Barrier(comm);
    }*/
    if(rank == 0) cout<<"final packing time "<<omp_get_wtime()-start_t<<endl;
    #endif

}


void forest::find_all_knn(binData *allQueryData, int k,
                          vector< pair<double, long> > *all_glb_knn)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int glb_nq = allQueryData->numof_points;

    pair<double, long> *all_local_knn = new pair<double, long> [glb_nq*k];
    #pragma omp parallel for
    for(int i = 0; i < glb_nq*k; i++)
        all_local_knn[i] = make_pair<double, long>(DBL_MAX, -1);

//cout<<"\tenter find_all_knn "<<endl;

    pair<double, long> *tmp = new pair<double, long> [k];
    for(int t = 0; t < trees.size(); t++) {
        vector<pair<double, long> > *kneighbors = new vector<pair<double, long> >();
        vector<long> *queryIDs = new vector<long>();
        find_knn_in_tree(allQueryData, k, trees[t], kneighbors, queryIDs);

        //cout<<"\t iter "<<t<<" find_knn_in_tree done"<<endl;

        // merge local results
        //#pragma omp parallel for
        for(int i = 0; i < queryIDs->size(); i++) {
            int qid = (*queryIDs)[i];
            int aloc = 0;
            int bloc = 0;
            for(int j = 0; j < k; j++) {
                if( all_local_knn[qid*k+aloc].second == (*kneighbors)[i*k+bloc].second && bloc == k-1 ) 
                    (*kneighbors)[i*k+bloc] = make_pair<double, long>(DBL_MAX, -1);
                if( all_local_knn[qid*k+aloc].second == (*kneighbors)[i*k+bloc].second && bloc < k-1 ) bloc++;
                if( all_local_knn[qid*k+aloc].first <= (*kneighbors)[i*k+bloc].first ) {
                    tmp[j] = all_local_knn[qid*k+(aloc++)];
                }
                else {
                    tmp[j] = (*kneighbors)[i*k+(bloc++)];
                }
            }   // end for j
            for(int j = 0; j < k; j++) all_local_knn[qid*k+j] = tmp[j];
        }
        delete kneighbors;
        delete queryIDs;

    } // end for t
    delete [] tmp;

//cout<<"\tlocal search tree done "<<endl;

    // global merge
    all_glb_knn->resize(glb_nq*k);
    globalMerge(all_local_knn, &((*all_glb_knn)[0]), glb_nq, k);
    delete [] all_local_knn;

//cout<<"\tglobal merge done"<<endl;

    // output local merge result
    #if _FOREST_DEBUG_
    /*
    for(int r = 0; r < size; r++) {
        if(rank == r) {
            for(int i = 0; i < glb_nq; i++) {
                cout<<"rank "<<rank<<" (global merge) ID "<<allQueryData->gids[i]<<":  ";
                for(int j = 0; j < k; j++) {
                    cout<<"("<<(*all_glb_knn)[i*k+j].second
                        <<","<<(*all_glb_knn)[i*k+j].first<<")  ";
                }
                cout<<endl;
            }
            cout<<endl;
        }
        cout.flush();
        MPI_Barrier(comm);
    }*/
    #endif

}


void forest::globalMerge(pair<double, long> *all_local_knn, pair<double, long> *all_glb_knn, int n, int k)
{
    _K_FOR_REDUCE_ = k;

    MPI_Op myOp;
    MPI_Op_create(myReduceOp, true, &myOp);
    MPI_Datatype knntype;
    MPI_Type_contiguous(sizeof(pair<double, long>)*k, MPI_BYTE, &knntype);
    MPI_Type_commit(&knntype);
    MPI_Allreduce( all_local_knn, all_glb_knn, n, knntype, myOp, comm);
}


void forest::seperate_knn(vector< pair<double, long> > *all_glb_knn, int k,
                          vector< pair<double, long> * > &knn,
                          vector<int> &ksize)
{
    int glb_nq = all_glb_knn->size() / k;
    ksize.resize(glb_nq);
    if(knn.empty()) {
        knn.resize(glb_nq);
        for(int i = 0; i < glb_nq; i++)
            knn[i] = new pair<double, long> [k];
    }

    // --------------------------------------------
    // It is better to use hash table,
    // but unordered_set is in boost::tr1 or C++11
    // I do not have hash_table header on hand,
    // for this time being, I just use vector<int>,
    // each bit in the int represent true or false
    // as long as glb_numof_points is not too large, it is fine
    // --------------------------------------------
    binNode *leaf = trees[0];
    while(NULL != leaf->kid) leaf = leaf->kid;
    bitvec table;
    table.create(glb_numof_points);
    for(int i = 0; i < leaf->data->gids.size(); i++)
        table.set(leaf->data->gids[i]);
    for(int i = 0; i < glb_nq; i++) {
        int local_k = 0;
        for(int j = 0; j < k; j++) {
            long refid = (*all_glb_knn)[i*k+j].second;
            if(table.get(refid)) local_k++;
        }
        ksize[i] = local_k;
        int idx = 0;
        for(int j = 0; j < k; j++) {
            long refid = (*all_glb_knn)[i*k+j].second;
            if(table.get(refid)) knn[i][idx++] = (*all_glb_knn)[i*k+j];
        }
    }
}


void myReduceOp(void *in_, void *inout_, int *n, MPI_Datatype *dptr)
{
    pair<double, long> *in = (pair<double, long> *)in_;
    pair<double, long> *inout = (pair<double, long> *)inout_;

    int k = _K_FOR_REDUCE_;
    int numof_points = (*n);

    //cout<<" (myReduceOp) numof_points = "<<numof_points<<", k = "<<k
    //    <<" n = "<<(*n)<<endl;

    pair<double, long> *merge = new pair<double, long> [k];

    //#pragma omp parallel for
    for(int i = 0; i < numof_points; i++) {
        int idx1 = 0;
        int idx2 = 0;
        for(int j = 0; j < k; j++) {
           if(in[i*k+idx1].second == inout[i*k+idx2].second && idx2 == k-1) 
               inout[i*k+idx2] = make_pair<double, long>(DBL_MAX, -1);
           if(in[i*k+idx1].second == inout[i*k+idx2].second && idx2 < k-1) idx2++;
           if(in[i*k+idx1].first <= inout[i*k+idx2].first) {
               merge[j] = in[i*k+idx1++];
           }
           else {
               merge[j] = inout[i*k+idx2++];
           }
        }
        for(int j = 0; j < k; j++)
            inout[i*k+j] = merge[j];
    }

    delete [] merge;
}


// ---------------- bitvec utility class ----------------

void bitvec::create(long numof_bits)
{
    int numof_int = numof_bits/sizeof(int) + 1;
    array.resize(numof_int);
}

void bitvec::clear()
{
    for(int i = 0; i < array.size(); i++) array[i] = 0;
}

void bitvec::reset(long idx)
{
    int id = idx / sizeof(int);
    int pos = idx % sizeof(int);
    int number = array[id];
    number &= ~(1 << pos);
    array[id] = number;
}

void bitvec::set(long idx)
{
    int id = idx / sizeof(int);
    int pos = idx % sizeof(int);
    int number = array[id];
    number |= 1 << pos;
    array[id] = number;
}

void bitvec::flip(long idx)
{
    int id = idx / sizeof(int);
    int pos = idx % sizeof(int);
    int number = array[id];
    number ^= 1 << pos;
    array[id] = number;
}

bool bitvec::get(long idx)
{
    int id = idx / sizeof(int);
    int pos = idx % sizeof(int);
    int number = array[id];
    bool bit = number & (1 << pos);
    return bit; 
}


