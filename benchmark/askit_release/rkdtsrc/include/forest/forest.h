#ifndef _FOREST_H_
#define _FOREST_H_

#include <mpi.h>
#include <fstream>
#include <sstream>
#include "binTree.h"


class forest
{
    public:
        vector<binNode *> trees;
        binData *datapool;
        // ideally, it should be hashmap.. I do not find a good hashmap to use
        // may change it later, map is too slow. I would assume the glb_numof_points is not too large
        vector<int> gid2lid;

        int maxp;
        int maxLevel;
        int minCommSize;
        long glb_numof_points;  // set in plant()
        bool flag_rotate_back;
        MPI_Comm comm;
        string path;
        int shared_tree_iters;

        forest() : maxp(1000), maxLevel(20), minCommSize(1), flag_rotate_back(false), comm(MPI_COMM_WORLD), path("./"), datapool(NULL), glb_numof_points(0), shared_tree_iters(1) {}
        ~forest();

        void setLeafSize(int _maxp) { maxp = _maxp; }
        int getLeafSize() const { return maxp; }

        void setHeight(int _height) { maxLevel = _height; }
        int getHeight() const { return maxLevel; }

        void setComm(MPI_Comm comm_) { comm = comm_; }
        MPI_Comm getComm() const { return comm; }

        void setCommSize(int _size) { minCommSize = _size; }
        int getCommSize() const { return minCommSize; }

        void setRotateDataBack(bool _rotback) { flag_rotate_back = _rotback; }
        bool isRotBack() const { return flag_rotate_back; }

        void setSharedTreeIters(int _iter) { shared_tree_iters = _iter; }
        int getSharedTreeIters() const { return shared_tree_iters; }

        long getGlbNumPoints() const { return glb_numof_points; }

        int getLocalID(long glb_id) { return gid2lid[glb_id]; }

        void plant(binData *inData, int numof_trees);

        void find_knn_in_tree(binData *allQueryData, int k, binNode *root,
                              vector< pair<double, long> > *&kneighbors,
                              vector<long> *queryIDs);

        void find_knn(binData *allQueryData, int k,
                      vector< pair<double, long> * > &knn,
                      vector<int> &ksize);

        void find_all_knn(binData *allQueryData, int k,
                          vector< pair<double, long> > *all_glb_knn);

        void seperate_knn(vector< pair<double, long> > *all_glb_knn, int k,
                          vector< pair<double, long> * > &knn,
                          vector<int> &ksize);

        void globalMerge(pair<double, long> *all_local_knn,
                         pair<double, long> *all_glb_knn, int n, int k);
};


class bitvec
{
private:
    vector<int> array;
public:
    void create(long numof_bits);
    void clear();
    void reset(long idx);
    void set(long idx);
    void flip(long idx);
    bool get(long idx);
};


// --------- some utility function --------

void myReduceOp(void *in_, void *inout_, int *n, MPI_Datatype *dptr);

inline string bxitoa(int value)
{
    ostringstream o;
    if(!(o << value)) 
        return "";
    return o.str();
}

class pairless
{
public:
    bool operator() (const pair<double, long> &a, const pair<double, long> &b) { return a.first < b.first; }
};

#endif
