#ifndef __FKSTREE_H__
#define __FKSTREE_H__

#include <mpi.h>

#include <cstdlib>
#include <vector>
#include <set>

#include "fks_ompNode.h"
#include "binTree.h"
#include "askit_utils.hpp"

namespace askit {


template<typename A, typename B, typename C>
class triple {
public:
  A first;
  B second;
  C third;

  triple() : first(0), second(0), third(0) {}

  triple(A a, B b, C c) : first(a), second(b), third(c) {}

  bool operator<(const triple &a) const {
    if(first < a.first){
      return true;
    } else if (first == a.first) {
      if( second < a.second ) {
        return true;
      } else if( second == a.second ) {
        if(third < a.third)
          return true;
        else
          return false;
      } else {
        return false;
      }
    }
    else return false;
  }

  void operator=(const triple<A,B,C>& a) {
    first = a.first;
    second = a.second;
    third = a.third;
  }

  static bool firstLess( triple<A,B,C> a, triple<A,B,C> b ){
    return a.first < b.first;
  }

  static bool secondLess( triple<A,B,C> a, triple<A,B,C> b ){
    return a.second < b.second;
  }

  static bool thirdLess( triple<A,B,C> a, triple<A,B,C> b ){
    return a.third < b.third;
  }

  static bool firstEqual( triple<A,B,C> a, triple<A,B,C> b ){
    return a.first == b.first;
  }

  static bool secondEqual( triple<A,B,C> a, triple<A,B,C> b ){
    return a.second == b.second;
  }

  static bool thirdEqual( triple<A,B,C> a, triple<A,B,C> b ){
    return a.third == b.third;
  }
};


template<typename T>
int binary_search(vector<T> &arr, T target)
{
    int left = 0, right = arr.size()-1;
    while(left <= right) {
        int mid = left + (right-left)/2;
        if(arr[mid] == target) {
            return mid;
        }
        else if(arr[mid] < target) {
            left = mid+1;
        }
        else {
            right = mid-1;
        }
    }
    return -1;
}


typedef struct t_fksCtx {
  t_fksCtx() : need_knn(true) {} // set need_knn to default to true for backwards compatibility
    int k;                  // 'k' nearest neighbors
    int fks_mppn;           // fks_tree: (to do the free kernel summation) maximum points per leaf node
    int fks_maxLevel;       // fks_tree: (to do the free kernel summation) maximum level of the tree
    int minCommSize;        // minimum communicator size for both tree, usually set to be 1
    int min_skeleton_level; // minimum skeletionization level
    bool check_knn_accuracy; // whether check knn or not
    bool need_knn; // set to true if we need knn, false otherwise (for gm algorithm)
} fksCtx;


class fks_mpiNode : public binNode
{
public:
    // - it also has the following member
    // vector<double> proj;
    // double median;
    // binNode *kid;    // not used in fks_mpiNode
    // binNode *parent;     // not used in fks_mpiNode
    // int level;
    // binData *data;   // points in leaf node
    // MPI_Comm comm
    //

    fks_mpiNode *fks_kid;
    fks_mpiNode *fks_parent;
    fksData *skeletons;
    long node_morton;

    // the number of points owned by all the descendants of this node
    long num_points_owned;

    //vector<auxTriple> excl_knn_of_this_node;
    // triple<gid, morton, dist>
    vector< triple<long, long, double> > excl_knn_of_this_node;
    // For the split k version of the algorithm, we maintain a list of 
    // the \nkprune nearest neighbors of each skeleton point to use 
    // in truncating the interaction lists
    vector<long> pruning_neighbor_list;


    fks_mpiNode() : fks_kid(NULL), fks_parent(NULL), skeletons(NULL), node_morton(0) {;}
    fks_mpiNode(int ci): binNode(ci), fks_kid(NULL), fks_parent(NULL), skeletons(NULL), node_morton(0) {;}
    ~fks_mpiNode();

    void Insert(fks_mpiNode *in_parent, int maxp, int maxLevel, int minCommSize, MPI_Comm inComm, binData *inData);

};


class fksTree {
public:
    MPI_Comm comm;

    fks_mpiNode *root_mpi;
    fks_ompNode *root_omp;
    fks_ompNode *root_let;
    vector<fks_ompNode *> letNodeList;
    map<long, int> letNodeMap;

    fksData *inProcData;        // it would store all the data this processor needs
    map<long, int> inProcMap;   // global id to local id in inProcData
    vector< pair<double, long> > *inProcKNN;   // knn for each point of the distribed leaf
    pair<double, double> knn_acc;   // knn accuracy <hit rate, relative error>

    int numof_points_of_dist_leaf;          // numof points of distribured leaf
    int numof_neighbors_not_in_dist_leaf;   // numof neighbors not in the distributed leaf
    int numof_points_of_share_leaf;         // numof points of shared leaf
    int glbN;
    int depth_mpi;
    int depth_omp;
    int depth_let;

    // update_charges need to exchage_let(skeletons) again, store
    // set_leaves and set_skeletons for later use
    // triple<node_gid, node_morton, node_level>
    set< triple<long, long, int> > my_set_leaves;
    set< triple<long, long, int> > my_set_skeletons;


    // +++++++++ definitions for separate test points ++++++++

    // When we add testing points, these are the new leaves and skeletons
    // ASKIT iterates through these to update the interaction lists
    set <triple<long, long, int> > testing_set_leaves;
    set <triple<long, long, int> > testing_set_skeletons;


    vector<int> internal_origin_ranks;    // ranks where the original points locate now in the tree
    vector<long> internal_origin_mortons; // morton ids of the original points
    fksData *inProcTestData;
    map<long, int> inProcTestMap;   // global to local id in inProcTestData
    vector< pair<long, long> > *inProcTestKNN;  // pair<global_id, morton_id>

    // timers for benchmarking and profiling
    
    // time to exchange the skeleton frontier (for adaptive level restriction)
    double frontier_exchange_time;
    
    // time to call initial prune by KNN in LET() -- now with skeleton frontier
    double prune_by_knn_time;


    // ========== member functions ==========

    fksTree() : root_mpi(NULL), root_omp(NULL), root_let(NULL),
                inProcData(NULL), inProcKNN(NULL),
                inProcTestData(NULL), inProcTestKNN(NULL),
                comm(MPI_COMM_WORLD) { }
    ~fksTree();

    ////////// New testing functions //////////////////////
    void DistributeTestingPoints(fksData *inData, vector< pair<double, long> > *inKNN);

    // Takes interaction lists as arguments and fills them in with global 
    // ids of source nodes -- this avoids traversing the tree twice
    void UpdateTestLET(vector<vector<long> >& direct_ints,
      vector<vector<long> >& approx_ints, long k);    // IMPORTANT: append new ones to letNodeList only


    double* address_coords() const { return inProcData->X.data(); }
    double* address_charges() const { return inProcData->charges.data(); }
    long* address_gids() const { return inProcData->gids.data(); }
    long* address_mortons() const { return inProcData->mortons.data(); }
    pair<double, long>* address_knn() const { return inProcKNN->data(); }

    int pos(long gid);
    // does the same thing with test points
    int test_pos(long test_gid);

    int morton_to_rank(long node_morton, int node_level);
    long morton_to_lid(long node_morton, int node_level);
    long morton_to_gid(long node_morton, int node_level);

    // build tree and read knn
    void build(fksData *inData, void *ctx);
    void knn(fksData *inData, vector< pair<double, long> > *inKNN);

    // leaf data
    bool getLeafData(fks_ompNode *ompLeaf, fksData *leaf);
    // sampling num_neighbors is an argument used for the split neighbors case
    bool getLeafexclNN(fks_ompNode *ompLeaf, fksData *exclNN, int sampling_num_neighbors);

    // build let
    fks_ompNode* let_node(long node_gid);
    // direct_interactions: list of target to source node global id interactions
    // put this here to avoid a second tree traversal for interaction lists
    // in ASKIT
    void LET(int min_skeleton_global_level,
                  set< triple<long, long, int> > &set_leaves,
                  set< triple<long, long, int> > &set_skeletons,
                  vector<pair<long, int> >& my_skeleton_frontier, int k, 
                  vector<vector<long> >& direct_interactions,
                  vector<vector<long> >& approx_interactions);
                  
                  
    void exchange_let(set< triple<long, long, int> > &set_leaves,
                      set< triple<long, long, int> > &set_skeletons);

    // This version is just for debugging -- bill
    void exchange_let(set< triple<long, long, int> > &set_leaves,
                      set< triple<long, long, int> > &set_skeletons,
                      vector<int>& skeleton_sizes, bool check_sizes);

    void exchange_updated_let(set< triple<long, long, int> > &set_skeletons,
                          vector<int>& skeleton_sizes, bool check_sizes);

    void printLET_preorder(fks_ompNode *inNode);

    // distribured part of skeletonization
    void mergeNNList(fks_mpiNode *inNode, int max_size);
    bool getDistNodeExclNN(fks_mpiNode *inNode, fksData *exclNN);
    void uniformSampleSibling(fks_mpiNode *inNode, int Ns, fksData *sampleData);
    void mergeSkeletonsOfKids(fks_mpiNode *inNode, fksData *ske);

    // made this public to replace the dupicated version in askit_utils.h
    bool belong2Node(long node_morton, int node_level, long point_morton);

    // made this public to use it in UpdateCharges in AskitAlg
    void exchange_charges(int numof_points, double *charges,
                          int numof_request_points, long *request_gids,
                          double *request_charges,
                          MPI_Comm comm);

    void shuffle_back(int numof_points, double *values, long *gids,
                      int shuffled_numof_points, double *shuffled_values, MPI_Comm comm);

    void shuffle_back(int numof_points, int dim, long *gids, double *values,
                           int shuffled_numof_points, double *shuffled_values, MPI_Comm comm);



private:

    int min_training_skeleton_level;

    // the slice of the tree where we can first skeletonize -- used for pruning
    // in the adaptive level restriction case
    vector<pair<long, int> > skeleton_frontier;

    void exchange_basic_knn(fksData *inData, vector< pair<double, long> > *inKNN);
    void exchange_knn_data(fksData *inData);

    void modify_root_omp_id(fks_ompNode *inNode);
    void graft_omp_to_let();
    fks_ompNode *insert_node_by_morton(fks_ompNode *inNode, long morton, int level);
    void level_order_traversal(fks_ompNode *root);     // level order traversal
    void update_let_node_list(fks_ompNode *root);

    // direct_interactions -- list of global ids of source nodes which interact
    // directly with this target
    // approx_interactions -- same for skeletons
    void pruneByKNN(long node_morton, int node_level,
            long *knn_morton_ids, int k, int min_skeleton_global_level,
            set< triple<long, long, int> > &set_leaves,
            set< triple<long, long, int> > &set_skeletons,
            vector<long>& direct_interactions,
            vector<long>& approx_interactions,
            vector<pair<long, int> >& skeleton_frontier);
            
   bool prunable(long node_morton, int node_level,
          long *knn_morton_ids, int k, int min_skeleton_global_level);
    // made this public
    //bool belong2Node(long node_morton, int node_level, long point_morton);
    int check_bit(long number, int bit);

    // struct comp_second_first {
//         bool operator ()(pair<double, long> const& a, pair<double, long> const& b) const {
//           if (a.second != b.second)
//           {
//             if (a.second < b.second)
//               return true;
//             else
//               return false;
//           }
//           else {
//             if (a.first < b.first)
//               return true;
//             else
//               return false;
//           }
//         }
//       };

    struct comp_second {
        bool operator ()(pair<double, long> const& a, pair<double, long> const& b) const {
          return a.second < b.second;
        }
    };

    struct equ_second {
        bool operator ()(pair<double, long> const& a, pair<double, long> const& b) const {
          return a.second == b.second;
        }
    };

    struct comp_neighbor_unique_sort {
        bool operator ()(std::pair<double,long> const& a, std::pair<double, long> const& b) const {
          if (a.second < b.second)
            return true;
          else if (a.second == b.second && a.first < b.first)
            return true;
          else 
            return false;
        }
    };

};


void print_data(fksData *data, MPI_Comm comm);

void print_data(fksData *data, map<long, int> &mymap, MPI_Comm comm);

void print_data(binData *data, MPI_Comm comm);

void print_set(set< triple<long, long, int> > &myset, MPI_Comm comm);

void print_knn(vector<long> &queryIDs, vector< pair<double, long> > *knns, MPI_Comm comm);

void print_tree_single(fks_ompNode *inNode);

void print_tree(fks_ompNode *inNode, MPI_Comm comm);

void print_let_node_list_single(vector<fks_ompNode*>& list);

void print_let_node_list(vector<fks_ompNode*>& list);


} // namespace

#endif
