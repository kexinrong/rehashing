
#ifndef ASKIT_UTILS_HPP_
#define ASKIT_UTILS_HPP_

//#include <mkl.h>
//#include <mkl_lapacke.h>

// Trying this for mkl inclusion weirdness
//#include "id.hpp"
#include "fks_ompNode.h"
#include "parallelIO.h"
#include "generator.h"
#include "blas_headers.hpp"
// #include "fksTree.h"

/**
 * A couple of utility functions that get used in more than one kernel summation
 * algorithm. 
 */

namespace askit {

  // Computes K * q -- is just a wrapper for calls to compute kernel and 
  // blas matvec
  template<class TKernel>
  void ComputeKernelMatvec(TKernel& kernel, std::vector<double>& K, 
    std::vector<double>::iterator target_begin, std::vector<double>::iterator target_end,
    std::vector<double>::iterator source_begin, std::vector<double>::iterator source_end, 
    std::vector<double>::iterator charge_begin, 
    std::vector<double>::iterator potentials_out_begin, int dim, std::vector<int>& source_inds);
    
  // Computes K * q when K has exactly one row -- uses dot instead of matvec
  template<class TKernel>  
  double ComputeKernelDotProd(TKernel& kernel, std::vector<double>& K,
    std::vector<double>& target, 
    std::vector<double>::iterator source_begin, std::vector<double>::iterator source_end,
    std::vector<double>::iterator charge_begin, int dim, std::vector<int>& source_inds);
    

  // Builds the list reprentation of the tree needed by the parallel upward pass
  // Called with the root and an empty list to begin construction
  void ConstructTreeList(fks_ompNode* node, std::vector<fks_ompNode*>& list,
    int& max_tree_level);
  
  // True if the MID belongs to the node
  bool NodeOwnsPoint(fks_ompNode* node, long point_mid, int global_level = 0);
  
  // nodes are pairs <MID, level> representing tree nodes
  // This function returns true if node1 comes before node2 in a postorder tree
  // traversal
  bool LessThanTreeOrder(const pair<long, int>& node1, const pair<long, int>& node2);
  
  // returns true if par is an ancestor of child, false otherwise
  bool isAncestor(const pair<long, int>& par, const pair<long, int>& child);
  
  // Computes the minimum and maximum value the kernel can take between the 
  // query point and the node with bounding ball given by centroid and radius
  template<class TKernel>
  std::pair<double, double> ComputeNodeBounds(TKernel& kernel, 
      std::vector<double>& query, std::vector<double>& centroid, double radius);

  // Inverts the contents of target_to_node.
  // 
  // target_to_node is a list of nodes for each target.  Inverting it means that
  // we compute a list of targets for each node, which is output in 
  // node_to_target
  void InvertInteractionList(vector<vector<int> >& target_to_node, vector<vector<int> >& node_to_target);

  // Reads in the data set, distributes among processes, and fills in the 
  // global ids
  fksData* ReadDistData(string& data_filename, string& charge_filename, long glb_N, int d, bool is_binary_file);


  // Identifies which gids will be used to estimate error and collects the 
  // coordinates of all of these points on all MPI processes
  //
  // -- error_point_coordinates -- coordinates of all points used to estimate the 
  // approximation error
  // -- my_error_gids -- the global ids of error points that this MPI rank owns
  // -- num_error_gids -- the number of error points per MPI rank
  // -- error_gids -- the global ids of all the error check points 
  void CollectErrorCheckCoordinates(vector<double>& error_point_coordinates, 
    vector<long>& my_error_gids, vector<int>& num_error_gids, vector<long>& error_gids,
    fksData* refData, long global_N, int num_error_checks);

  // Stores the inputs for the askit algorithm
  // Just makes the mains more organized
  class AskitInputs {

  public:

    // The number of targets to use when constructing the skeleton
    int num_skel_targets;
    // The rank of ID to use for skeleton construction
    int rank;
    // The maximum number of points in the metric tree
    int max_points_per_node;
    // The maximum depth of the metric tree
    int max_tree_level;
    // The number of neighbors per point to use
    int num_neighbors_in;
    // The number of neighbor search iterations
    //int num_neighbor_iterations;
    // The minimum level (from the root) at which we build a skeleton
    int min_skeleton_level;
    // The amount of oversampling to do when looking for uniform targets
    int oversampling_factor_in;
    // The minimum number of uniform samples we always take 
    int num_uniform_required;
    
    // filename containing the KNN info
    std::string knn_info_file;
    // is the knn file in binary or not
    bool is_binary;
    
    // The maximum level of the kd-tree used for nearest neighbor search
    //int rkdt_maxLevel;
    // The maximum points per node of the kd-tree used for nearest neighbor 
    // search
    //int rkdt_mppn;
    // flag to compress self interactions 
    bool compress_self;
    // flag to use self interactions in target list when computing skeletons
    bool skeletonize_self;
    // flag to use the adaptive rank selection criterion
    // In this case, rank becomes the max rank
    bool use_adaptive_id;
    // flag to do the adaptive rank selection with the simplified heuristic
    // fabs(r_ss / r_11) < epsilon
    bool use_simplified_adaptive_id;
    // in the simplified case, if this is false, we use the absolute condition
    // fabs(r_ss) < epsilon instead
    bool do_absolute_id_cutoff;
    // flag to only do direct interactions at the leaves (and prune everything
    // else) while still using neighbors for sampling
    bool traverse_to_self_only;
    // flag to use the absolute number of targets
    // otherwise, the number of targets for a skeleton is fixed as 
    // num_skel_targets * M log M + num_uniform_required for M columns
    bool do_absolute_num_targets;
    
    // Flag to control the adaptive level restriction -- i.e. to never prune
    // a node that we couldn't compress below rank s_max
    bool do_adaptive_level_restriction;

    // the absolute error tolerance for the adaptive rank id
    double id_tol;
    
    // do the scaled near field version of the adaptive rank algorithm
    bool do_scale_near_adaptive;
  
    // if true, we don't store the P matrix in the skeletons
    // If set: don't run UpdateCharges
    bool dont_store_proj;

    // If true, we do the FMM version of the algorithm    
    bool do_fmm;
    
    // If true, we attempt to split nodes to get as much merging as possible in 
    // the creation of the FMM node-to-node interaction lists
    bool merge_aggressive;
    
    // Flag to save the training set potentials
    bool save_training_potentials;
    
    // filename for training set potentials
    char* training_potentials_filename;
    
    // Flag for the version where we split the neighbors into pruning and 
    // sampling sets
    bool do_split_k;

    // The number of pruning neighbors to use
    // k = pruning_num_neighbors + sampling_num_neighbors
    int pruning_num_neighbors;
    
    // The number of neighbors to pass up in the skeletonization phase
    // defaults to 2 for now
    int neighbors_to_pass_up;
    

    AskitInputs(int num_skel_targets, int rank,
      int max_points_per_node, int max_tree_level, int num_neighbors_in,
      int min_skeleton_level, int oversampling_factor_in, int num_uniform_required,
      const char* knn_info, bool binary=false)
      :
      num_skel_targets(num_skel_targets),
      rank(rank),
      max_points_per_node(max_points_per_node),
      max_tree_level(max_tree_level),
      num_neighbors_in(num_neighbors_in),
      min_skeleton_level(min_skeleton_level),
      oversampling_factor_in(oversampling_factor_in),
      num_uniform_required(num_uniform_required),
      knn_info_file(knn_info),
      compress_self(false),
      skeletonize_self(false),
      use_adaptive_id(false),
      use_simplified_adaptive_id(false),
      do_adaptive_level_restriction(false),
      traverse_to_self_only(false),
      // this defaults to true to make the old tests work, however, the
      // main will set it to false by default
      do_absolute_num_targets(true),
      do_scale_near_adaptive(false),
      id_tol(0.0),
      is_binary(binary),
      dont_store_proj(false),
      do_fmm(false),
      merge_aggressive(false),
      save_training_potentials(false),
      // the relative criterion is the old default
      do_absolute_id_cutoff(false),
      do_split_k(false),
      pruning_num_neighbors(0), 
      neighbors_to_pass_up(4)
      {}

  }; // class

} // namespace


#include "askit_utils_impl.hpp"

#endif




