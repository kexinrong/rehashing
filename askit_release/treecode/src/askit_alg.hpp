
#ifndef ASKIT_ALG_HPP_
#define ASKIT_ALG_HPP_

#include "kernel_inputs.hpp"
#include "id.hpp"
#include "askit_utils.hpp"
#include "fksTree.h"
#include "ompUtils.h"

#include "parallelIO.h"


#include <time.h>


namespace askit {


template<class TKernel>
class AskitAlg {

public:

  /**
  * Inputs:
  * - refData -- data structure with point coordinates
  * - kernel_inputs -- input data structure for kernel, contains bandwidth, etc.
  * - askit_inputs -- input data structure containing tree parameters, etc.
  *
  * These are the order of inputs to askit_inputs:
  * - num_skel_targets -- number of targets required to build an ID. If there 
  * aren't this many neighbors, will sample the far field.
  * - rank -- Number of columns to use in an ID.
  * - max_points_per_node -- Maximum leaf size for the fks (fast kernel 
  * summation) tree.
  * - max_tree_level -- Maximum depth of the fks tree
  * - num_neighbors_in -- Number of neighbors of each point to compute.
  * - num_neighbor_iterations -- Number of random trials in NN search.
  * - min_skeleton_level -- Closest level to the root at which skeletons are 
  * computed.  IMPORTANT: Don't set this to be less than 2.
  * - oversampling_factor_in -- The amount of oversampling to do in order to 
  * obtain far-field points
  * - rkdt_maxLevel -- the maximum level of the randomized kd-tree used to find
  * nearest neighbors.
  * - rkdt_mppn -- The maximum leaf size in the rkd tree.
  * - do_variable_band -- Turns on the variable bandwidth Gaussian kernel. 
  * Defaults to false.
  */
  AskitAlg(fksData* refData, KernelInputs& kernel_inputs, AskitInputs& askit_inputs);


  ~AskitAlg();

  /*
   * Computes the approximate potential at every query using the tree.
   */
  std::vector<double> ComputeAll();
  
  /*
   * Computes all approximate potentials for the test points.  Assumes we have
   * called AddTestPoints to update the data structures and LET. 
   */
  std::vector<double> ComputeAllTestPotentials();

  
  /**
   * Updates the charge vector for another computation. Calls a new upward 
   * pass, but doesn't do anything else.
   */
  void UpdateCharges(std::vector<double>& new_charges);
  
  
  /**
   * Adds a test set (distinct target set).  
   * 
   * num_test_points -- number of points in the test set total.
   * test_data -- test data for this MPI rank.
   * knn_filename -- file containing nearest neighbor info for the test set.
   * is_binary -- flag, set to true if knn file is binary, false otherwise.
   */
  void AddTestPoints(long num_test_points, fksData* test_data, const char* knn_filename, bool is_binary);
  
  /**
   * Computes the exact potential for all of the target coordinates.
   *
   * Returns the potentials on MPI rank 0. 
   */  
  std::vector<double> ComputeDirect(vector<double>& target_coordinates);

  /**
   * Computes the potentials from nearest neighbors only.
   * 
   * my_test_gids -- The list of global ids of targets owned by this MPI rank.
   * my_test_lids -- The list of target local ids (on this rank).
   * num_test_lids -- The number of targets owned by each rank.
   * displ -- The displacement of target ids for each rank (for MPI collectives).
   */
  vector<double> ComputeDistNN(vector<long>& my_test_gids, vector<int>& my_test_lids, 
    vector<int>& num_test_lids, vector<int>& displ);


  /**
   * Computes the potentials for test points from nearest neighbors only.
   * 
   * my_test_gids -- The list of global ids of targets owned by this MPI rank.
   * my_test_lids -- The list of target local ids (on this rank).
   * num_test_lids -- The number of targets owned by each rank.
   * displ -- The displacement of target ids for each rank (for MPI collectives).
   */
  vector<double> ComputeTestDistNN(vector<long>& my_test_gids, vector<int>& my_test_lids, 
    vector<int>& num_test_lids, vector<int>& displ);


  /** 
   * Collects the computed potentials from the different ranks.
   * 
   * my_test_gids -- The list of global ids of targets owned by this MPI rank.
   * my_test_lids -- The list of target local ids (on this rank).
   * num_test_lids -- The number of targets owned by each rank.
   * displ -- The displacement of target ids for each rank (for MPI collectives).
   */
  vector<double> CollectPotentials(vector<double>& potentials, vector<long>& my_test_gids,
    vector<int>& my_test_lids, vector<int>& num_test_lids, vector<int>& displ);


  // Just returns the bandwidth of the Gaussian kernel if needed elsewhere
  double bandwidth() const { return kernel_params.bandwidth; }


  ///////////// Counters and Timers /////////////////////////////////////
  
  // The number of kernel evaluations in the downward pass of the algorithm
  long num_downward_kernel_evals;
  
  // counter for number of nodes that used uniform samples in skeletonization
  long num_nodes_with_uniform;
  
  // Counters for the adaptive rank version
  long total_skeleton_size;

  // The size of skeleton for each node constructed locally
  // the pair is <size, tree level>
  vector<pair<int, int> > self_skeleton_sizes;
    
  
  /////////// timers ///////////////
  
  // Time taken to build trees and exchange NN information
  double tree_build_time;
  
  // Time taken to perform the upward pass and compute skeletons
  double skeletonization_time;
  
  // Time taken for inverting and blocking the interaction lists
  double list_blocking_time;

  // Time to compute evaluation phase for all targets
  double evaluation_time;

  // Time taken for direct (exact) computations, if performed
  double exact_comp_time;
  
  // Time taken to build and exchange LET -- this also does the traversal for 
  // the target interaction lists
  double let_traversal_time;
  
  // Time taken to call update charges
  double update_charges_time;
  
  
  ///////////// Timers for test point evaluation ////////////////
  
  // Time to read in the test set and exchange points between ranks
  double dist_test_set_time;

  // Time to update the LET for test point evaluation, also includes forming
  // the test target point interaction lists
  double update_test_let_time;
  
  // Time to compute evaluation phase for test point targets
  double test_evaluation_time;

  // Time taken to build interaction lists for test points
  double test_list_blocking_time;

  
  
  ///////////// Parameters ///////////////
  
  // number of target points owned by this node
  // we only need to compute potentials for the first N points 
  int N;
  
  // total number of points among all processes
  long global_N;
  
  // number of test points owned by this process
  int N_test;
  
  // total number of test points
  long global_N_test;
  
  // The size of skeletons for each node in my LET
  vector<int> skeleton_sizes;
   
  // a scan of the above -- used for constructing and updating the charge table
  vector<int> skeleton_sizes_scan;

  // the collected charges of the skeleton points
  vector<double> skeleton_charges;
 
  // The total size of all of the approximate interaction lists for all 
  // target points
  long leaf_list_size_before;

  // The size of approximate interaction lists for each target point after
  // we have merged and removed node-to-node interactions
  long leaf_list_size_after;
 
  long node_node_list_size;
 
 
  /////////////
  
  // Pointer to the tree
  fksTree* tree;
  
  // The L1 norm of the charge vector -- used in error estimation
  double charge_l1_norm;
  double charge_l2_norm_sqr;
  
  // For the adaptive case with near field scaling, this is the minimum 
  // value S_N across all leaves 
  // public so I can report it from runs
  double near_scale;
  
  
  
  ///////////////////////////////////////////////
  
  // extra timers
  
  double construct_tree_list_time;
  double compute_leaf_neighbors_time;
  double merge_neighbor_lists_time;
  double collect_local_samples_time;
  double collect_neighbor_coords_time;
  double distributed_upward_pass_time;
  double uniform_sample_sibling_time;
  double dist_compute_skeleton_time;
  double merge_skeletons_of_kids_time;
  double merge_nn_list_time;
  double get_dist_node_excl_nn_time;
  double compute_adaptive_id_time;
  double apply_proj_time;
  double subsample_self_targets_time;
  double compute_skeleton_time;
  double solve_for_proj_time;
  double qr_time;
  double max_qr_time;
  double kernel_compute_time;
  double exchange_let_time;
  
  // FMM implementation timers
  double pass_potentials_down_time;
  double merge_fmm_lists_basic_time;
  double merge_basic_set_difference_time;
  double fmm_add_to_map_time;
  
  // Aggressive merging timers
  double merge_fmm_lists_aggressive_time;
  double less_than_tree_order_time;
  double is_ancestor_time;
  double split_node_time;
  double merge_tree_list_full_time;
  
  // New adaptive rank skeletonization timers
  double compute_near_scale_time;
  
  // adaptive level restriction timers
  double frontier_exchange_time;
  double prune_by_knn_time;
  
  
protected:

  /////////////////////// parameters ///////////////////////
 
  KernelInputs kernel_params;
 
  // The number of targets that we use to build a skeleton
  // If we don't have enough neighbors, we get some uniformly random points
  // In the adaptive case, this is added to the number of sources
  // In the non-adaptive case, this is absolute
  int num_skeleton_targets;
  
  // The rank of ID to compute
  // For adaptive, this is the maximum possible rank
  int id_rank;
  
  // Number of neighbors
  int num_neighbors_per_point;
  
  // Note that pruning_num_neighbors + sampling_num_neighbors = num_neighbors_per_point

  // Flag for whether we are going to split 
  bool do_split_k;

  // number of neighbors used for sampling
  int sampling_num_neighbors;
  
  // number of neighbors used for pruning
  int pruning_num_neighbors;
  
  // Scaling factor for the number of neighbors in the list to pass up 
  // to the next level in skeletonization
  int neighbors_to_pass_up;
  
  
  // The dimension of the data
  int dim;
  
  // How much oversampling do we do to get a good uniform sample of the far field?
  int oversampling_factor;
  
  // The minimum number of uniform samples that we always take
  int num_uniform_required;

  // at any level of the tree above this one, we don't try to construct a skeleton and never prune
  int min_skeleton_level;
  
  // The global number of processors -- used so that we know if we need to 
  // deal with a LET
  int comm_size;
  
  // MPI rank of this process
  int my_mpi_rank;
  
  // The size of leaves used in the tree construction
  int leaf_size;
  
  // Flag to set whether we want to compute exact self-interactions or not
  // defaults to false
  // If true, then we always use skeletons to evaluate
  bool compress_self_interactions;
  
  // Do we want to include the self interactions in the computation of 
  // skeletons?
  bool skeletonize_self_interactions;
  
  // If true, then we compute the rank of skeleton to use adaptively
  bool use_adaptive_id;

  // If true, then we do the adaptive rank selection with the simplified 
  // cutoff criterion:
  // we stop when fabs(r_ss / r_11) < epsilon
  bool use_simplified_adaptive_id;
  
  // If false, then we use an absolute condition fabs(r_ss) < epsilon
  bool do_absolute_id_cutoff;
  
  // Flag that determines whether we include the near field scale S_N in 
  // the adaptive rank estimation
  bool do_scale_near_adaptive;
  
  // If this flag is set, then we stop skeletonizing (and don't prune the node)
  // if we cannot compress below rank s_max
  // This only applies with the simplified adaptive rank algorithm
  bool do_adaptive_level_restriction;
  
  // If true, traverses the tree as if k = 1, even though the larger value of 
  // k is used for sampling
  bool traverse_to_self_only;
  
  // If true, then we use the number of targets as an absolute quantity
  // if false, then we use num_skeleton_targets * M log M + num_uniform_required
  // targets for M columns
  bool do_absolute_num_targets;
  
  // If true, we no longer store the skeleton's matrix P
  // WARNING: if this is true, UpdateCharges will currently fail
  bool dont_store_proj;
  
  // For the adaptive ID case, this is the absolute error cutoff in our  
  // singular value estimate
  double id_tol;
  
  // number of levels above root_omp -- used for checking node membership from
  // MIDs
  int num_global_levels;
  
  // Flag to determine the merging strategy used in creating FMM node-to-node
  // interaction lists
  bool merge_aggressive;
  
  // Controls whether we do the FMM version of the algorithm
  bool do_fmm;
  
  /////////// Extra counters for test point implementation

  // The size of letNodeList before it gets updated
  long old_let_size;
  
  // The size of tree->inProcData before we update with test points
  long old_inprocdata_size;
  
  // The size of charge_table before we call UpdateTestMergedInteractionList
  long old_charge_table_size;
  
  // we use this to determine where the original skeleton indices end and the 
  // ones for new skeletons introduced for test points starts
  // If a node's skeleton_size_scan + its skeleton size is larger than this, 
  // then it is a testing node 
  long training_skeleton_size_cutoff;
  
  
  // local memory to work with
  // I will need some storage to play with for ID, kernel sums, etc.
  // These are allocated once and re-used.
  // The vectors are indexed by thread id

  // The kernel object has some internal storage.  
  std::vector<TKernel*> kernels;

  // Workspace for ID's
  std::vector<IDWorkspace*> id_workspaces;

  // Space for a kernel matrix
  std::vector<std::vector<double> > kernel_mat_space;

  // data for leaves 
  std::vector<fksData*> thread_leaf_data;
  
  // data for leaf neighbors
  std::vector<fksData*> thread_leaf_nn_data;
  
  // This is the list of nodes that I own that can be skeletonized but whose
  // parent can't be
  // It's empty if all my nodes (up to the one in the distributed tree where 
  // I'm local rank 1) can be skeletonized
  vector<pair<long, int> > my_skeleton_frontier;
  
  
  //////////// Interaction List stuff ///////////
  
  // direct_interactions[i] is a list of node local ids for which 
  // query point i needs to compute a direct interaction
  //std::vector<std::vector<int> > direct_interactions;
  std::vector<std::vector<int> > direct_target_to_node;
  
  // approx_interactions[i] is the list of node far-field ids for which 
  // target point i needs to compute a direct interaction
  //std::vector<std::vector<int> > approx_interactions;
  std::vector<std::vector<int> > approx_target_to_node;
  
  // list of target local ids for which node with local id i needs to compute 
  // direct interaction
  //std::vector<std::vector<int> > node_interactions;
  std::vector<std::vector<int> > direct_node_to_target;
  
  // For each node, a list of the local ids owned by the node
  // don't need this anymore, is stored in leaf omp_Nodes
  //std::vector<std::vector<int> > node_local_ids;
  
  // list of target local ids for which node with local id i needs to compute 
  // the far-field interaction
  //std::vector<std::vector<int> > far_field_interactions;
  std::vector<std::vector<int> > approx_node_to_target;


  /////// These versions are the same, but for the test points

  // direct_interactions[i] is a list of node local ids for which 
  // query point i needs to compute a direct interaction
  //std::vector<std::vector<int> > direct_interactions;
  std::vector<std::vector<int> > direct_test_target_to_node;
  
  // approx_interactions[i] is the list of node far-field ids for which 
  // target point i needs to compute a direct interaction
  //std::vector<std::vector<int> > approx_interactions;
  std::vector<std::vector<int> > approx_test_target_to_node;
  
  // list of target local ids for which node with local id i needs to compute 
  // direct interaction
  //std::vector<std::vector<int> > node_interactions;
  std::vector<std::vector<int> > direct_test_node_to_target;
  
  // For each node, a list of the local ids owned by the node
  // don't need this anymore, is stored in leaf omp_Nodes
  //std::vector<std::vector<int> > node_local_ids;
  
  // list of target local ids for which node with local id i needs to compute 
  // the far-field interaction
  //std::vector<std::vector<int> > far_field_interactions;
  std::vector<std::vector<int> > approx_test_node_to_target;



  //////////////// Merged Interaction Lists /////////////////////////////
  
  std::vector<double> charge_table;
  std::vector<std::vector<int> > node_source_inds;
  
  std::vector<std::vector<int> > training_target_inds;
  vector<vector<int> > test_target_inds;
  
  std::vector<std::vector<int> > charge_inds;

  // tells us where to store the potentials for the FMM version of the algorithm
  vector<vector<int> > potential_map;

  // Need to save these separately to preserve order correctly when adding test 
  // points later on  
  vector<vector<int> > approx_source_inds;
  vector<vector<int> > direct_source_inds;

  vector<vector<int> > approx_charge_inds;
  vector<vector<int> > direct_charge_inds;
  

  // We need test point versions of all of these because we may want to do 
  // more training after testing
  vector<vector<int> > test_approx_source_inds;
  vector<vector<int> > test_direct_source_inds;

  vector<vector<int> > test_approx_charge_inds;
  vector<vector<int> > test_direct_charge_inds;
  
  std::vector<std::vector<int> > test_node_source_inds;
  std::vector<std::vector<int> > test_charge_inds;



  // Don't need any of this with new LET representation
  
  // List representation of the tree
  // node n is at index (1 << n.level)-1 + n.lnid
  std::vector<fks_ompNode*> tree_list;
  // The maximum depth of any leaf in the tree 
  int max_tree_level;
  
  int num_omp_leaves;  
  
  // List representation of the LET nodes
  //std::vector<fks_ompNode*> let_node_list;
  
  
  double debug_potential_before;
  double debug_potential_after;
  

  ///////////// methods ///////////////
  
  // Constructs the skeletons in parallel by traversing the tree level by 
  // level
  void ParallelUpwardPass();
  
  // Computes skeletons for the distributed tree using communication
  // The arguments are the left and right children's neighbor list of this 
  // MPI rank's root_omp node
  void DistributedUpwardPass(std::vector<std::pair<long, double> >& left_list,
      std::vector<std::pair<long, double> >& right_list,
      vector<long>& left_pruning_list,
      vector<long>& right_pruning_list);
  
  // Fills in (and truncates) the list of neighbor indices for a leaf node
      // returns the number of points in the node
  int ComputeLeafNeighbors(fks_ompNode* node, std::vector<std::pair<long, double> >& neighbor_inds);
  
  // Applies the matrix P to compute the effective charges in skeletonization
  void ApplyProj(vector<double>& charges_in, vector<double>& charges_out,
    vector<double>& proj, int rank, vector<lapack_int>& perm);

  // updates the skeleton charges
  void UpdateSkeletonCharges();

  // Computes the estimate of the near field contribution in the adaptive 
  // rank algorithm
  void ComputeNearScale();

  
  // This version takes in the pre-processed vectors of coordinates and charges
  // and fills in the skeleton
  // Assumes that the skeleton is already allocated 
  // Returns false if the skeletonization fails in the adaptive case, true otherwise
  // m is the number of points owned by the node
  void ComputeSkeleton(fksData* my_skeleton, long m,
  std::vector<double>& source_vec, std::vector<double>& charge_vec,
  std::vector<long>& source_inds, 
  vector<double>& near_vec, vector<double>& unif_vec, bool printout = false);
  

  // This version collects the coordinates from the given indices and node 
  // information 
  void ComputeSkeleton(fks_ompNode* node, std::vector<std::pair<long, double> >& neighbor_inds,
      std::vector<double>& uniform_samples);


  // Collects the coorindates of the given neighbor pairs into target_vec
  void CollectNeighborCoords(std::vector<std::pair<long, double> >& neighbor_inds,
      std::vector<double>& target_vec, int num_targets_needed);


  // Updates the interaction lists after we have added test points
  void UpdateTestMergedInteractionLists();


  // Give  the number of samples needed, computes their indices from 
  // SampleFarField, then returns the vector of coordinates
  std::vector<double> CollectLocalSamples(fks_ompNode* node, 
    std::vector<std::pair<long, double> >& neighbor_inds,
    int num_points_needed);
  
  // Obtains num_needed samples from points not in node or forbidden_inds
  std::vector<int> SampleFarField(int num_needed, std::vector<int>& forbidden_inds, 
                                  fks_ompNode* node);
  
  // Prints the skeleton for debugging purposes
  void PrintSkeleton(fksData* skel);
  
  // Goes through the LET to set up node ids and node_local_ids_let
  //void ProcessLET(fks_ompNode* node);
  
  // Given the list of direct_interactions per query point, we need to 
  // turn it into one for each source node 
  // approx_tn and direct_tn are the approximate and direct target to node lists
  // approx_nt and direct_nt are the node to target lists (filled in by this 
  // function)
  void ComputeNodeInteractionList(vector<vector<int> >& approx_tn, vector<vector<int> >& direct_tn,
    vector<vector<int> >& approx_nt, vector<vector<int> >& direct_nt);
  
  // Creates and fills in the interaction lists for source nodes -- 
  // direct_source_inds, approx_source_inds, direct_charge_inds, 
  // approx_charge_inds.
  // Also creates and fills in skeleton_sizes and skeleton_sizes_scan
  void CreateSourceInteractionLists();

  // updates the charge table in between iterations
  void UpdateChargeTable();

  // After we have collected the interaction lists, this does the near field 
  // computation
  std::vector<double> ComputeNearField();
  
  // Compute the far-field interactions
  std::vector<double> ComputeFarField();
  
  // This is the interface template for either the GNF library, or just 
  // evaluation with OpenMP and BLAS
  void ComputeAllNodeInteractions(vector<double>& target_coordinates, 
    vector<double>& source_coordinates,
    vector<double>& charges,
    vector<vector<int> >& potential_inds,
    vector<vector<int> >& source_inds,
    vector<vector<int> >& target_inds,
    vector<vector<int> >& charge_inds,
    vector<double>& potentials_out);
  
    // This computes a single interaction
  void ComputeNodeInteraction(double* target_coordinate_table, double* source_coordinate_table,
    double* charge_table, std::vector<int>& potential_inds,
    std::vector<int>& source_inds, std::vector<int>& target_inds, std::vector<int>& charge_inds, 
    std::vector<double>& potentials);

  
  // Merges the two neighbor lists, removing duplicates 
  // left_list, right_list -- lists for the children
  // left_pruning_list, right_pruning_list -- list of neighbors in teh 
  // pruning lists for the left and right children
  // output_list -- on exit, contains the merged lists
  // node -- Need to know which node this is for so that we can exclude
  // neighbors that are owned by the list
  // Returns: number of source points
  int MergeNeighborLists(const std::vector<std::pair<long, double> >& left_list, 
    const std::vector<std::pair<long, double> >& right_list, 
    const vector<long>& left_pruning_list, const vector<long>& right_pruning_list,
    std::vector<std::pair<long, double> >& output_list,
    fks_ompNode* node);
    
  
  // Takes in a vector of possible target coordinates and subsamples it to 
  // have the desired number of targets  
  std::vector<double> SubsampleSelfTargets(const std::vector<double>& target_coords, int num_targets_needed);
  
  // u is the vector of potentials -- first N entries are for the targets 
  // themselves, rest are for the skeleton points
  void PassPotentialsDown(vector<double>& u);
  
  // Computes the interaction lists for the FMM by merging the lists for 
  // the targets in each node
  void ComputeFMMInteractionLists(vector<vector<int> >& approx_target_to_node,
    vector<vector<int> >& target_nodes_for_source);
  
  // Just performs a simple intersection of the two sorted lists list1 and list2
  // storing the output in list_out.  Then removes any elements of list_out from
  // list1 and list2
  void MergeTargetListsFMMBasic(vector<int>& list1, 
    vector<int>& list2, vector<int>& list_out);
  
  // Merges the lists (sorted in pre-order traversal order)
  void MergeTargetListsFMMFull(vector<int>& list1, 
    vector<int>& list2, vector<int>& list_out);

  // Returns true if the first node is an ancestor of the second
  // Inputs are <Morton ID, level>
  bool isAncestor(pair<long, int>& par, pair<long, int>& child);

  // Splits the given node <Morton ID, level> the number of times and returns
  // the resulting array of nodes
  vector<pair<long, int> > SplitNode(pair<long, int>& val, int num_splits);
  
  
  bool LessThanTreeOrder(pair<long, int>& node1, pair<long, int>& node2);
  
  // Comparison structure for neighbor index, distance pairs -- we want to sort by the index
  struct comp_neighbor_index {
      bool operator ()(std::pair<long,double> const& a, std::pair<long, double> const& b) const {
        return a.first < b.first;
      }
  };

  // Comparison structure for neighbor index, distance pairs -- we want to sort by the index
  struct eq_neighbor_index {
      bool operator ()(std::pair<long,double> const& a, std::pair<long, double> const& b) const {
        return a.first == b.first;
      }
  };

  // Comparison structure for both -- Use this one before calling unique -- this 
  // will ensure that unique returns the instance of the neighbor with the closest
  // distance
  /*
  struct comp_neighbor_unique_sort {
      bool operator ()(std::pair<long,double> const& a, std::pair<long, double> const& b) const {
        if (a.first < b.first)
          return true;
        else if (a.first == b.first && a.second < b.second)
          return true;
        else 
          return false;
      }
  };
  */
  struct comp_neighbor_unique_sort {
      bool operator ()(std::pair<long,double> const& a, std::pair<long, double> const& b) const {
        if (a.first < b.first)
          return true;
        else if (a.first == b.first && a.second < b.second)
          return true;
        else 
          return false;
      }
  };

  // Comparison structure for neighbor distances -- sorts by dist to find closest 
  struct comp_neighbor_dist {
      bool operator ()(std::pair<long,double> const& a, std::pair<long, double> const& b) const {
        return a.second < b.second;
      }
  };


  

}; // class


} // namespace 

#include "askit_alg_impl.hpp"


#endif




