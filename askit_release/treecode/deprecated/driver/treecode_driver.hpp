
#ifndef TREECODE_DRIVER_HPP_
#define TREECODE_DRIVER_HPP_

// TODO: make cmake do this correctly
#include "../kernel/gaussian_kernel.hpp"
#include "../id/id.hpp"
#include "../fkstree/fksTree.h"

class TreecodeDriver {

public:
  
  TreecodeDriver(fksData* refData, double bandwidth, int num_skel_targets, int rank,
int max_points_per_node, int max_tree_level, int min_comm_size_per_node,
int num_neighbors_in, int num_neighbor_iterations, int min_skeleton_level, int oversampling_factor_in);
  
  ~TreecodeDriver();
  
  double Compute(int query_lid);

  double ComputeNaive(long query_id);
  
  double ComputeNN(int query_id);
  
  
protected:

  class QueryPoint
  {
  
  public: 
    
    std::vector<double> coords;
    
    std::vector<int> nn_morton_ids;
    
    double potential;
    
    QueryPoint(std::vector<double>& coords_in, std::vector<int>& ids_in)
      :
    coords(coords_in),
    nn_morton_ids(ids_in),
    potential(0.0)
    {}
  
    ~QueryPoint() {}
    
  }; // class QueryPoint

  // The data owned by this processor
  fksData* my_data;

  fksTree* tree;
  
  // parameters
  
  // The number of targets that we use to build a skeleton
  // If we don't have enough neighbors, we get some uniformly random points
  int num_skeleton_targets;
  
  // The rank of ID to compute
  int id_rank;
  
  // Number of neighbors
  int num_neighbors_per_point;
  
  // The dimension of the data
  int dim;
  
  // How much oversampling do we do to get a good uniform sample of the far field?
  int oversampling_factor;

  // at any level of the tree above this one, we don't try to construct a skeleton and never prune
  int min_skeleton_level;
  
  
  // local memory to work with
  // I will need some storage to play with for ID, kernel sums, etc.
  
  // This is a vector because each thread needs it's own kernel workspace
  std::vector<GaussianKernel> kernels;

  // So that threads can compute ID's in parallel
  std::vector<IDWorkspace> id_workspaces;

  // workspace to build kernel matrices in 
  std::vector<double*> kernel_mat_space;

  // data for leaves 
  std::vector<fksData*> thread_leaf_data;
  
  // data for leaf neighbors
  std::vector<fksData*> thread_leaf_nn_data;
  

  // methods
  
  std::vector<long> UpwardPass(fks_ompNode* node);
  
  void ComputeSkeleton(fks_ompNode* node, std::vector<long>& neighbor_inds);
  
  bool CanPrune(fks_ompNode* node, QueryPoint& query);
  
  void DownwardPass(fks_ompNode* node, QueryPoint& query);
  
  std::vector<int> SampleFarField(int num_needed, std::vector<int>& forbidden_inds, fks_ompNode* node);
  
  bool NodeOwnsPoint(fks_ompNode* node, long point_mid);

}; // class


#endif
