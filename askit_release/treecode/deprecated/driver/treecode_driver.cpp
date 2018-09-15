
#include "treecode_driver.hpp"


TreecodeDriver::TreecodeDriver(fksData* refData, double bandwidth, int num_skel_targets, int rank,
  // rest is for the tree
int max_points_per_node, int max_tree_level, int min_comm_size_per_node,
int num_neighbors_in, int num_neighbor_iterations, int min_skel_level, int oversampling_factor_in) 
  :
num_skeleton_targets(num_skel_targets),
id_rank(rank),
num_neighbors_per_point(num_neighbors_in),
min_skeleton_level(min_skel_level),
  oversampling_factor(oversampling_factor_in)
{

  int num_threads = omp_get_num_threads();

  // TODO: is this big enough? It should give me an error message if it's not
  int max_kernel_mat_size = std::max(max_points_per_node * num_skel_targets, 2 * id_rank * num_skel_targets);
  // need this if we're going to do an exact computation
  max_kernel_mat_size = std::max(max_kernel_mat_size, refData->numof_points);

  kernels.resize(num_threads, GaussianKernel(bandwidth, max_kernel_mat_size));

  // We do the ID on either leaves, or the union of the children's skeletons
  int max_id_cols = std::max(max_points_per_node, 2 * id_rank);
  id_workspaces.resize(num_threads, IDWorkspace(rank, max_id_cols));

  // allocate space to construct kernel submatrices in
  kernel_mat_space.resize(num_threads);

  thread_leaf_data.resize(num_threads);
  
  thread_leaf_nn_data.resize(num_threads);
  
  // TODO: in parallel
  for (int i = 0; i < num_threads; i++)
  {
    kernel_mat_space[i] = new double[max_kernel_mat_size];
    thread_leaf_data[i] = new fksData();
    thread_leaf_nn_data[i] = new fksData();
  }

  // TODO: make this a tuning parameter
  oversampling_factor = 10;


  fksCtx *ctx = new fksCtx();
  ctx->k = num_neighbors_per_point;
  ctx->minCommSize = min_comm_size_per_node;
  ctx->rkdt_niters = num_neighbor_iterations;
  ctx->rkdt_mppn = max_points_per_node;
  ctx->rkdt_maxLevel = max_tree_level;
  ctx->fks_mppn = max_points_per_node;
  ctx->fks_maxLevel = max_tree_level;

  tree = new fksTree;
  tree->build(refData, ctx);
  
  dim = tree->inLeafData->dim;
  
  // initialize by computing for the nearest neighbors
  // Do a bottom up traversal
  UpwardPass(tree->root_omp);
  
  delete ctx;
  
}

TreecodeDriver::~TreecodeDriver()
{

  if (tree)
  {
    delete tree;
    tree = NULL;
  }
  
  int num_threads = omp_get_num_threads();
  
  // TODO: in parallel?
  for (int i = 0; i < num_threads; i++)
  {
    delete[] kernel_mat_space[i];
    delete thread_leaf_data[i];
    delete thread_leaf_nn_data[i];
  }
  
}


void TreecodeDriver::ComputeSkeleton(fks_ompNode* node, std::vector<long>& neighbor_inds)
{
  
  // for leaves, fill out neighbor inds
  // for internal nodes, receives it filled out
  
  int my_thread_id = omp_get_thread_num();
  
  fksData* my_skeleton = node->skeletons;
  
  // TODO: is this right? 
  if (my_skeleton == NULL)
  {
    node->skeletons = new fksData();
    my_skeleton = node->skeletons;
  }
    
  int num_sources;
  int num_targets;
  
  std::vector<double> source_vec;
  std::vector<double> target_vec;
  
  std::vector<double> charge_vec;
  
  // are we a leaf?
  if (node->leftNode == NULL)
  {

    // Set up the source and target points
    fksData* leaf_data = thread_leaf_data[my_thread_id];
    bool get_data = tree->getLeafData(node, leaf_data);
    if (!get_data)
    {
      std::cout << "Failed to get leaf data\n";
    }
    
    num_sources = leaf_data->numof_points;
    // TODO: this copies, this is unnecessary
    source_vec = leaf_data->X;
    
    fksData* leaf_nn_data = thread_leaf_nn_data[my_thread_id];
    tree->getLeafexclNN(node, leaf_nn_data);
    
    int num_neighbors = leaf_nn_data->numof_points;
    //double* neighbor_points = leaf_nn_data->X.data();

    // set the neighbors for passing up the tree
    neighbor_inds.assign(leaf_nn_data->lids.begin(), leaf_nn_data->lids.end());

    // get the charges
    // TODO: eliminate copying
    charge_vec = leaf_data->charges;
    
    // we have to pad with randomly sampled points
    if (num_neighbors < num_skeleton_targets)
    {
      int num_points_needed = num_skeleton_targets - num_neighbors;
      
      std::vector<int> samples = SampleFarField(num_points_needed, neighbor_inds);
      
      num_targets = num_neighbors + samples.size();
      
      target_vec = leaf_nn_data->X;

      for (int i = 0; i < samples.size(); i++)
      {
        target_vec.insert(target_vec.end(), tree->inLeafData->X.begin() + dim * samples[i], 
          tree->inLeafData->X.begin() + dim * (samples[i] + 1));
      }
      
    }
    else {
      // we don't have to sample anything
      
      num_targets = num_neighbors;
      target_vec = leaf_nn_data->X;
      
    } // done sampling
    
  }// are we a leaf?
  else {
    // we're an internal node
    
    fks_ompNode* left_child = node->leftNode;
    fks_ompNode* right_child = node->rightNode;
    
    fksData* left_skel = left_child->skeletons;
    fksData* right_skel = right_child->skeletons;
    
    // Combine the skeletons of the children
    num_sources = left_skel->numof_points + right_skel->numof_points;
    source_vec = left_skel->X;
    source_vec.insert(source_vec.end(), right_skel->X.begin(), right_skel->X.end());
    
    charge_vec = left_skel->charges;
    charge_vec.insert(charge_vec.end(), right_skel->charges.begin(), right_skel->charges.end());
    
    // neighbor inds was passed in filled out
    int num_neighbors = neighbor_inds.size();

    // get the coordinates of the neighbors
    for (int i = 0; i < num_neighbors; i++) 
    {
      target_vec.insert(target_vec.end(), tree->inLeafData->X.begin()+neighbor_inds[i]*dim, 
        tree->inLeafData->X.begin() + (neighbor_inds[i] + 1)*dim);
    }
    
    if (num_neighbors < num_skeleton_targets)
    {
      
      int num_points_needed = num_skeleton_targets - num_neighbors;
      
      std::vector<int> samples = SampleFarField(num_points_needed, neighbor_inds);
      
      num_targets = num_neighbors + samples.size();
      
      for (int i = 0; i < samples.size(); i++)
      {
        target_vec.insert(target_vec.end(), tree->inLeafData->X.begin() + dim * samples[i], 
          tree->inLeafData->X.begin() + dim * (samples[i] + 1));
      }
      
    } // do we need to sample?
    else {
      // we don't have to sample anything
      
      num_targets = num_neighbors;
      
    }
    
  } // internal node
  
  // Can we compress at all?
  if (num_sources > id_rank)
  {
    std::vector<double> effective_charges(id_rank);
    
    // Compute the kernel matrix
    double* K = kernel_mat_space[my_thread_id];
    kernels[my_thread_id].Compute(target_vec.data(), num_targets, source_vec.data(), num_sources, dim, K);
    
    // Compute the ID and store the skeleton
    std::vector<lapack_int> skeleton_inds(id_rank);
    std::vector<double> proj(id_rank * (num_sources - id_rank));
    compute_id(K, num_targets, num_sources, id_rank, skeleton_inds, proj, id_workspaces[my_thread_id]);
    
    // Compute the effective charges by applying proj to the charges
    cblas_dgemv(CblasColMajor, CblasNoTrans, id_rank, num_sources - id_rank, 1.0, 
      proj.data(), id_rank, charge_vec.data(), 1, 0.0, effective_charges.data(), 1);
    
    // add back the contribution of the skeleton points' charges
    for (int i = 0; i < id_rank; i++)
    {
      effective_charges[i] += charge_vec[skeleton_inds[i]];
    }
  
    my_skeleton->charges.assign(effective_charges.begin(), effective_charges.end());

    // TODO: this can be done more efficiently 
    for (int i = 0; i < id_rank; i++)
    {
      for (int d = 0; d < dim; d++)
      {
        my_skeleton->X.push_back(source_vec[skeleton_inds[i]+d]); 
      }
    } // loop over skeleton to store
  
    my_skeleton->numof_points = id_rank;

  }
  else {
    // we don't have enough sources to compress, so just pass things up the tree
    my_skeleton->charges.assign(charge_vec.begin(), charge_vec.end());
    my_skeleton->X.assign(source_vec.begin(), source_vec.end());
    my_skeleton->numof_points = num_sources;
    
  }
  
  my_skeleton->dim = dim;
  
  // TODO: not setting either of these for now, make sure they aren't needed later
  //my_skeleton->gids;
  //my_skeleton->mortons;
  
} // ComputeSkeleton()



std::vector<long> TreecodeDriver::UpwardPass(fks_ompNode* node)
{
  
  std::vector<long> neighbor_inds;
  
  if (node->leftNode == NULL)
  {
    
    // fills out the inds
    if (node->level >= min_skeleton_level)
      ComputeSkeleton(node, neighbor_inds);
  
  }
  else 
  {

    // bottom up traversal    
    std::vector<long> left_neighbors = UpwardPass(node->leftNode);
    std::vector<long> right_neighbors = UpwardPass(node->rightNode);
    
    if (node->level >= min_skeleton_level) {
      // Merge the left and right lists into neighbor_inds
      for (int i = 0; i < left_neighbors.size(); i++)
      {
        int neighbor_local_ind = left_neighbors[i];
        long this_neighbor_id = tree->inLeafData->mortons[neighbor_local_ind];
        if (!NodeOwnsPoint(node->rightNode, this_neighbor_id))
        {
          neighbor_inds.push_back(left_neighbors[i]);
        }
      }
      for (int i = 0; i < right_neighbors.size(); i++)
      {
        int neighbor_local_ind = left_neighbors[i];
        long this_neighbor_id = tree->inLeafData->mortons[neighbor_local_ind];
        if (!NodeOwnsPoint(node->leftNode, this_neighbor_id))
        {
          neighbor_inds.push_back(right_neighbors[i]);
        }    
      }
    
      ComputeSkeleton(node, neighbor_inds);
    } // are we low enough to build a skeleton for this node? 
    
  } // not a leaf
  
  return neighbor_inds;
  
} // UpwardPass



bool TreecodeDriver::CanPrune(fks_ompNode* node, QueryPoint& query)
{
  
  // If we didn't construct a skeleton at this level, don't try to prune
  if (node->level < min_skeleton_level)
    return false;
  
  // Determine if the node owns a neighbor
  for (int i = 0; i < query.nn_morton_ids.size(); i++)
  {
    if (!NodeOwnsPoint(node, query.nn_morton_ids[i]))
      return false;
  }  
  
  return true;
  
}


void TreecodeDriver::DownwardPass(fks_ompNode* node, QueryPoint& query) 
{

  int my_thread_id = omp_get_thread_num();

  if (CanPrune(node, query))
  {
  
    std::cout << "Doing prune\n";
  
    // Add the potential into the query
    int num_targets = 1;
    int num_sources = node->skeletons->numof_points;
    
    double* skel_coords = node->skeletons->X.data();
    
    double* kernel_mat = kernel_mat_space[my_thread_id];
    kernels[my_thread_id].Compute(query.coords.data(), num_targets, node->skeletons->X.data(), num_sources, dim, kernel_mat);

    double* charges = node->skeletons->charges.data();

    double potential = cblas_ddot(num_sources, kernel_mat, 1, charges, 1);
    
    std::cout << "Potential from prune: " << potential << "\n";
    
    // store the results, whereever they go
    query.potential += potential;
  
  }
  else if(node->leftNode == NULL)
  {
    
    std::cout << "Doing exact base case\n";
    
    // do the base case
    // compute the kernel for all of the points and the neighbors
    int num_targets = 1;
    
    fksData* leaf_data = thread_leaf_data[my_thread_id];
    bool got_data = tree->getLeafData(node, leaf_data);
    // TODO: fail better
    if (!got_data)
    {
      std::cout << "couldn't get leaf_data\n";
    }
    
    int num_sources = leaf_data->numof_points;
    
    std::cout << "Doing base case for " << num_sources << " sources\n";
    
    double* kernel_mat = kernel_mat_space[my_thread_id];
    kernels[my_thread_id].Compute(query.coords.data(), num_targets, leaf_data->X.data(), num_sources, dim, kernel_mat);

    double* charges = leaf_data->charges.data();

    double potential = cblas_ddot(num_sources, kernel_mat, 1, charges, 1);
    
    std::cout << "Potential from base case " << potential << "\n";
    
    // store the results
    query.potential += potential;
    
  }
  else {
    
    std::cout << "Recursing\n";
    
    DownwardPass(node->leftNode, query);
    DownwardPass(node->rightNode, query);

  }
  
} // Downward Pass


double TreecodeDriver::Compute(int query_lid) 
{
  
  std::vector<double> query_coords(dim);
  
  for (int i = 0; i < dim; i++)
  {
    query_coords[i] = tree->inLeafData->X[query_lid*dim + i];
  }
  
  std::vector<std::pair<double, long> > query_nn_list;
  query_nn_list.assign(tree->knnInfoForLeaf->begin()+query_lid, 
    tree->knnInfoForLeaf->begin() + query_lid + num_neighbors_per_point);

  std::vector<int> query_nn_lids(query_nn_list.size());
  
  for (int i = 0; i < query_nn_lids.size(); i++)
  {
    query_nn_lids[i] = tree->leafmap[query_nn_list[i].second];
    //std::cout << "query nn gid: " << query_nn_list[i].second << ", lid: " << query_nn_lids[i] << "\n"; 
  }
  
  QueryPoint query(query_coords, query_nn_lids);
  
  DownwardPass(tree->root_omp, query);
  
  return query.potential;
  
}

double TreecodeDriver::ComputeNaive(long query_id)
{
  
  int my_thread_id = omp_get_thread_num();
  
  std::vector<double> query_coords(dim);
  for (int i = 0; i < dim; i++)
  {
    query_coords[i] = tree->inLeafData->X[query_id*dim + i];
  }
  
  double* ref_coords = tree->inLeafData->X.data();
  double* charges = tree->inLeafData->charges.data();

  double* K = kernel_mat_space[my_thread_id];
  kernels[my_thread_id].Compute(query_coords.data(), 1, ref_coords, tree->inLeafData->numof_points, dim, K);
  
  /*
  double potential = 0.0;
  for (int i = 0; i < tree->inLeafData->numof_points; i++)
  {
    double this_result = K[i] * charges[i];
    
    if (isinf(this_result))
      std::cout << "bad result\n";
    
    potential += this_result;
  }
  */
  double potential = cblas_ddot(tree->inLeafData->numof_points, K, 1, charges, 1);
  
  //std::cout << "Exact potential: " << potential << "\n";
  
  return potential;
  
}

double TreecodeDriver::ComputeNN(int query_id)
{
  
  int my_thread_id = omp_get_thread_num();
  
  std::vector<double> query_coords(dim);
  for (int i = 0; i < dim; i++)
  {
    query_coords[i] = tree->inLeafData->X[query_id*dim + i];
  }
  
  std::vector<std::pair<double, long> > query_nn_gids;
  query_nn_gids.assign(tree->knnInfoForLeaf->begin()+query_id, 
    tree->knnInfoForLeaf->begin() + query_id + num_neighbors_per_point);

  std::vector<int> query_nn_lids(query_nn_gids.size());
  for (int i = 0; i < query_nn_lids.size(); i++)
  {
    query_nn_lids[i] = tree->leafmap[query_nn_gids[i].second];
  }
  
  std::vector<double> ref_coords(dim * query_nn_lids.size());
  std::vector<double> charges(query_nn_lids.size());

  for (int i = 0; i < query_nn_lids.size(); i++)
  {
    charges[i] = tree->inLeafData->charges[query_nn_lids[i]];
    for (int d = 0; d < dim; d++)
    {
      ref_coords[i*dim + d] = tree->inLeafData->X[query_nn_lids[i]*dim + d];
    }
  }
  
  double* K = kernel_mat_space[my_thread_id];
  kernels[my_thread_id].Compute(query_coords.data(), 1, ref_coords.data(), query_nn_lids.size(), dim, K);
  
  double potential = cblas_ddot(query_nn_lids.size(), K, 1, charges.data(), 1);
  
  return potential;
    
}


// TODO: assume that forbidden inds is already sorted? 
std::vector<int> TreecodeDriver::SampleFarField(int num_needed,
  std::vector<int>& forbidden_inds, fks_ompNode* node)
{

  int num_local_ids = tree->inLeafData->numof_points;
  // samples needs to contain num_needed indices from 1:num_local_ids that aren't contained in forbidden_inds
  std::vector<int> samples(num_needed * oversampling_factor);
  
  int num_avail = num_local_ids - forbidden_inds.size();
  if (num_avail < num_needed) 
  {
    // then, we fail, 
    // TODO: how to handle failure?
    std::cout << "Failure case not yet implemented!\n";
  }
  
  // TODO: should I assume that this is already sorted? 
  std::sort(forbidden_inds.begin(), forbidden_inds.end());
  
  // Generate the samples
  std::vector<int> oversamples(num_needed * oversampling_factor);
  for (int i = 0; i < oversamples.size(); i++)
  {
    oversamples[i] = rand() % num_local_ids;
    
    long sample_mid = tree->inLeafData->mortons[oversamples[i]];
    if (NodeOwnsPoint(node, sample_mid)) 
    {
      // special failure value
      oversamples[i] = INT_MAX;
    }
    
  }
  std::sort(oversamples.begin(), oversamples.end());
  
  // Call set_difference to eliminate forbidden indices from the samples
  std::vector<int>::iterator sample_end;
  sample_end = std::set_difference(oversamples.begin(), oversamples.end(), 
                forbidden_inds.begin(), forbidden_inds.end(), samples.begin());
  
  // Make sure we don't repeat a sample
  std::vector<int>::iterator final_it;
  // TODO: don't have to look at all of it?
  final_it = std::unique(samples.begin(), samples.end());
  
  // only return the number we need
  if (final_it - samples.begin() > num_needed && samples[num_needed - 1] < INT_MAX) {
    samples.resize(num_needed);
  }
  else {
    // we failed for some reason
    // TODO: handle this gracefully
    std::cout << "Uniform sampling failure\n";
  }
  
  return samples;
  
} // SampleFarField


bool TreecodeDriver::NodeOwnsPoint(fks_ompNode* node, long point_mid)
{
  
  int level = node->level;
  
  // this is built from the right, so the rightmost level bits are the ones we care about
  long node_id = node->node_morton;
  
  //std::cout << "Prune check: \n";
  //std::cout << "node id: " << std::hex << node_id << "\n";
  //std::cout << "point id: " << std::hex << point_mid << "\n";
  
  // Now, the question is: how is the point id built
  // how do I  know how far to shift it in order to compare?
  
  long mask = ~(1 << (level+1));

  bool res = ~(node_id ^ point_mid) & mask;
  
  return res;
  
}




