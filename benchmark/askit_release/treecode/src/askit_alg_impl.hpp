using namespace askit;

#include <stack>

#define _OUTPUT_INFO_ false
#define _DEBUG_UPDATE_CHARGES false
#define _DEBUG_INT_LISTS_ false
#define _DEBUG_SKELETON_SCAN_ false

template<class TKernel>
AskitAlg<TKernel>::AskitAlg(fksData* refData, KernelInputs& kernel_inputs, AskitInputs& askit_inputs)
  :
  kernel_params(kernel_inputs),
  num_skeleton_targets(askit_inputs.num_skel_targets),
  neighbors_to_pass_up(askit_inputs.neighbors_to_pass_up),
  id_rank(askit_inputs.rank),
  num_neighbors_per_point(askit_inputs.num_neighbors_in),
  do_split_k(askit_inputs.do_split_k),
  pruning_num_neighbors(askit_inputs.pruning_num_neighbors),
  min_skeleton_level(askit_inputs.min_skeleton_level),
  oversampling_factor(askit_inputs.oversampling_factor_in),
  num_uniform_required(askit_inputs.num_uniform_required),
  num_downward_kernel_evals(0),
  tree_build_time(0.0),
  skeletonization_time(0.0),
  list_blocking_time(0.0),
  test_list_blocking_time(0.0),
  evaluation_time(0.0),
  test_evaluation_time(0.0),
  exact_comp_time(0.0),
  let_traversal_time(0.0),
  update_charges_time(0.0),
  leaf_size(askit_inputs.max_points_per_node),
  compress_self_interactions(askit_inputs.compress_self),
  skeletonize_self_interactions(askit_inputs.skeletonize_self),
  use_adaptive_id(askit_inputs.use_adaptive_id),
  use_simplified_adaptive_id(false),
  do_absolute_id_cutoff(askit_inputs.do_absolute_id_cutoff),
  do_scale_near_adaptive(askit_inputs.do_scale_near_adaptive),
  do_adaptive_level_restriction(askit_inputs.do_adaptive_level_restriction),
  traverse_to_self_only(askit_inputs.traverse_to_self_only),
  do_absolute_num_targets(askit_inputs.do_absolute_num_targets),
  dont_store_proj(askit_inputs.dont_store_proj),
  id_tol(askit_inputs.id_tol),
  do_fmm(askit_inputs.do_fmm),
  near_scale(0.0),
  num_nodes_with_uniform(0),
  // extra timers
  construct_tree_list_time(0.0),
  compute_leaf_neighbors_time(0.0),
  merge_neighbor_lists_time(0.0),
  collect_local_samples_time(0.0),
  collect_neighbor_coords_time(0.0),
  distributed_upward_pass_time(0.0),
  uniform_sample_sibling_time(0.0),
  dist_compute_skeleton_time(0.0),
  merge_skeletons_of_kids_time(0.0),
  merge_nn_list_time(0.0),
  get_dist_node_excl_nn_time(0.0),
  compute_adaptive_id_time(0.0),
  apply_proj_time(0.0),
  subsample_self_targets_time(0.0),
  compute_skeleton_time(0.0),
  solve_for_proj_time(0.0),
  qr_time(0.0),
  max_qr_time(0.0),
  kernel_compute_time(0.0),
  exchange_let_time(0.0),
  dist_test_set_time(0.0),
  pass_potentials_down_time(0.0),
  leaf_list_size_before(0),
  leaf_list_size_after(0),
  merge_aggressive(askit_inputs.merge_aggressive),
  merge_basic_set_difference_time(0.0),
  merge_fmm_lists_aggressive_time(0.0),
  less_than_tree_order_time(0.0),
  is_ancestor_time(0.0),
  split_node_time(0.0),
  merge_tree_list_full_time(0.0),
  update_test_let_time(0.0),
  merge_fmm_lists_basic_time(0.0),
  fmm_add_to_map_time(0.0),
  compute_near_scale_time(0.0)
  // debug_potential_before(0.0),
  // debug_potential_after(0.0)
{
  
  MPI_Comm_rank(MPI_COMM_WORLD, &my_mpi_rank);
  //cout << "rank " << my_mpi_rank << " entering constructor.\n";   // BXC

  // seed the random number generator for sampling the far field
  srand(time(NULL));

  // check the adaptive flags
  if (askit_inputs.use_simplified_adaptive_id)
  {
    use_simplified_adaptive_id = true;
    use_adaptive_id = false;
  }
  
  if (askit_inputs.do_split_k)
  {
    // if we have the default, then we just go with k / 2
    if (pruning_num_neighbors == 0)
      pruning_num_neighbors = num_neighbors_per_point / 2;

    sampling_num_neighbors = num_neighbors_per_point - pruning_num_neighbors;

  }
  else 
  {
    pruning_num_neighbors = 0;
    sampling_num_neighbors = num_neighbors_per_point;
  }

  int num_threads = omp_get_max_threads();

  // cout << "Using up to " << num_threads << " threads\n";

  // allocate space to construct kernel submatrices in
  kernel_mat_space.resize(num_threads);
  thread_leaf_data.resize(num_threads);
  thread_leaf_nn_data.resize(num_threads);

  // We do the ID on either leaves, or the union of the children's skeletons, 
  // so this is the maximum number of columns we might need
  int max_id_cols = std::max(askit_inputs.max_points_per_node, 2 * id_rank);
  id_workspaces.resize(num_threads);

  
  // Set up the data for tree construction
  fksCtx *ctx = new fksCtx();
  ctx->k = num_neighbors_per_point;
  ctx->minCommSize = 1;
  //ctx->rkdt_niters = askit_inputs.num_neighbor_iterations;
  //ctx->rkdt_mppn = askit_inputs.rkdt_mppn;
  //ctx->rkdt_maxLevel = askit_inputs.rkdt_maxLevel;
  ctx->fks_mppn = askit_inputs.max_points_per_node;
  ctx->fks_maxLevel = askit_inputs.max_tree_level;
  // Checks the accuracy of knn search
  ctx->check_knn_accuracy = true;

  // Read the KNN info from a file
  vector< pair<double, long> > *kNN_rkdt = new vector< pair<double, long> >;

  if(askit_inputs.is_binary) {
    knn::binread_knn(askit_inputs.knn_info_file.c_str(), refData->gids,
      num_neighbors_per_point, kNN_rkdt);        
  }
  else {
    knn::dlmread_knn(askit_inputs.knn_info_file.c_str(), refData->gids,
      num_neighbors_per_point, kNN_rkdt);
  }

  if(_OUTPUT_INFO_) {
      MPI_Barrier(MPI_COMM_WORLD);
      if(my_mpi_rank == 0) {
        cout << "Rank " << my_mpi_rank << " finished reading neighbors.\n";
      }
  }

  double build_start = omp_get_wtime();

  tree = new fksTree;
  tree->build(refData, ctx);

  if(my_mpi_rank == 0) std::cout << "Rank: " << my_mpi_rank << ": Finished building tree.\n";
  
  // Tree building complete

  // I'm setting this here because once we start exchanging neighbors and
  // LET nodes, there will be extra points stored here
  N = tree->inProcData->numof_points;

  // make sure everyone knows the global N
  //MPI_Allreduce(&N, &global_N, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  global_N = tree->root_mpi->num_points_owned;

  if(my_mpi_rank == 0) std::cout << "Rank: " << my_mpi_rank << ": calling knn().\n\n";
  
  // cout << "Tree building only time: " << omp_get_wtime() - build_start << "\n";
  
  // Now, we can exchange the KNN info given the tree
  tree->knn(refData, kNN_rkdt);

  // set the number of levels above tree->root_omp
  num_global_levels = (tree->depth_mpi > 1) ? tree->depth_mpi - 1 : 0;

  // set local dimension parameter
  dim = tree->inProcData->dim;

  // set the size of the global tree
  MPI_Comm_size(tree->comm, &comm_size);

  MPI_Barrier( MPI_COMM_WORLD );

  tree_build_time = omp_get_wtime() - build_start;
  
  
  
  
  
  if (kernel_params.do_variable_bandwidth)
  {
    double h_scale = 0.5;
    // Now, we need to set up the h array in the kernel params
    kernel_params.variable_h.resize(tree->inProcData->numof_points, 10000.0);
    
    kernel_params.variable_h[0] = 0.01;
    
    // kernel_params.variable_h[0] = 100.0;
//
// #pragma omp parallel for
//     for (int i = 0; i < tree->inProcData->numof_points; i++)
//     {
//
//       // Just a placeholder for variable bandwidth kernel
//       // We don't know the nearest neighbor info for sources we received from
//       // another process
//       double norm_x_sqr = 0.0;
//       for (int j = 0; j < dim; j++)
//       {
//         // this is really inefficient, but we only do it once and don't
//         // count it for timing
//         norm_x_sqr = tree->inProcData->X[i*dim + j] * tree->inProcData->X[i*dim + j];
//       }
//
//       double h_i = kernel_params.bandwidth * (1 + 0.5/(1.0 + exp(-1.0 * norm_x_sqr)));
//       // I think these need to be interpreted as this
//       kernel_params.variable_h[i] = -1.0 / (2.0 * h_i * h_i);
//
//     } // loop over source points
    
  } // doing variable bandwidth
    
  kernels.resize(num_threads);

  // Pre-allocate everything
  for (int i = 0; i < num_threads; i++)
  {
    //kernel_mat_space[i].resize();
    thread_leaf_data[i] = new fksData();
    thread_leaf_nn_data[i] = new fksData();
    id_workspaces[i] = new IDWorkspace(id_rank, max_id_cols);
    kernels[i] = new TKernel(kernel_params);
  }
  
  
  
  // cout << "Tree building + KNN time: " << build_time << "\n";

  // Compute the nearest neighbors and construct the skeletons
  if(_OUTPUT_INFO_) {
    MPI_Barrier(MPI_COMM_WORLD);
    if(my_mpi_rank == 0) std::cout << "Rank: " << my_mpi_rank << ": Constructing skeletons.\n";
  }
  
  // We're still building the list representation of the local tree for the 
  // upward pass 
  // TODO: replace this with the traversal in fksTree
  max_tree_level = 0;

  double construct_tree_list_start = omp_get_wtime();

  // Construct neighbor lists: pair of local index and distance to neighbor
  ConstructTreeList(tree->root_omp, tree_list, max_tree_level);

  construct_tree_list_time = omp_get_wtime() - construct_tree_list_start;


  if (do_scale_near_adaptive)
  {
    ComputeNearScale();
  }

  double upward_start = omp_get_wtime();
  //time_t upward_start = time(NULL);
  //double upward_start = dsecnd();
  ParallelUpwardPass();
  //upward_pass_time = dsecnd() - upward_start;
  //upward_pass_time = difftime(time(NULL), upward_start);
  skeletonization_time = omp_get_wtime() - upward_start;

  // Can't be sure of the ordering because it's built in parallel
  sort(my_skeleton_frontier.begin(), my_skeleton_frontier.end(), askit::LessThanTreeOrder);
  
  
  // cout << "\n num global levels: " << num_global_levels << "\n";
  // cout << "\nskel frontier: \n";
  // for (int i = 0; i < my_skeleton_frontier.size(); i++)
  // {
  //   cout << "(" << my_skeleton_frontier[i].first << ", " << my_skeleton_frontier[i].second << "); ";
  // }
  // cout << "\n\n";


  if(_OUTPUT_INFO_) {
    MPI_Barrier(MPI_COMM_WORLD);
    if(my_mpi_rank == 0) std::cout << "Rank: " << my_mpi_rank << ": Constructing skeletons Done.\n";
  }

  // resize the interaction lists for later
  approx_target_to_node.resize(N);
  direct_target_to_node.resize(N);
  

  MPI_Barrier( MPI_COMM_WORLD );
  double let_start = omp_get_wtime();
  
  
  set< triple<long, long, int> > set_leaves;
  set< triple<long, long, int> > set_skeletons;
  int num_neighbors_for_let;
  if (traverse_to_self_only)
    num_neighbors_for_let = 1;
  else if(do_split_k)
    num_neighbors_for_let = pruning_num_neighbors;
  else
    num_neighbors_for_let = num_neighbors_per_point;
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  vector<vector<long> > direct_target_to_node_gid(N);
  vector<vector<long> > approx_target_to_node_gid(N);
  
  tree->LET(min_skeleton_level, set_leaves, set_skeletons, 
    my_skeleton_frontier, num_neighbors_for_let,
    direct_target_to_node_gid, approx_target_to_node_gid);

  MPI_Barrier(MPI_COMM_WORLD);
  
  frontier_exchange_time = tree->frontier_exchange_time;
  prune_by_knn_time = tree->prune_by_knn_time;
  
  // if (my_mpi_rank == 0)
  // {
  //   cout << "\n\nSkeleton frontier: \n";
  //   for (int i = 0; i < my_skeleton_frontier.size(); i++)
  //   {
  //     cout << "( " << my_skeleton_frontier[i].first << ", " << my_skeleton_frontier[i].second << "), ";
  //   }
  //   cout << "\n\n";
  // }  
  
  if(my_mpi_rank == 0) cout << "rank " << my_mpi_rank << " exchanging LET.\n";  
  
  if (comm_size > 1)
  {
    double exchange_start = omp_get_wtime();
    // this does the actual communication now
    tree->exchange_let(set_leaves, set_skeletons, skeleton_sizes, false);
    exchange_let_time = omp_get_wtime() - exchange_start;

  }  // more than one MPI rank
  else {
    // Need to set up letNodeMap, letNodeList
    tree->letNodeList = tree_list;
    
    // need to build letNodeMap as well for the single MPI rank case
    for (int i = 0; i < tree->letNodeList.size(); i++)
    {
      tree->letNodeMap.insert(make_pair((long) i, i));
      tree->letNodeList[i]->global_node_id = (long) i;
    }
    
  } // only one MPI rank
  
  if(_OUTPUT_INFO_) {
    MPI_Barrier( MPI_COMM_WORLD );
    if(my_mpi_rank == 0)
      std::cout << "Rank: " << my_mpi_rank << ": Finished Communicating LET.\n";
  }


  // cout << "\nTarget 0 direct interactions: \n";
  // for (int i = 0; i < direct_target_to_node_gid[0].size(); i++)
  // {
  //   fks_ompNode* node = tree->let_node(direct_target_to_node_gid[0][i]);
  //   cout << "(" << node->node_morton << ", " << node->level << "), ";
  // }
  // cout << "\nApprox interactions: \n";
  // for (int i = 0; i < approx_target_to_node_gid[0].size(); i++)
  // {
  //   fks_ompNode* node = tree->let_node(approx_target_to_node_gid[0][i]);
  //   cout << "(" << node->node_morton << ", " << node->level << "), ";
  // }
  // cout << "\n\n";



  // Now, we'll fill these in from the global id versions above
#pragma omp parallel for 
  for (long i = 0; i < N; i++)
  {
  
    direct_target_to_node[i].resize(direct_target_to_node_gid[i].size());
    approx_target_to_node[i].resize(approx_target_to_node_gid[i].size());
  
    // loop over node global ids, replace with index into letNodeList
    for (int j = 0; j < direct_target_to_node_gid[i].size(); j++)
    {
      map<long, int>::iterator map_it = tree->letNodeMap.find( direct_target_to_node_gid[i][j] );
      assert(map_it != tree->letNodeMap.end());
      int node_lid = map_it->second;
      direct_target_to_node[i][j] = node_lid;        
    }

    // do the same for the approx indices
    for (int j = 0; j < approx_target_to_node_gid[i].size(); j++)
    {
      map<long, int>::iterator map_it = tree->letNodeMap.find( approx_target_to_node_gid[i][j] );
      assert(map_it != tree->letNodeMap.end());
      int node_lid = map_it->second;
      approx_target_to_node[i][j] = node_lid;        
    }
  
  }  // loop over target points

  direct_node_to_target.resize(tree->letNodeList.size());
  approx_node_to_target.resize(tree->letNodeList.size());
  
  old_let_size = tree->letNodeList.size();
  old_inprocdata_size = tree->inProcData->numof_points;
  
  num_omp_leaves = (1 << (tree->depth_omp - 1));

  // finished with let construction
  let_traversal_time = omp_get_wtime() - let_start;
  
  
  ///////////////  Invert the interaction lists  //////////////////
  // This has to go after the LET construction
  double list_start = omp_get_wtime();

  // moved out here for code reuse of ComputeNodeInteractionList
  CreateSourceInteractionLists();

  vector<vector<int> > approx_node_to_target_potential;

  vector<vector<int> > fmm_source_node_to_target_node;

  if (do_fmm)
  {
    
    // Need skeleton_sizes & scan, approx_source_inds, and approx_charge_inds before calling this     
    ComputeFMMInteractionLists(approx_target_to_node, fmm_source_node_to_target_node);
  }
  
  // These have to be called after ComputeFMMInteractionLists()
  InvertInteractionList(approx_target_to_node, approx_node_to_target);
  InvertInteractionList(direct_target_to_node, direct_node_to_target);

  approx_node_to_target_potential = approx_node_to_target;

  if (do_fmm)
  {
    
    // Now we have target_nodes_for_source -- list of target nodes which 
    // interact node-to-node with each source
#pragma omp parallel for
    for (int i = 0; i < tree->letNodeList.size(); i++)
    {
      
      fks_ompNode* source_node = tree->letNodeList[i];
      
      vector<int>& target_nodes = fmm_source_node_to_target_node[i];
      
      for (int j = 0; j < target_nodes.size(); j++)
      {
        
        fks_ompNode* target_node = tree->letNodeList[target_nodes[j]];
       
        vector<int> target_skel_inds(target_node->skeletons->numof_points);
        vector<int> target_u_inds(target_node->skeletons->numof_points);

        for (int k = 0; k < target_u_inds.size(); k++)
        {
          target_skel_inds[k] = tree->pos(target_node->skeletons->gids[k]);
          target_u_inds[k] = N + skeleton_sizes_scan[target_nodes[j]] + k;
        }
        
        // insert the target skeleton indices
        approx_node_to_target[i].insert(approx_node_to_target[i].end(), target_skel_inds.begin(), target_skel_inds.end());
        
        // insert the potential indices
        approx_node_to_target_potential[i].insert(approx_node_to_target_potential[i].end(), target_u_inds.begin(), target_u_inds.end());

      } // loop over targets for this source
      
    } // loop over source nodes
    
  } // setting up fmm interaction lists

  // Now set up the tables and maps for the efficient inverted evaluation
  
  charge_table = tree->inProcData->charges;
  charge_table.insert(charge_table.end(), skeleton_charges.begin(), skeleton_charges.end());

  // set this for the test update step
  old_charge_table_size = charge_table.size();

  node_source_inds = direct_source_inds;
  node_source_inds.insert(node_source_inds.end(), approx_source_inds.begin(), approx_source_inds.end());

  training_target_inds = direct_node_to_target;
  training_target_inds.insert(training_target_inds.end(), approx_node_to_target.begin(), approx_node_to_target.end());

  charge_inds = direct_charge_inds;
  charge_inds.insert(charge_inds.end(), approx_charge_inds.begin(), approx_charge_inds.end());

  // I think this just gets ignored for the FMM version
  potential_map = direct_node_to_target;
  potential_map.insert(potential_map.end(), approx_node_to_target_potential.begin(), approx_node_to_target_potential.end());

  list_blocking_time = omp_get_wtime() - list_start;

  long num_kernel_evals = 0;
  
  // cout << "\nPrinting all interaction counts: \n";
#pragma omp parallel for reduction(+:num_kernel_evals)
  for (int i = 0; i < training_target_inds.size(); i++)
  {
    // cout << "(" << training_target_inds[i].size() << " x " << node_source_inds[i].size() << ")\n";
    num_kernel_evals += training_target_inds[i].size() * node_source_inds[i].size();
  }
  
  if (my_mpi_rank == 0)
    cout << "\nTotal kernel evaluations: " << num_kernel_evals << "\n\n";

  num_downward_kernel_evals = num_kernel_evals;


  // Compute charge norms

  double local_charge_l1_norm = 0.0;
#pragma omp parallel for reduction(+:local_charge_l1_norm)
  for (int i = 0; i < N; i++)
  {
    local_charge_l1_norm += fabs(tree->inProcData->charges[i]);
  }
  // need a local variable for the OpenMP reduction
  charge_l1_norm = local_charge_l1_norm;


  double local_charge_l2_norm = 0.0;
#pragma omp parallel for reduction(+:local_charge_l2_norm)
  for (int i = 0; i < N; i++)
  {
    local_charge_l2_norm += tree->inProcData->charges[i] * tree->inProcData->charges[i];
  }
  // need a local variable for the OpenMP reduction
  charge_l2_norm_sqr = local_charge_l2_norm;


  // for the adaptive rank, need to set the tolerance for each skeletonization
  // max_tree_level is set in the upward pass
  if (use_adaptive_id)
  {
    //id_tol = id_tol / log2((double)max_tree_level);
    if(_OUTPUT_INFO_) cout << "input ID tolerance: " << id_tol << ", N: " << N << ", leaf_size: " << leaf_size << "\n";
    id_tol = id_tol / log2((double)N / (double)leaf_size);
    if(_OUTPUT_INFO_) cout << "using effective id tolerance: " << id_tol << "\n";
  }
  
    
  if (_DEBUG_UPDATE_CHARGES && my_mpi_rank == 0)
  {
    
    cout << "\n\nskeleton set: \n";
    
    set<triple<long,long,int> >::iterator it;
    
    for (it = tree->my_set_skeletons.begin(); it != tree->my_set_skeletons.end(); it++)
    {
      fks_ompNode* node = tree->let_node(it->first);
      cout << "(" << (*it).first <<", " << it->second << ", " << it->third << ", " << node->skeletons->charges[0] << "), ";
    }
    cout << "\n\n";
      
  }
    
    
    
  delete ctx;

  if(_OUTPUT_INFO_) {
    MPI_Barrier( MPI_COMM_WORLD );
    if(my_mpi_rank == 0)
      std::cout << "Rank: " << my_mpi_rank << " finished with upward pass\n";
  }

} // constructor()



template<class TKernel>
void AskitAlg<TKernel>::UpdateCharges(std::vector<double>& new_charges)
{
  // we assume that 'new_charges' contains the tree->numof_points_of_dist_leaf
  // charges that are corresponding to the points after build the tree
  // so we first should shuffle them back to the origianl rank,
  // then we can all tree->exchange_charges() -- Bo

  int rank, size;
  MPI_Comm_rank(tree->comm, &rank);
  MPI_Comm_size(tree->comm, &size);
  
  if (dont_store_proj)
  {
    cout << "ERROR: calling UpdateCharges without storing Proj matrix is not supported!!!\n";
  }
  
  double update_charges_start = omp_get_wtime();

 
  // if (rank == 0)
//   {
//
//     cout << "charges before \n";
//
//     fks_mpiNode *curr_mpi = tree->root_mpi;
//     while(curr_mpi->fks_kid != NULL)
//       curr_mpi = curr_mpi->fks_kid;
//
//     fksData* skel = curr_mpi->skeletons;
//
//     for (int i = 0; i < skel->numof_points; i++)
//     {
//       cout << skel->charges[i] << ", ";
//     }
//     cout << "\n\n";
//   }
  MPI_Barrier(tree->comm);
  //if(rank == 0) cout << "rank " << rank << " exchanging updated charges.\n";
  if(size == 1) {
    assert(tree->inProcData->numof_points == new_charges.size());
    #pragma omp parallel if(new_charges.size() > 1000)
    {
      #pragma omp for
      for(int i = 0; i < tree->inProcData->numof_points; i++)
        tree->inProcData->charges[i] = new_charges[i];
    }
  } 
  else {
    assert( new_charges.size() == tree->numof_points_of_dist_leaf );
    int divd = tree->glbN / size;
    int rem = tree->glbN % size;
    int numof_original_points = rank < rem ? (divd+1) : divd;
    if (_DEBUG_UPDATE_CHARGES)
      cout << "rank " << rank << " allocating new space for charges\n";
    double *original_charges = new double [numof_original_points];
    if (_DEBUG_UPDATE_CHARGES)
      cout << "rank " << rank << " calling shuffle back\n";
    tree->shuffle_back(new_charges.size(), new_charges.data(), tree->inProcData->gids.data(), numof_original_points, original_charges, tree->comm);
    if (_DEBUG_UPDATE_CHARGES)
      cout << "rank " << rank << " calling exchange charges, before: " << tree->inProcData->charges.size() << "\n";
    tree->exchange_charges(numof_original_points, original_charges,
          tree->inProcData->numof_points, tree->inProcData->gids.data(),
          tree->inProcData->charges.data(), tree->comm);
    if (_DEBUG_UPDATE_CHARGES)
      cout << "rank " << rank << " finished exchange charges, after: " << tree->inProcData->charges.size() << "\n";
    delete [] original_charges;
  }

  if (_DEBUG_UPDATE_CHARGES)
    cout << "Rank " << rank << " calling UpdateSkeletonCharges, num_charges before: " << tree->inProcData->charges.size() << "\n";
  
  UpdateSkeletonCharges();
  
  if (_DEBUG_UPDATE_CHARGES)
    cout << "Rank " << rank << " finished UpdateSkeletonCharges, num_charges after: " << tree->inProcData->charges.size() << "\n";

  // need to exchange new skeleton charges in let
  if(size > 1) {
    if (_DEBUG_UPDATE_CHARGES)
      cout << "Rank " << rank << " exchanging updated LET, num_charges before: " << tree->inProcData->charges.size() << "\n";
    
    // tree->exchange_let(empty_set_leaves, tree->my_set_skeletons, skeleton_sizes, true);
    tree->exchange_updated_let(tree->my_set_skeletons, skeleton_sizes, _DEBUG_UPDATE_CHARGES);
    
    
    if (_DEBUG_UPDATE_CHARGES && my_mpi_rank == 0)
    {
    
      cout << "\n\nskeleton set: \n";
    
      set<triple<long,long,int> >::iterator it;
    
      for (it = tree->my_set_skeletons.begin(); it != tree->my_set_skeletons.end(); it++)
      {
        fks_ompNode* node = tree->let_node(it->first);
        cout << "(" << (*it).first <<", " << it->second << ", " << it->third << ", " << node->skeletons->charges[0] << "), ";
      }
      cout << "\n\n";
      
    }
    
    
    if (_DEBUG_UPDATE_CHARGES)
      cout << "Rank " << rank << " finished exchanging updated LET, num_charges after: " << tree->inProcData->charges.size() << "\n";
  }

  MPI_Barrier(tree->comm);
  //if(rank == 0) cout << "Rank " << rank << " updating charge table\n";
  // now, update the table for the efficient evaluation algorithm
  UpdateChargeTable();

  // update the timer
  update_charges_time += omp_get_wtime() - update_charges_start;


  // Update the L1 norm of the charges for error estimation
  double local_charge_l1_norm = 0.0;
#pragma omp parallel for reduction(+:local_charge_l1_norm)
  for (int i = 0; i < N; i++)
  {
    local_charge_l1_norm += fabs(tree->inProcData->charges[i]);
  }
  // need a local for the OpenMP reduction
  charge_l1_norm = local_charge_l1_norm;

  // Update the L1 norm of the charges for error estimation
  double local_charge_l2_norm = 0.0;
#pragma omp parallel for reduction(+:local_charge_l2_norm)
  for (int i = 0; i < N; i++)
  {
    local_charge_l2_norm += tree->inProcData->charges[i] * tree->inProcData->charges[i];
  }
  // need a local for the OpenMP reduction
  charge_l2_norm_sqr = local_charge_l2_norm;

} // UpdateCharges

// Given new original charges (on all of the points), we update the charges in 
// the skeletons
template<class TKernel>
void AskitAlg<TKernel>::UpdateSkeletonCharges()
{

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);


  if (_DEBUG_SKELETON_SCAN_)
  {
    cout << "rank " << my_mpi_rank << ": Beginning of UpdateSkeletonCharges check.\n";

    for (int i = 0; i < tree->letNodeList.size(); i++)
    {
    
      fks_ompNode* node = tree->letNodeList[i];
    
      if (node && node->skeletons)
      {
      
        if (skeleton_sizes[i] != node->skeletons->numof_points)
        {
          cout << "Rank " << my_mpi_rank << ": failing on node " << i << ". size: " << skeleton_sizes[i] << ", numof_points: " << node->skeletons->numof_points << "\n";
          assert(skeleton_sizes[i] == node->skeletons->numof_points);
        }
      } // does it have a skeleton?
    
    } // loop over nodes
  
  } // if debug skeleton scan


  for (int level_ind = max_tree_level;
       level_ind >= min_skeleton_level - num_global_levels && level_ind > 0;
       level_ind--)
  {
  
    std::vector<fks_ompNode*>::iterator tree_level = tree_list.begin() + (1 << level_ind) - 1;

    int num_nodes = 1 << level_ind;

#pragma omp parallel for
    for (int node_ind = 0; node_ind < num_nodes; node_ind++)
    {

      //std::cout << "doing node " << node_ind << "\n";

      fks_ompNode* node = *(tree_level + node_ind);
    
      // In the adaptive rank case, we'll skip any node that we couldn't 
      // skeletonize
      if (node->skeletons->cant_prune)
        continue;
    
      // collect the children's charges (or the inputs if a leaf)
      vector<double> charges_in;
      if (node->leftNode == NULL)
      {
        
        int num_points = node->leaf_point_gids.size();
        charges_in.resize(num_points);
        
        for (int i = 0; i < num_points; i++)
        {
          int lid = tree->pos(node->leaf_point_gids[i]);
          charges_in[i] = tree->inProcData->charges[lid];
        }
          
      }
      else {
  
        int num_points = node->leftNode->skeletons->numof_points;
        num_points += node->rightNode->skeletons->numof_points;
        
        charges_in = node->leftNode->skeletons->charges;
        charges_in.insert(charges_in.end(), node->rightNode->skeletons->charges.begin(),
            node->rightNode->skeletons->charges.end());
        
      }
    
      ApplyProj(charges_in, node->skeletons->charges, node->skeletons->proj, 
          node->skeletons->numof_points, node->skeletons->skeleton_perm);
    
      // now, I need to update the LET list
      if (comm_size > 1)
      {
        
        long node_gid = tree->morton_to_gid(node->node_morton, node->level);
        fks_ompNode* let_node = tree->let_node(node_gid);
        
        // If these don't match, then something is wrong
        assert(let_node->skeletons->numof_points == node->skeletons->numof_points);
        assert(let_node->skeletons->numof_points == skeleton_sizes[let_node->lnid]);
        
        let_node->skeletons->charges = node->skeletons->charges;

      }

    } // loop over nodes in level
  
  } // loop over levels of local tree
  
  if (_DEBUG_UPDATE_CHARGES)
    cout << "Rank " << my_mpi_rank << " finished with update charges in local tree\n";
  
  // update omp_root (if needed)
  
  // now, update for the distributed tree
  // Now, pass up the MPI tree
  fks_mpiNode *curr_mpi = tree->root_mpi;
  while(curr_mpi->fks_kid != NULL)
    curr_mpi = curr_mpi->fks_kid;
  
  // if (_DEBUG_UPDATE_CHARGES)
    // cout << "Rank " << my_mpi_rank << " curr_mpi skel: " << curr_mpi->skeletons << ", omp_root skel " << tree->root_omp->skeletons << "\n";
  
  if (curr_mpi->level >= min_skeleton_level)
  {
    
    if (curr_mpi->skeletons && !curr_mpi->skeletons->cant_prune) {
      
      fks_ompNode* omp_root = tree->root_omp;
    
      vector<double> charges_in = omp_root->leftNode->skeletons->charges;
      charges_in.insert(charges_in.end(), omp_root->rightNode->skeletons->charges.begin(),
        omp_root->rightNode->skeletons->charges.end());
    
      // cout << "Rank " << rank << " doing ApplyProj for omp root.\n";
  //     cout << "Rank " << rank << " omp_root skeleton " << omp_root->skeletons << "\n";
  //     cout << "Rank " << rank << " curr mpi skeleton " << curr_mpi->skeletons << "\n";
  //     cout << "Rank " << rank << " compressing " << charges_in.size() << " to rank " << curr_mpi->skeletons->numof_points << " with proj size: " << curr_mpi->skeletons->proj.size() << " and perm size " << curr_mpi->skeletons->skeleton_perm.size() << "\n";
  //
      // do Apply proj for omp_root
      ApplyProj(charges_in, curr_mpi->skeletons->charges, curr_mpi->skeletons->proj,
        curr_mpi->skeletons->numof_points, curr_mpi->skeletons->skeleton_perm);

      // now, I need to update the LET list
      if (comm_size > 1)
      {
      
        long node_gid = tree->morton_to_gid(curr_mpi->node_morton, curr_mpi->level);
        fks_ompNode* let_node = tree->let_node(node_gid);
      
        assert(let_node->skeletons->numof_points == curr_mpi->skeletons->numof_points);
      
        let_node->skeletons->charges = curr_mpi->skeletons->charges;
      
      } // need to copy into the LET node in the distributed case

    } // can we prune this node at all? 
    
    // the very bottom is omp_root, which we already took care of
    curr_mpi = curr_mpi->fks_parent;
  
  } // is root_omp low enough to be skeletonized

  if (_DEBUG_UPDATE_CHARGES)
    cout << "Rank " << my_mpi_rank << " finished with update charges in omp root\n";
  
  while(curr_mpi->level >= min_skeleton_level)
  {

    int local_rank;
    MPI_Comm_rank(curr_mpi->comm, &local_rank);
    
    // TODO: don't need this here, just need to get the charges
    // I think this function can alter some things inside inProcData
    fksData merged_skeleton;
    // This function checks the cant_prune flag, so we always call it for all 
    // processes -- this prevents deadlocking when only some processes call the 
    // function 
    tree->mergeSkeletonsOfKids(curr_mpi, &merged_skeleton);

    if (local_rank == 0 && curr_mpi->skeletons && !curr_mpi->skeletons->cant_prune)
    {
      
      // cout << "Rank " << rank << " doing ApplyProj for node level " << curr_mpi->level << ".\n";
//       cout << "Rank " << rank << " compressing " << merged_skeleton.charges.size() << " to rank " << curr_mpi->skeletons->numof_points << " with proj size: " << curr_mpi->skeletons->proj.size() << " and perm size " << curr_mpi->skeletons->skeleton_perm.size() << "\n";

      ApplyProj(merged_skeleton.charges, curr_mpi->skeletons->charges, curr_mpi->skeletons->proj,
        curr_mpi->skeletons->numof_points, curr_mpi->skeletons->skeleton_perm);
    
      // now, I need to update the LET list
      if (comm_size > 1)
      {
    
        long node_gid = tree->morton_to_gid(curr_mpi->node_morton, curr_mpi->level);
        fks_ompNode* let_node = tree->let_node(node_gid);
    
        assert(let_node->skeletons->numof_points == curr_mpi->skeletons->numof_points);
    
        let_node->skeletons->charges = curr_mpi->skeletons->charges;
    
        if (_DEBUG_UPDATE_CHARGES && my_mpi_rank == 0)
        {
          cout << "\n\n Updating LET charges in UpdateSkeletonCharges: ";
          cout << let_node->skeletons->charges[0] << "\n";
        }
    
      }
      
    }
      
    curr_mpi = curr_mpi->fks_parent;

  } // loop over the part of the tree for which we have skeletons

  if (_DEBUG_SKELETON_SCAN_)
  {
    cout << "rank " << my_mpi_rank << ": End of UpdateSkeletonCharges check.\n";

    for (int i = 0; i < tree->letNodeList.size(); i++)
    {
    
      fks_ompNode* node = tree->letNodeList[i];
    
      if (node && node->skeletons)
      {
      
        if (skeleton_sizes[i] != node->skeletons->numof_points)
        {
          cout << "Rank " << my_mpi_rank << ": failing on node " << i << ". size: " << skeleton_sizes[i] << ", numof_points: " << node->skeletons->numof_points << "\n";
          assert(skeleton_sizes[i] == node->skeletons->numof_points);
        }
      } // does it have a skeleton?
    
    } // loop over nodes
  
  } // if debug skeleton scan
  

} // UpdateSkeletonCharges


template<class TKernel>
AskitAlg<TKernel>::~AskitAlg()
{

  // Free the tree
  // This is causing a weird MPI crash, add it back later
  if (tree && 0)
  {
    delete tree;
    tree = NULL;
  }

  int num_threads = omp_get_max_threads();

  // Free all of our preallocated space
  for (int i = 0; i < num_threads; i++)
  {
    // changed to vector
    //delete[] kernel_mat_space[i];
    delete thread_leaf_data[i];
    delete thread_leaf_nn_data[i];
    delete id_workspaces[i];
    delete kernels[i];
  }

  if (tree != NULL)
  {
    delete tree;
    tree = NULL;
  }



}


// Returns vector of sampled coordinates (uniform_sampling)
template<class TKernel>
std::vector<double> AskitAlg<TKernel>::CollectLocalSamples(fks_ompNode* node, 
  std::vector<std::pair<long, double> >& neighbor_inds, int num_points_needed)
{

  // collect the indices to be passed to SampleFarField
  std::vector<int> forbidden_inds;
  for (int i = 0; i < neighbor_inds.size(); i++)
  {
    // these are the forbidden local ids 
    int neighbor_lid = tree->pos(neighbor_inds[i].first);
    forbidden_inds.push_back(neighbor_lid);
    
  } // loop to check which ones are owned by this MPI rank

  // samples are local ids    
  std::vector<int> uniform_inds = SampleFarField(num_points_needed, forbidden_inds, node);
  
  // Copy the coorindates into the vector
  std::vector<double> uniform_samples;
  uniform_samples.reserve(uniform_inds.size() * dim);
  
  for (int i = 0; i < uniform_inds.size(); i++)
  {
    uniform_samples.insert(uniform_samples.end(), 
      tree->inProcData->X.begin() + uniform_inds[i]*dim,
      tree->inProcData->X.begin() + (uniform_inds[i]+1)*dim);
  } // loop over sampled indices
  
  return uniform_samples;
  
} // CollectLocalSamples


template<class TKernel>
std::vector<int> AskitAlg<TKernel>::SampleFarField(int num_needed,
  std::vector<int>& forbidden_inds, fks_ompNode* node)
{

  // What range should we sample indices from?
  int num_local_ids = N;
  // samples needs to contain num_needed indices from 1:num_local_ids that aren't contained in forbidden_inds
  std::vector<int> samples;
 
  // Is this even possible?
  int num_avail = num_local_ids - forbidden_inds.size();
  if (num_avail < num_needed && num_needed > 0)
  {
    cout << "avail: " << num_avail << ", needed: " << num_needed << "\n";
    // then, we fail,
    // TODO: how to handle failure?
    std::cout << "Sample failure case not yet implemented!\n";
  }
 
  // sort the forbidden inds for failure check
  std::sort(forbidden_inds.begin(), forbidden_inds.end());
 
  // Generate the samples
  for (int i = 0; i < num_needed * oversampling_factor; i++)
  {
    int sample = rand() % num_local_ids;
   
    // Check if the sample is owned by the node, and discard if so
    long sample_mid = tree->inProcData->mortons[sample];

    // do need to update the node here, because the levels are not updated
    // yet by LET
    if (!tree->belong2Node(node->node_morton, node->level + num_global_levels, sample_mid))
    {
      samples.push_back(sample);
    }
   
  } // generating samples

  // TODO: check for failure here
  std::sort(samples.begin(), samples.end());

  // Call set_difference to eliminate forbidden indices from the samples
  std::vector<int>::iterator sample_end;
  std::vector<int> samples_out(samples.size());
  sample_end = std::set_difference(samples.begin(), samples.end(),
                forbidden_inds.begin(), forbidden_inds.end(), samples_out.begin());
 
  // Make sure we don't repeat a sample
  std::vector<int>::iterator final_it;
  final_it = std::unique(samples_out.begin(), sample_end);
 
  // only return the number we need
  if (final_it - samples_out.begin() >= num_needed) {
    samples_out.resize(num_needed);
  }
  else {
    // we failed for some reason
    // TODO: handle this gracefully
    cout << "Sampled: " << final_it - samples_out.begin() << ", Num needed: " << num_needed << "\n";
    std::cerr << "Uniform sampling failure\n";
  }
 
  return samples_out;
 
} // SampleFarField


// Just prints the skeleton for debugging purposes
template<class TKernel>
void AskitAlg<TKernel>::PrintSkeleton(fksData* skel)
{
 
  if (skel) {
    for (int i = 0; i < skel->numof_points; i++)
    {
      std::cout << "Point: " << skel->gids[i] << ":";
      std::cout << " charge: " << skel->charges[i] << ". (";
      int lid = tree->pos(skel->gids[i]);
      for (int d = 0; d < dim; d++)
      {
        cout << tree->inProcData->X[d + lid*dim] << ", ";
      }
      cout << ")\n";
    }
  }
  else {
    std::cout << "No skeleton.\n";
  }
  cout << "\n";
}


// Applies the permuation and proj to charges to get effective charges
template<class TKernel>
void AskitAlg<TKernel>::ApplyProj(vector<double>& charges_in, vector<double>& charges_out,
  vector<double>& proj, int rank, vector<lapack_int>& perm)
{
  
  // int mpi_rank;
  // MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  
  int num_sources = charges_in.size();

  // check if there is compression before we try to do the matvec
  if (num_sources > rank) {

    charges_out.resize(rank);
  
    vector<double> permuted_charges(num_sources);
    // apply the permutation to the charges
    for (int i = 0; i < num_sources; i++)
    {
      permuted_charges[i] = charges_in[perm[i]];
    } // loop over charges

    // Apply the proj matrix to the charges of the non-skeleton points
    // i.e. the ones in permuted_charges+rank:end
    int ns_m_id = num_sources - rank;
    double oned = 1.0;
    double zerod = 0.0;
    int onei = 1;
    cblas_dgemv("N", &rank, &ns_m_id, &oned, 
      proj.data(), &rank, permuted_charges.data() + rank, &onei, &zerod, charges_out.data(), &onei);

    // Add in the contribution due to the skeleton points themselves
    for (int i = 0; i < rank; i++)
    {
      charges_out[i] += permuted_charges[i];
    }
  
  }
  else {
    // if no compression, then the charges are unchanged
    charges_out = charges_in;
  }
  
}



// Returns a bool -- true if the skeletonization was successful, false otherwise
template<class TKernel>
void AskitAlg<TKernel>::ComputeSkeleton(fksData* my_skeleton, long m,
  std::vector<double>& source_vec, std::vector<double>& charge_vec,
  std::vector<long>& source_inds, 
  vector<double>& near_vec, vector<double>& unif_vec, bool printout)
{
  
  int num_sources = source_vec.size() / dim;
  
  int num_near = near_vec.size() / dim;
  int num_unif = unif_vec.size() / dim;

  int num_targets = num_near + num_unif;
   
  int my_thread_id = omp_get_thread_num();

  vector<int> source_local_ids(num_sources);
  for (int i = 0; i < num_sources; i++)
    source_local_ids[i] = tree->pos(source_inds[i]);

  // if (printout)
  // {
  //   cout << "num sources: " << num_sources << ", num_near: " << num_near << ", num_unif: " << num_unif << "\n";
  // }

  if (num_unif > 0)
  {
#pragma omp atomic
    num_nodes_with_uniform++;
  }

  // Compute the ID and store the skeleton
  //std::vector<lapack_int> skeleton_inds;
  
  int rank = 0;
  
  // is it even possible to compress here?
  if (use_adaptive_id || use_simplified_adaptive_id || num_sources > id_rank) 
  {

    // if (printout)
    //   cout << "computing " << num_targets << " x " << num_sources << " id.\n";
  
    double this_solve_time = 0.0;
    double this_qr_time = 0.0;
      
    if (use_adaptive_id) {

      // this will now force compression if the adaptive version fails
      rank = compute_adaptive_id(*(kernels[my_thread_id]), source_vec, num_sources, 
        near_vec, num_near, unif_vec, num_unif, dim, global_N, m, my_skeleton->skeleton_perm, my_skeleton->proj, 
        *(id_workspaces[my_thread_id]), id_tol, id_rank, printout, source_local_ids);

    }
    else // not doing the complicated version
    {

      // Compute the kernel matrix between sources and targets
      double kernel_compute_start = omp_get_wtime();
      
      vector<double> target_vec = near_vec;
      target_vec.insert(target_vec.end(), unif_vec.begin(), unif_vec.end());
      std::vector<double>& K = kernel_mat_space[my_thread_id];
      kernels[my_thread_id]->Compute(target_vec.begin(), target_vec.end(), 
          source_vec.begin(), source_vec.end(), dim, K, source_local_ids);

#pragma omp atomic
      kernel_compute_time += omp_get_wtime() - kernel_compute_start;

      if (do_scale_near_adaptive && use_simplified_adaptive_id) {
        
        double compute_adaptive_id_start = omp_get_wtime();

        // TODO: should I use the actual leaf size here? 
        rank = compute_adaptive_id_scale_near(K.data(), num_targets, num_sources, my_skeleton->skeleton_perm, 
            my_skeleton->proj, *(id_workspaces[my_thread_id]), id_tol, id_rank, printout, 
            leaf_size, m, near_scale, this_solve_time, this_qr_time);
        
#pragma omp atomic 
        compute_adaptive_id_time += omp_get_wtime() - compute_adaptive_id_start;
        
      }
      else if (use_simplified_adaptive_id) {

        double compute_adaptive_id_start = omp_get_wtime();

        // scale by the number of columns we're not considering
        double absolute_scale = sqrt((double)m / (double)num_sources );
        // scale by the number of rows we left out as well
        absolute_scale *= sqrt((double)(global_N - m) / (double)num_targets );

        rank = compute_adaptive_id_simplified(K.data(), num_targets, num_sources, my_skeleton->skeleton_perm, 
            my_skeleton->proj, *(id_workspaces[my_thread_id]), id_tol, id_rank, do_absolute_id_cutoff, absolute_scale,
            printout, this_solve_time, this_qr_time);
        
#pragma omp atomic 
        compute_adaptive_id_time += omp_get_wtime() - compute_adaptive_id_start;

      }
      else {

        rank = compute_id(K.data(), num_targets, num_sources, id_rank, my_skeleton->skeleton_perm, my_skeleton->proj, 
          *(id_workspaces[my_thread_id]), this_solve_time, this_qr_time);

      }
      
      // If we're doing adaptive level restriction and we didn't compress 
      // down to fewer than s_max columns, then we failed to skeletonize
      if (do_adaptive_level_restriction && use_simplified_adaptive_id && rank >= id_rank)
      {
        
        rank = 0;
        my_skeleton->cant_prune = true;
        my_skeleton->numof_points = 0;
        
      } // adaptive level restriction
      

    } // not doing standard adaptive ID
  
#pragma omp atomic
    solve_for_proj_time += this_solve_time;
        
#pragma omp atomic
    qr_time += this_qr_time;
        
#pragma omp atomic write
    max_qr_time = max(max_qr_time, this_qr_time);
        
  
  
    // if (printout)
    //   cout << "Found rank " << rank << " ID.\n";
  
    // We check if the adaptive rank finding failed to compress here
    if (do_adaptive_level_restriction && rank == 0)
    {
      // we don't need to do anything in this case
    }
    else if (rank < num_sources && rank > 0) 
    {
      
      // if (printout)
//  cout << "computing effective charges\n";
      
      double apply_proj_start = omp_get_wtime();
      
      ApplyProj(charge_vec, my_skeleton->charges, my_skeleton->proj, rank, 
        my_skeleton->skeleton_perm);

#pragma omp atomic
      apply_proj_time += omp_get_wtime() - apply_proj_start;
  
      my_skeleton->gids.resize(rank);
      for (int i = 0; i < rank; i++)
      {
        my_skeleton->gids[i] = source_inds[my_skeleton->skeleton_perm[i]];
      }

      my_skeleton->numof_points = rank;
      my_skeleton->dim = dim;
    
    }
    else {
      
      //std::cout << "failed to compress: rank: " << rank << ", sources: " << num_sources << "\n";
      
      // we weren't able to compress
      my_skeleton->charges = charge_vec;
      my_skeleton->gids = source_inds;
      my_skeleton->numof_points = num_sources;
      my_skeleton->dim = dim;      
    
      // if we weren't able to compress, then rank will be negative
      // in this case, we need to mark the failure
      if (rank < 0)
      {
        // this case shouldn't happen any more -- we switch to the fixed rank
        // algorithm if it does
        std::cout << "Failure to skeletonize -- max rank exceeded.\n";
      }
    
    }
    
  } // trying to compress
  else {
    
    // there are too few sources, so make the skeleton consist of 
    // everything
    
    my_skeleton->charges = charge_vec;
    my_skeleton->numof_points = num_sources;
    my_skeleton->gids = source_inds;
    my_skeleton->dim = dim;
    // we need to set this because it will be used to compute the local ids
    // later for the FMM version
    my_skeleton->skeleton_perm.resize(num_sources);
    for (int i = 0; i < num_sources; i++)
    {
      my_skeleton->skeleton_perm[i] = i;
    }
    
  } // can't compress -- too few sources
  
  // if (printout)
  //   cout << "Computed rank " << my_skeleton->numof_points << " skeleton.\n\n";
  
  // free up the memory if we're not storing skeletons
  if (dont_store_proj)
  {
    my_skeleton->proj.clear();
  }
  
} // ComputeSkeleton



// If node is a leaf, neighbor_inds is already filled out
// Otherwise, this function will fill it out
template<class TKernel>
void AskitAlg<TKernel>::ComputeSkeleton(fks_ompNode* node, 
  std::vector<std::pair<long, double> >& neighbor_inds,
  std::vector<double>& uniform_samples)
{
 
  int my_thread_id = omp_get_thread_num();

  int my_rank;
  MPI_Comm_rank(tree->comm, &my_rank);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  bool printout;
  // if (_OUTPUT_INFO_)
  //   printout = (node->node_morton == 0);
  // else
  //   printout = false;

  // printout = (node->node_morton == 0);
  printout = false;
  
  // Initialize the skeleton data structure -- the tree doesn't do this
  fksData* my_skeleton = node->skeletons;
  if (my_skeleton == NULL)
  {
    node->skeletons = new fksData();
    my_skeleton = node->skeletons;
  }
   
  // We'll collect sources and targets differently depending on whether it is a
  // leaf or not
  int num_sources;
  int num_targets;
  int num_neighbors;
 
  // coordinates of sources and targets
  std::vector<double> source_vec;
  std::vector<double> target_vec;
 
  std::vector<long> source_inds;

  // coorindates of charges
  std::vector<double> charge_vec;
 
  // are we a leaf?
  if (node->leftNode == NULL)
  {
    
    //cout << "getting leaf data\n";

    // Get the data owned by this leaf
    fksData* leaf_data = thread_leaf_data[my_thread_id];
    //std::cout << "Node " << node->lnid << " getting data\n";
    tree->getLeafData(node, leaf_data);
    //std::cout << "Node " << node->lnid << " finished getting data\n";
   
    source_inds = leaf_data->gids;
   
    num_sources = leaf_data->numof_points;
    
    source_vec = leaf_data->X;
    // get the charges
    charge_vec = leaf_data->charges;

  }// are we a leaf?
  else {
    // we're an internal node
   
    
    fks_ompNode* left_child = node->leftNode;
    fks_ompNode* right_child = node->rightNode;
   
    fksData* left_skel = left_child->skeletons;
    fksData* right_skel = right_child->skeletons;
   
   
    // If we're doing adaptive level restriction and we couldn't compress one of
    // the children, then we can't compress this node either
    if (do_adaptive_level_restriction && (left_skel->cant_prune || right_skel->cant_prune))
    {
      my_skeleton->cant_prune = true;
      my_skeleton->numof_points = 0;
      
      // we need to check if one of the children needs to go in the frontier
      if (!left_skel->cant_prune)
      {
        // This needs to be critical, because we'll update the skeleton 
        // frontier, however, this shouldn't happen very often
#pragma omp critical
        {
          cout << "rank " << my_mpi_rank << " adding frontier (left child couldn't skeletonize) " << node->leftNode->node_morton << ", " << node->leftNode->level+num_global_levels << "\n";
          my_skeleton_frontier.push_back(make_pair(node->leftNode->node_morton, node->leftNode->level+num_global_levels));
        }
      }
      // only 1 of these can be true
      else if (!right_skel->cant_prune)
      {
#pragma omp critical
        {
          cout << "rank " << my_mpi_rank << " adding frontier (right child couldn't skeletonize) " << node->rightNode->node_morton << ", " << node->rightNode->level+num_global_levels << "\n";
          my_skeleton_frontier.push_back(make_pair(node->rightNode->node_morton, node->rightNode->level+num_global_levels));
        }        
      }
      
      return;
    }
   
    // Combine the skeletons of the children
    num_sources = left_skel->numof_points + right_skel->numof_points;
    //source_vec = left_skel->X;
    //source_vec.insert(source_vec.end(), right_skel->X.begin(), right_skel->X.end());
   
    charge_vec = left_skel->charges;
    charge_vec.insert(charge_vec.end(), right_skel->charges.begin(), right_skel->charges.end());
   
    source_inds = left_skel->gids;
    source_inds.insert(source_inds.end(), right_skel->gids.begin(), right_skel->gids.end());

   
    // Note that we only skeletonize points we own
    // So, the coorindates are always in tree->inProcData
    // we still need to convert global ids to local ids, though
    for (int i = 0; i < num_sources; i++)
    {
      long skel_gid = source_inds[i];
      int skel_lid = tree->pos(skel_gid);
      
      // we only ever skeletonize points we own, so these live here
      std::vector<double>::iterator skel_coord = tree->inProcData->X.begin() + skel_lid*dim;
      source_vec.insert(source_vec.end(), skel_coord, skel_coord + dim);
      
    } // loop to get skeleton coordinates
    
  } // internal node
  
  // for now, putting this to be the maximum, need to think about the right 
  // way to specify this
  int num_targets_needed;
  if (do_absolute_num_targets)
  {
    num_targets_needed = num_skeleton_targets;
  }
  else {
    //num_targets_needed = num_skeleton_targets * num_sources * log(num_sources);
    num_targets_needed = num_skeleton_targets * num_sources;
    
    //std::cout << "num_skeleton_targets: " << num_skeleton_targets << ", num_sources: " << num_sources << ", targets needed: " << num_targets_needed << "\n";
  }

  //cout << "subsampling self targets\n";

  double subsample_self_targets_start = omp_get_wtime();
  
  target_vec = SubsampleSelfTargets(source_vec, num_targets_needed);

#pragma omp atomic
  subsample_self_targets_time += omp_get_wtime() - subsample_self_targets_start;

  // if (printout)
  //   std::cout << "Num self targets: " << target_vec.size() / dim << "\n";

  //cout << "collecting neighbor coords\n";

  // now, get the neighbors, if needed
  // num_targets_needed is the total number we need, not just the ones needed here
  double collect_neighbor_coords_start = omp_get_wtime();
    
  CollectNeighborCoords(neighbor_inds, target_vec, num_targets_needed);

#pragma omp atomic
  collect_neighbor_coords_time += omp_get_wtime() - collect_neighbor_coords_start;


  if (printout)
    std::cout << "Num sources: " << num_sources << ", Num self+neighbor targets: " << target_vec.size() / dim <<"\n";

  // If we have some uniform samples, then we add them now
  int num_uniform_needed = num_targets_needed - (target_vec.size() / dim) + num_uniform_required;
  //if (uniform_samples.size() > 0 && num_uniform_needed > 0)
  //    target_vec.insert(target_vec.end(), uniform_samples.begin(), uniform_samples.begin() + num_uniform_needed*dim);

  //cout << "need " << num_uniform_needed << " uniform samples \n";

  if (uniform_samples.size() > 0 && num_uniform_needed > 0)
  {
    uniform_samples.resize(num_uniform_needed * dim);
  }
  
//   if (printout && node->level == 10)
//   {
//     // cout << "\nSource coord 1: \n";
// //     for (int i = 0; i < dim; i++)
// //     {
// //       cout << source_vec[i] << " ";
// //     }
// //     cout << "\n\n";
// //
//     cout << "\nSources:\n";
//     for (int i = 0; i < source_inds.size(); i++)
//     {
//       cout << source_inds[i] << " ";
//     }
//     cout << "\n\n";
//
//     cout << "Targets:\n";
//     for (int i = 0; i < neighbor_inds.size() && i < 2 * source_inds.size(); i++)
//     {
//       cout << neighbor_inds[i].first << " ";
//     }
//     cout << "\n\n";
//
//   }

  // if (printout)
//     std::cout << "Num self+neighbor+uniform targets " << (target_vec.size() / dim) + (uniform_samples.size() / dim) << "\n";

  // this flag prints out some debugging info for the adaptive rank 
  // skeletonization -- can safely be set to false if you don't want it

  // Now, actually compute the skeleton
  ComputeSkeleton(my_skeleton, node->num_points_owned, source_vec, charge_vec, source_inds, 
    target_vec, uniform_samples, printout);
    
    // We can't prune this node, so we need to put its children in the frontier
    // we know it is either a leaf or both children could prune
  if (my_skeleton->cant_prune)
  {
    
    // its internal, so put both children in the frontier
    if (node->leftNode)
    {
#pragma omp critical // need this to modify the frontier
        {
          cout << "rank " << my_mpi_rank << " adding frontier (ComputeSkeleton failed) " << node->leftNode->node_morton << ", " << node->leftNode->level+num_global_levels  << "\n";
          cout << "rank " << my_mpi_rank << " adding frontier (ComputeSkeleton failed) " << node->rightNode->node_morton << ", " << node->rightNode->level+num_global_levels << "\n";
          my_skeleton_frontier.push_back(make_pair(node->leftNode->node_morton, node->leftNode->level+num_global_levels ));
          my_skeleton_frontier.push_back(make_pair(node->rightNode->node_morton, node->rightNode->level+num_global_levels));
        }
    } // not a leaf
    else {
      // is a leaf -- how to handle this? 
      cout << "WARNING: unskeletonized leaves not implemented yet.\n";
      
    }
    
  } // couldn't prune the node 

  // store the local ids for each point for the FMM step
  if (node->leftNode == NULL)
  {

    node->skeletons->local_ids.resize(node->leaf_point_gids.size());
    for (int i = 0; i < node->leaf_point_gids.size(); i++)
    {
      int child_ind = node->skeletons->skeleton_perm[i];
      int lid = tree->pos(node->leaf_point_gids[child_ind]);
      node->skeletons->local_ids[i] = lid;
    }
    
  }

} // ComputeSkeleton()


template<class TKernel>
std::vector<double> AskitAlg<TKernel>::SubsampleSelfTargets(const std::vector<double>& target_coords, int num_targets_needed)
{

  std::vector<double> target_coords_out;

  // Do we want to include self interactions at all?
  if (skeletonize_self_interactions)
  {
    // if we have too many self-targets, then we reduce to some other number
    int num_targets_in = target_coords.size() / dim;
    
    //std::cout << "num targets in: " << num_targets_in << ", wanted: " << num_wanted << "\n";
  
    if (num_targets_needed < num_targets_in)
    {
    
      std::vector<int> target_inds(num_targets_in);
      for (int i = 0; i < num_targets_in; i++)
      {
        target_inds[i] = i;
      }
      std::random_shuffle(target_inds.begin(), target_inds.end());
    
      // now, collect the coordinates
      for (int i = 0; i < num_targets_needed; i++)
      {
        target_coords_out.insert(target_coords_out.end(), target_coords.begin()+target_inds[i]*dim,
          target_coords.begin()+(target_inds[i]+1)*dim);
      }
    
    } // do we need to subsample
    else {
      target_coords_out = target_coords;
    }  
  
  } // are we using self interactions at all
  
  return target_coords_out;

} // subsample self targets


// This function computes the scaling parameter S_N for the improved 
// adaptive rank skeletonization algorithm
template<class TKernel>
void AskitAlg<TKernel>::ComputeNearScale()
{

  double compute_near_scale_start = omp_get_wtime();

  double my_near_scale = 1000000;

  // IMPORTANT: need to have called ConstructTreeList before this
  int level_ind = max_tree_level;

  std::vector<fks_ompNode*>::iterator tree_level = tree_list.begin() + (1 << level_ind) - 1;

  int num_nodes = 1 << level_ind;

  // Iterate over all leaf nodes
#pragma omp parallel for reduction(min:my_near_scale)
  for (int node_ind = 0; node_ind < num_nodes; node_ind++)
  {
    
    int my_thread_id = omp_get_thread_num();
    
    fks_ompNode* leaf = *(tree_level + node_ind);
    
    // Get the leaf point coordinates
    fksData* leaf_data = thread_leaf_data[my_thread_id];
    tree->getLeafData(leaf, leaf_data);
    int num_cols = leaf_data->numof_points;
    vector<double> source_vec = leaf_data->X;
    
    vector<int> source_local_inds(num_cols);
    for (int i = 0; i < num_cols; i++)
      source_local_inds[i] = tree->pos(leaf_data->gids[i]);
    
    // need to collect local ids of source points for kernel computation in the 
    // variable bandwidth case 
    
    // cout << "Found " << num_cols << " leaf points.\n";

    // include the sources in the targets
    vector<double> target_vec = source_vec;
    // Get the leaf's neighbor list
    vector<pair<long, double> > neighbor_list;
    ComputeLeafNeighbors(leaf, neighbor_list);
    // cout << "Found " << neighbor_list.size() << " neighbors.\n";
    
    // Truncate the neighbor list if it is longer than something? 
    int num_targets_needed = 2 * num_cols;
    CollectNeighborCoords(neighbor_list, target_vec, num_targets_needed);

    // The number of leaf points plus the number of neighbors we kept
    int num_rows = target_vec.size() / dim;

    // cout << "Found " << num_rows << " leaf points plus neighbors\n";
    
    // Form the matrix K of interactions here
    std::vector<double>& K = kernel_mat_space[my_thread_id];
    kernels[my_thread_id]->Compute(target_vec.begin(), target_vec.end(), 
        source_vec.begin(), source_vec.end(), dim, K, source_local_inds);
    
    // Now, compute the QR factorization
    IDWorkspace& workspace = *(id_workspaces[my_thread_id]);
    
    // Check if we allocated enough space
    if (num_cols > workspace.tau.size())
    {
      // std::cout << "Resizing ID workspace\n";
      workspace.tau.resize(num_cols);
    }

    lapack_int lda = num_rows; // because it's col major
  
    // set all pivots to 0, this indicates that everything is available to be 
    // pivoted 
    // We'll just ignore this, however
    vector<lapack_int> skeleton_out(num_cols, 0);
  
    // scalar factors of elementary reflectors
    double* tau = workspace.tau.data();
  

    double qr_start = omp_get_wtime();
    // Now, compute the pivoted QR
    lapack_int output = LAPACKE_dgeqp3(LAPACK_COL_MAJOR, num_rows, num_cols, K.data(), 
        lda, skeleton_out.data(), tau);
      
    qr_time = omp_get_wtime() - qr_start;

    // Get the smallest singular value (or diagonal entry of R)
    double r_mm = fabs(K[(num_cols-1) + (num_cols-1) * lda]);

    my_near_scale = min(my_near_scale, r_mm);
    
  }
  
  // Reduce to get the global minimum
  MPI_Allreduce(&my_near_scale, &near_scale, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  
  if (my_mpi_rank == 0)
    cout << "near_scale: " << near_scale << "\n";
  
  compute_near_scale_time = omp_get_wtime() - compute_near_scale_start;
  
} // ComputeNearScale


template<class TKernel>
void AskitAlg<TKernel>::CollectNeighborCoords(std::vector<std::pair<long, double> >& neighbor_inds,
    std::vector<double>& target_vec, int num_targets_needed)
{
  // now, copy out the targets
  // We have to do it this way, because we don't know which ones we kept 
  // from the sorting step -- this can probably be improved, though

  // We also stop this loop once we have enough targets
  for (int i = 0; i < neighbor_inds.size() && (target_vec.size() / dim) < num_targets_needed; i++)
  {

    int neighbor_lid = tree->pos(neighbor_inds[i].first);

    int neighbor_coord_begin = neighbor_lid * dim;
    int neighbor_coord_end = (neighbor_lid + 1) * dim;

    target_vec.insert(target_vec.end(), tree->inProcData->X.begin() + neighbor_coord_begin,
        tree->inProcData->X.begin() + neighbor_coord_end);
  
  } // loop over neighbor_inds
  
} // CollectNeighborCoords



// fills in the neighbor list for a leaf and truncates it if necessary
// returns number of sources
template<class TKernel>
int AskitAlg<TKernel>::ComputeLeafNeighbors(fks_ompNode* node, 
  std::vector<std::pair<long, double> >& neighbor_inds)
{

  int my_thread_id = omp_get_thread_num();

  // Get the leaf's exclusive nearest neighbors
  fksData* leaf_nn_data = thread_leaf_nn_data[my_thread_id];
  //std::cout << "Rank " << my_rank << " getting NN data\n";
  
  // This now just gets us the neighbors for the sampling (excluding the ones
  // for pruning)
  tree->getLeafexclNN(node, leaf_nn_data, sampling_num_neighbors);

  //std::cout << "finished getting NN data\n";
  
  // set the neighbors for passing up the tree
  neighbor_inds.resize(leaf_nn_data->numof_points);
  for (int i = 0; i < leaf_nn_data->numof_points; i++)
  {
    // getLeafexclNN fills the charges with the distances now
    double neighbor_dist = leaf_nn_data->charges[i];
    neighbor_inds[i] = std::pair<long, double>(leaf_nn_data->gids[i], neighbor_dist);
  } // loop over neighbors
  
  
  // We're setting the number of neighbors we need to keep to the minimum 
  // required
  int truncation_size;
  if (do_absolute_num_targets)
    truncation_size = num_skeleton_targets * neighbors_to_pass_up;
  else
  {
    //truncation_size = num_skeleton_targets * leaf_size * log(leaf_size);
    //truncation_size = num_skeleton_targets * leaf_size;
    // We want to keep more neighbors in the adaptive rank case 
    truncation_size = num_skeleton_targets * neighbors_to_pass_up * max(leaf_size, 2*id_rank);
  }
    
  // Now, we need to truncate the neighbor list
  if (leaf_nn_data->numof_points > truncation_size) {
    
    // cout << "Leaf truncating from " << neighbor_inds.size() << " to " << truncation_size << " neighbors.\n";
    // Sort the neighbors by distance 
    std::sort(neighbor_inds.begin(), neighbor_inds.end(), comp_neighbor_dist());
    neighbor_inds.resize(truncation_size);
  
  } // truncating neighbor list
  // else {
  //   cout << "Leaf keeping all " << neighbor_inds.size() << " neighbors.\n";
  // }
  
  return leaf_nn_data->numof_points;

} // ComputeLeafNeighbors


// returns number of sources -- i.e. the sum of the skeleton sizes
template<class TKernel>
int AskitAlg<TKernel>::MergeNeighborLists(const std::vector<std::pair<long, double> >& left_list, 
  const std::vector<std::pair<long, double> >& right_list, 
  // these are lists of global ids of the pruning neighbors of the skeleton points of the child nodes
  const vector<long>& left_pruning_list, const vector<long>& right_pruning_list,
  std::vector<std::pair<long, double> >& output_list,
  fks_ompNode* node)
{
  
  // clear out anything that may be left in it
  output_list.clear();

  // Now, merge them
  std::vector<std::pair<long, double> > merged_neighbors = left_list;
  merged_neighbors.insert(merged_neighbors.end(), right_list.begin(), right_list.end());

  // Remove duplicates
  if (merged_neighbors.size() > 1) {    
    // sort by the first index, with the second as a tiebreaker -- this
    // is important because it ensures that we get the closest neighhbor
    // back in the end -- std::unique guarantees order up to iterator
    std::sort(merged_neighbors.begin(), merged_neighbors.end());

    std::vector<std::pair<long, double> >::iterator it;
    it = std::unique(merged_neighbors.begin(), merged_neighbors.end(), eq_neighbor_index());
    merged_neighbors.resize(it - merged_neighbors.begin());

  }
  
  // remove anything in the other pruning list
  // we need to add the dummy distance onto the other list to use the std
  // vector manipulation methods
  vector<pair<long, double> > pruning_list(left_pruning_list.size() + right_pruning_list.size());
  int left_size = left_pruning_list.size();
  for (int i = 0; i < left_size; i++)
  {
    pruning_list[i] = make_pair(left_pruning_list[i], 0.0);
  }
  for (int i = 0; i < right_pruning_list.size(); i++)
  {
    pruning_list[i + left_size] = make_pair(right_pruning_list[i], 0.0);
  }
  
  sort(pruning_list.begin(), pruning_list.end());
  
  vector<pair<long, double> >::iterator it;
  vector<pair<long, double> > set_diff_output(merged_neighbors.size() + 1);
  
  // This will only consider the gid, not the distance
  it = set_difference(merged_neighbors.begin(), merged_neighbors.end(),
    pruning_list.begin(), pruning_list.end(), set_diff_output.begin(), 
    comp_neighbor_index());

  // Now, merged_neighbors has removed anything in the prunign neighbor 
  // list of any of the skeleton points
  merged_neighbors.assign(set_diff_output.begin(), it);


  int num_to_keep;
  int num_left = node->leftNode->skeletons->numof_points;
  int num_right = node->rightNode->skeletons->numof_points;
  int total = num_left + num_right;
  if (do_absolute_num_targets)
    num_to_keep = num_skeleton_targets * neighbors_to_pass_up;
  else {
    //num_to_keep = num_skeleton_targets * total * log(total);  
    //num_to_keep = num_skeleton_targets * total;
    // keep more neighbors for the adaptive rank case
    num_to_keep = num_skeleton_targets * 2*id_rank * neighbors_to_pass_up;
  }

  // Now, if there are too many, we sort them by distance to only keep 
  // the closest ones
  if (merged_neighbors.size() > num_to_keep)
  {
    std::sort(merged_neighbors.begin(), merged_neighbors.end(), comp_neighbor_dist());
  }

  // Now, we go through the merged neighbors and add them to neighbor 
  // list if they do not belong to this node
  for (int i = 0; i < merged_neighbors.size(); i++)
  {
    
    // Get the morton ID of this neighbor 
    int lid = tree->pos(merged_neighbors[i].first);
    long this_neighbor_mid = tree->inProcData->mortons[lid];

    // need to update the level here -- only happens in parallel upward pass
    if (!tree->belong2Node(node->node_morton, node->level + num_global_levels, this_neighbor_mid))
    {
      output_list.push_back(merged_neighbors[i]);
    }

  } // loop over merged neighbors
  
  // output_list is already sorted 
  if (output_list.size() > num_to_keep)
  {
    // cout << "Internal at level " << node->level << " truncating from " << output_list.size() << " to " << num_to_keep << " neighbors.\n";
    output_list.resize(num_to_keep);
  }
  // else {
  //   cout << "Internal at level " << node->level << " keeping all " << output_list.size() << " neighbors.\n";
  // }
  
  
  // return the total number of sources in this skeleton computation
  return total;
  
} // MergeNeighborLists


template<class TKernel>
void AskitAlg<TKernel>::ParallelUpwardPass()
{

  int my_rank;
  MPI_Comm_rank(tree->comm, &my_rank);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // cout << "rank " << my_rank << " starting upward pass.\n";

  //std::cout << "Rank " << my_rank << " constructing tree list\n";

  // -1 is a dummy level
  self_skeleton_sizes.resize(tree_list.size(), make_pair(0,-1));

  // Stores the neighbors of the level below the one we're currently working on
  std::vector<std::vector<std::pair<long, double> > > child_neighbors;
  // Stores the neighbors of nodes on the level we're currently working on
  std::vector<std::vector<std::pair<long, double> > > current_neighbors;

  vector<vector<long> > current_pruning_lists;
  vector<vector<long> > child_pruning_lists;

  //std::cout << "Rank " << my_rank << " building skeletons\n";

  //cout << "max level: " << max_tree_level << ", root omp level: " << tree->root_omp->level << ", num global levels: " << num_global_levels << "\n";

  // Traverse level by level, starting from the bottom
  // we stop when we either run out of shared memory levels or we reach the 
  // min_skeleton_level
  for (int level_ind = max_tree_level;
       level_ind >= min_skeleton_level - num_global_levels && level_ind > 0;
       level_ind--)
  {

    if (my_mpi_rank == 0)
        std::cout << "\nRank " << my_mpi_rank << ". Upward pass level " << level_ind << "\n";

    std::vector<fks_ompNode*>::iterator tree_level = tree_list.begin() + (1 << level_ind) - 1;

    int num_nodes = 1 << level_ind;

    // Create space for each node on this level
    current_neighbors.resize(num_nodes);
    current_pruning_lists.resize(num_nodes);

    //std::cout << "Rank " << my_rank << " on level " << level_ind << " with " << num_nodes << " nodes\n";

    #pragma omp parallel for
    for (int node_ind = 0; node_ind < num_nodes; node_ind++)
    {

      //std::cout << "doing node " << node_ind << "\n";

      fks_ompNode* node = *(tree_level + node_ind);

      if (node != NULL) {

        // we're not a leaf, so collect the children's neighbor lists
        std::vector<std::pair<long, double> >& neighbor_list = current_neighbors[node_ind];

        // vector of coorindates of uniform samples
        std::vector<double> uniform_samples;

        int num_sources;

        // If it's a leaf, then hand in the current node list location to be 
        // filled
        if (node->leftNode == NULL) {
          
          num_sources = node->leaf_point_gids.size();

          double compute_leaf_neighbors_start = omp_get_wtime();
          // need to fill in the list for a leaf
          ComputeLeafNeighbors(node, neighbor_list);
#pragma omp atomic
          compute_leaf_neighbors_time += omp_get_wtime() - compute_leaf_neighbors_start;
          
        }
        else {
        
          // Get the children's lists
          int left_ind = 2 * node_ind;
          int right_ind = 2 * node_ind + 1;
          std::vector<std::pair<long, double> >& left_list = child_neighbors[left_ind];
          std::vector<std::pair<long, double> >& right_list = child_neighbors[right_ind];

          vector<long>& left_pruning_list = child_pruning_lists[left_ind];
          vector<long>& right_pruning_list = child_pruning_lists[right_ind];

          double merge_neighbor_lists_start = omp_get_wtime();
          // Merge the children's lists          
          num_sources = MergeNeighborLists(left_list, right_list, 
            left_pruning_list, right_pruning_list, neighbor_list, node);
#pragma omp atomic 
          merge_neighbor_lists_time += omp_get_wtime() - merge_neighbor_lists_start;
        
        } // non-leaf case

        // If we don't have enough targets, then we'll sample uniformly
        // to fill in the rest
        int targets_available = neighbor_list.size();
        // if we're using the self interactions, then the sources are 
        // possible targets as well
        if (skeletonize_self_interactions)
          targets_available += num_sources;

        int targets_needed;
        if (do_absolute_num_targets)
          targets_needed = num_skeleton_targets;
        else
          //targets_needed = num_skeleton_targets * num_sources * log(num_sources);
          targets_needed = num_skeleton_targets * num_sources;
        
        // This is the number that aren't already taken care of by neighbors  
        targets_needed = max(0, targets_needed - targets_available);
        
        // we always sample at least this many targets uniformly
        targets_needed += num_uniform_required;

        //cout << "computing local samples\n";

        double collect_local_samples_start = omp_get_wtime();
        
        uniform_samples = CollectLocalSamples(node, neighbor_list, targets_needed);

#pragma omp atomic
        collect_local_samples_time += omp_get_wtime() - collect_local_samples_start;
          
        //cout << "collected " << uniform_samples.size() / dim << " local samples\n";
        
        //cout << "computing skeleton\n";

        /*
        int num_subtract = (size > 1) ? 3 : 0;
        if (node->node_morton == 248 && node->level == 7-num_subtract)
        // if (node->node_morton == 248 && node->level == 4) // 8 ranks
        {
            
          vector<long> source_gids = node->leftNode->skeletons->gids;
          source_gids.insert(source_gids.end(), node->rightNode->skeletons->gids.begin(), node->rightNode->skeletons->gids.end());
          vector<double> source_charges = node->leftNode->skeletons->charges;
          source_charges.insert(source_charges.end(), node->rightNode->skeletons->charges.begin(), node->rightNode->skeletons->charges.end());
          
          cout << "left size: " << node->leftNode->skeletons->numof_points;
          cout << " right id: " << node->rightNode->node_morton << " right level " << node->rightNode->level << "\n";
          cout << "\n\nSources: ";
          for (int i = 0; i < source_gids.size(); i++)
          {
            cout << "(" << source_gids[i] << " " << source_charges[i] << "), ";
          }
          // cout << "\n\n";
          // cout << "Neighbors: ";
          // for(int i = 0; i < neighbor_list.size(); i++)
          // {
          //   cout << "(" << neighbor_list[i].first << " " << neighbor_list[i].second << "), ";
          // }
          cout << "\n\n";
          cout << "Uniform size: " << uniform_samples.size() << "\n";
          cout << "\n\n";
          
        }
        */
        
        double compute_skeleton_start = omp_get_wtime();
        
        ComputeSkeleton(node, neighbor_list, uniform_samples);
        
#pragma omp atomic
        compute_skeleton_time += omp_get_wtime() - compute_skeleton_start;
        
        if (!node->skeletons->cant_prune && level_ind == min_skeleton_level - num_global_levels)
        {
#pragma omp critical
          {
            cout << "rank " << my_mpi_rank << " adding frontier (reached min skeleton level) " << node->node_morton << ", " << node->level+num_global_levels << "\n";            
            my_skeleton_frontier.push_back(make_pair(node->node_morton, node->level+num_global_levels));
          }
        }
          
        // now, we need to fill in the pruning list for this node
        vector<long>& this_pruning_list = current_pruning_lists[node_ind];
        this_pruning_list.clear();
        for (int i = 0; i < node->skeletons->numof_points; i++)
        {
          long gid = node->skeletons->gids[i];
          int lid = tree->pos(gid);
          for (int j = 0; j < pruning_num_neighbors; j++)
          {
            long ind = (long)lid * (long)num_neighbors_per_point + (long)j;
            // We can do this because we'll always have the NN of a skeleton
            // point in the local tree
            this_pruning_list.push_back((*tree->inProcKNN)[ind].second);
          }
        } // loop over skeleton points for this node
        
        // record the skeleton sizes for later
        // int num_skel_points = node->skeletons->cant_prune ? 0 : node->skeletons->numof_points;
        self_skeleton_sizes[(1 << level_ind) - 1 + node_ind] = make_pair(node->skeletons->numof_points, level_ind + num_global_levels);
        
      } // does the node exist at all?

    } // loop over nodes in this level (end of parallel for)

    // This is a barrier for all threads

    // Now, swap the contents of current_neighbors and child_neighbors for
    // the next iteration
    current_neighbors.swap(child_neighbors);
    current_pruning_lists.swap(child_pruning_lists);

  } // loop over levels in the tree

  // If we need to do the distributed upward pass
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  
  double distributed_upward_pass_start = omp_get_wtime();
  // Need to pass the two children of the root's neighbor lists
  if (comm_size > 1)
    DistributedUpwardPass(child_neighbors[0], child_neighbors[1],
      child_pruning_lists[0], child_pruning_lists[1]);

  distributed_upward_pass_time += omp_get_wtime() - distributed_upward_pass_start;
  
} // Parallel upward pass


template<class TKernel>
void AskitAlg<TKernel>::DistributedUpwardPass(std::vector<std::pair<long, double> >& left_list,
      std::vector<std::pair<long, double> >& right_list,
      vector<long>& left_pruning_list,
      vector<long>& right_pruning_list)
{

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  // move the pointer to the root omp level (log p )
  fks_mpiNode* curr_mpi = tree->root_mpi;
  while(curr_mpi->fks_kid != NULL)
    curr_mpi = curr_mpi->fks_kid;
  
  
  vector<double> source_coords;
  vector<double> source_charges;
  vector<long> source_inds;
  vector<double> target_coords;
  
  int num_sources, num_targets;
  
  // now, we'll recurse until we reach the min skeleton level
  while(curr_mpi->level >= min_skeleton_level)
  {
    
    int local_rank, local_size;
    MPI_Comm_rank(curr_mpi->comm, &local_rank);
    MPI_Comm_size(curr_mpi->comm, &local_size);
    
    // We'll just make sure that the skeleton gets created here
    if (!curr_mpi->skeletons)
      curr_mpi->skeletons = new fksData;
    
    
    // If we're at the level of omp_root, then we can get the skeletons from
    // the children directly
    if ((1 << curr_mpi->level) == size)
    {
     
      // check if both children could prune 
      if (!tree->root_omp->leftNode->skeletons->cant_prune && !tree->root_omp->rightNode->skeletons->cant_prune)
      {
      
        // now create the vector of charges
        source_charges = tree->root_omp->leftNode->skeletons->charges;
        std::vector<double>& right_charges = tree->root_omp->rightNode->skeletons->charges;
        source_charges.insert(source_charges.end(), right_charges.begin(), right_charges.end());

        // Create the vector of indices
        source_inds = tree->root_omp->leftNode->skeletons->gids;
        std::vector<long>& right_inds = tree->root_omp->rightNode->skeletons->gids;
        source_inds.insert(source_inds.end(), right_inds.begin(), right_inds.end());

        // create space for coordinates
        source_coords.resize(source_inds.size() * dim);

        // gather the source coordinates
        for (int i = 0; i < source_inds.size(); i++)
        {
          int lid = tree->pos(source_inds[i]);
          for (int d = 0; d < dim; d++)
          {
            source_coords[d+i*dim] = tree->inProcData->X[d + lid*dim];
          }
        } // loop over source gids
      
      }
      else
      {
        curr_mpi->skeletons->cant_prune = true;
        
        if (!tree->root_omp->leftNode->skeletons->cant_prune)
        {
          fks_ompNode* node = tree->root_omp->leftNode;
          // cout << "Rank " << my_mpi_rank << " adding frontier (omp root left child), level " << node->level + num_global_levels << "\n";
          my_skeleton_frontier.push_back(make_pair(node->node_morton, node->level + num_global_levels));
        }
        
        if (!tree->root_omp->rightNode->skeletons->cant_prune){
          fks_ompNode* node = tree->root_omp->rightNode;
          // cout << "Rank " << my_mpi_rank << " adding frontier (omp root right child), level " << node->level + num_global_levels << "\n";
          my_skeleton_frontier.push_back(make_pair(node->node_morton, node->level + num_global_levels));          
        }
        
      } // one (or both) children couldn't prune
    
    }
    else {
      // we're higher up in the distributed tree, so we need to call the 
      // communication routines

      
      double merge_skeletons_of_kids_start = omp_get_wtime();
      fksData merged_skeleton;
      // note, we assume that if one or both kids can't skeletonize, then 
      // our flag gets set in this routine
      tree->mergeSkeletonsOfKids(curr_mpi, &merged_skeleton);
      merge_skeletons_of_kids_time += omp_get_wtime() - merge_skeletons_of_kids_start;
      
      // If we cant skeletonize, but rank 0's child can, then it goes in the frontier
      if (curr_mpi->skeletons->cant_prune && local_rank == 0 && !curr_mpi->fks_kid->skeletons->cant_prune)
      {
        // don't need to add num_global_levels here -- MPI node should have the 
        // right level
        // cout << "rank " << my_mpi_rank << " adding frontier (couldn't skeletonize dist child) " << curr_mpi->fks_kid->level << "\n";
        my_skeleton_frontier.push_back(make_pair(curr_mpi->fks_kid->node_morton, curr_mpi->fks_kid->level));
      }
      
      // If we can't skeletonize, but rank size/2's child can, then it goes in the frontier
      if (curr_mpi->skeletons->cant_prune && local_rank == (local_size / 2) && !curr_mpi->fks_kid->skeletons->cant_prune)
      {
        // cout << "rank " << my_mpi_rank << ", local rank " << local_rank << " adding frontier (couldn't skeletonize dist sibling) " << curr_mpi->fks_kid->level << "\n";
        my_skeleton_frontier.push_back(make_pair(curr_mpi->fks_kid->node_morton, curr_mpi->fks_kid->level));
      }
      
      source_charges = merged_skeleton.charges;
      source_inds = merged_skeleton.gids;
      num_sources = source_inds.size();
      source_coords.resize(dim * num_sources);

      // get the source coords
      for (int i = 0; i < num_sources; i++)
      {
        int lid = tree->pos(source_inds[i]);
        assert(lid >= 0 && lid < tree->inProcData->numof_points);
        for (int d = 0; d < dim; d++)
        {
          source_coords[d + i*dim] = tree->inProcData->X[d + lid*dim];
        }
      } // loop over source inds
      
    } // higher than log p level
    
    ////////////////////////////////////////////////////////////////////
    // Now, we collect the target coordinates 
    
    num_targets = 0;
    target_coords.clear();

    int num_targets_needed;
    if (do_absolute_num_targets)
      num_targets_needed = num_skeleton_targets;
    else
      num_targets_needed = num_skeleton_targets * source_charges.size();

    if (curr_mpi->skeletons->cant_prune)
      num_targets_needed = 0;

    // do we want to include the sources in the targets?
    if (skeletonize_self_interactions)
    {
      target_coords = SubsampleSelfTargets(source_coords, num_targets_needed);
      num_targets = target_coords.size() / dim;
    }


    if ((1 << curr_mpi->level) == size)
    {
    
      // Merge the neighbor lists for the root_omp
      std::vector<std::pair<long, double> > leaf_node_neighbors;
  
      double merge_neighbor_lists_start = omp_get_wtime();
      MergeNeighborLists(left_list, right_list, left_pruning_list, 
        right_pruning_list, leaf_node_neighbors, tree->root_omp);
      merge_neighbor_lists_time += omp_get_wtime() - merge_neighbor_lists_start;

      double collect_neighbor_coords_start = omp_get_wtime();
      CollectNeighborCoords(leaf_node_neighbors, target_coords, num_targets_needed);
      collect_neighbor_coords_time += omp_get_wtime() - collect_neighbor_coords_start;

      // Fill in exclknn_of node for the curr_mpi
      curr_mpi->excl_knn_of_this_node.resize(leaf_node_neighbors.size());
      for(int i = 0; i < leaf_node_neighbors.size(); i++) {

          int idx = tree->pos(leaf_node_neighbors[i].first);

          // the triple is <gid, morton, dist>
          curr_mpi->excl_knn_of_this_node[i].first = leaf_node_neighbors[i].first;
          // need to get the morton ID out of the local storage
          curr_mpi->excl_knn_of_this_node[i].second = tree->inProcData->mortons[idx];
          curr_mpi->excl_knn_of_this_node[i].third = leaf_node_neighbors[i].second;

      } // loop over neighbors
    
    } // at level log p we get neighbors locally
    else 
    {

      double merge_nn_list_start = omp_get_wtime();
      fksData nn_data;
      tree->mergeNNList(curr_mpi, num_targets_needed * neighbors_to_pass_up);
      merge_nn_list_time += omp_get_wtime() - merge_nn_list_start;


      double get_dist_node_excl_nn_start = omp_get_wtime();
      // fill in the coordinates
      tree->getDistNodeExclNN(curr_mpi, &nn_data);
      // now neighbor info lives in exclKNNofLeaf in curr_mpi
      get_dist_node_excl_nn_time += omp_get_wtime() - get_dist_node_excl_nn_start;

      // only include the neighbors we actually need
      int num_neighbors_needed = num_targets_needed - num_targets;
      if (num_neighbors_needed > 0)
      {
        // neeed to make sure that we don't take more neighbors than are
        // available
        int neighbors_to_take = std::min(nn_data.numof_points, num_neighbors_needed);
        target_coords.insert(target_coords.end(), nn_data.X.begin(), nn_data.X.begin()+(neighbors_to_take*dim));
        num_targets += neighbors_to_take;
      }

    }

    num_targets = target_coords.size() / dim;

    // leaf_node_neighbors is now correct
    int nsamples = 0;
    if (num_targets < num_targets_needed)
      nsamples = num_targets_needed - num_targets;
    
    // we always take at least this many samples
    if (!curr_mpi->skeletons->cant_prune)
      nsamples += num_uniform_required;
  
    // cout << "Rank " << my_mpi_rank << " calling uniformSampleSibling\n";
  
    // need to sample some far-field points
    double uniform_sample_sibling_start = omp_get_wtime();
    fksData sampData;
    tree->uniformSampleSibling(curr_mpi, nsamples, &sampData);
    uniform_sample_sibling_time += omp_get_wtime() - uniform_sample_sibling_start;

    // cout << "Rank " << my_mpi_rank << " finished uniformSampleSibling\n";

    num_targets = target_coords.size() / dim + nsamples;
    
    int skel_failed = 0;

    // Only local rank 0 computes a skeleton
    if (local_rank == 0 && !curr_mpi->skeletons->cant_prune)
    {
      
      double dist_compute_skeleton_start = omp_get_wtime();
      ComputeSkeleton(curr_mpi->skeletons, curr_mpi->num_points_owned, source_coords,
        source_charges, source_inds, target_coords, sampData.X);
      dist_compute_skeleton_time += omp_get_wtime() - dist_compute_skeleton_start;

      // now we check if this skeletonization failed
      if (curr_mpi->skeletons->cant_prune)
      {  
        // if it failed here, then the kid succeeded, so the child goes in the 
        // skeleton frontier 
        
        if ((1 << curr_mpi->level) == size)
        {
          // in this case, both children could skeletonize, but this node failed
          // cout << "rank " << my_mpi_rank << " (couldn't skeletonize at level log p) " << tree->root_omp->leftNode->level + num_global_levels << "\n";
          // cout << " curr_mpi -> level " << curr_mpi->level << "\n";
          my_skeleton_frontier.push_back(make_pair(tree->root_omp->leftNode->node_morton, tree->root_omp->leftNode->level + num_global_levels));
          my_skeleton_frontier.push_back(make_pair(tree->root_omp->rightNode->node_morton, tree->root_omp->rightNode->level + num_global_levels));
        }
        else {        
          // cout << "rank " << my_mpi_rank << " (couldn't skeletonize, adding self child) " << curr_mpi->fks_kid->level << "\n";
          my_skeleton_frontier.push_back(make_pair(curr_mpi->fks_kid->node_morton, curr_mpi->fks_kid->level));
        }
        
        skel_failed = 1;
      }
      else {
        
        // we succeeded, but if we're at min skeleton level, then we add it to 
        // the frontier
        if (curr_mpi->level == min_skeleton_level)
        {
          // cout << "Rank " << my_mpi_rank << " MPI reached min_skeleton_level " << curr_mpi->level << "\n";
          my_skeleton_frontier.push_back(make_pair(curr_mpi->node_morton, curr_mpi->level));          
        }

        // update the skeleton sizes array
        self_skeleton_sizes.push_back(make_pair(curr_mpi->skeletons->numof_points, curr_mpi->level));
      }
      
    } // only compute the skeleton on local rank 0

    // Need this so that the sibling can add the child to its frontier
    // also, we'll use this to set the flag for everyone
    MPI_Bcast(&skel_failed, 1, MPI_INT, 0, curr_mpi->comm);

    // only do this if we're above level log p
    if (skel_failed)
    {
      curr_mpi->skeletons->cant_prune = true;
      // we need the second check so that we don't add the node twice at level
      // log p
      if (local_rank == local_size / 2 && !((1 << curr_mpi->level) == size))
      {
        // cout << "rank " << my_mpi_rank << " (couldn't skeletonize, adding sibling child) " << curr_mpi->fks_kid->level << "\n";
        my_skeleton_frontier.push_back(make_pair(curr_mpi->fks_kid->node_morton, curr_mpi->fks_kid->level));        
      }
    }
    
    // cout << "Rank " << my_mpi_rank << " moving pointer up tree.\n";
    
    // move the pointer up
    curr_mpi = curr_mpi->fks_parent;
    
  } // proceed up the tree until we reach min skeleton level
  
} // DistributedUpwardPass


// This function merges the near and far interaction lists for the 
// more efficient evaluation code
// This updates the charge vector, so it needs to be called after 
// UpdateCharges()
template<class TKernel>
void AskitAlg<TKernel>::CreateSourceInteractionLists()
{

  // first, make a vector of skeleton sizes
  skeleton_sizes.resize(tree->letNodeList.size());

  long local_total_skeleton_size = 0;

  // collect all of the skeleton sizes (and compute the total while we're 
  // at it)
#pragma omp parallel for reduction(+:local_total_skeleton_size)
  for (int i = 0; i < tree->letNodeList.size(); i++)
  {
    
    fks_ompNode* node = tree->letNodeList[i];
    if (node && node->skeletons)
      skeleton_sizes[i] = node->skeletons->numof_points;
    else
      skeleton_sizes[i] = 0;

    local_total_skeleton_size += skeleton_sizes[i];
    
  } // loop over nodes

  // have to do this because OMP can't reduce over a member variable
  total_skeleton_size = local_total_skeleton_size;
  // store this for later
  training_skeleton_size_cutoff = total_skeleton_size;
    
  // the combined list of skeleton charges (to be appended to the 
  // charge list in)
  // vector<double> skeleton_charges(total_skeleton_size);
  skeleton_charges.resize(total_skeleton_size);
  
  // now, do scan over skeleton sizes so we know where to store the charges  
  skeleton_sizes_scan.resize(tree->letNodeList.size());
  omp_par::scan(skeleton_sizes.data(), skeleton_sizes_scan.data(), skeleton_sizes.size());
  // note that the function is defined such that the first entry is zero and 
  // the last entry doesn't include the total (which we already computed)

  // The parallel scan isn't working right now, so this is in serial
  // skeleton_sizes_scan[0] = 0;
//   for (int i = 1; i < skeleton_sizes.size(); i++)
//   {
//     skeleton_sizes_scan[i] = skeleton_sizes_scan[i-1] + skeleton_sizes[i-1];
//   }

  /*
  if (my_mpi_rank == 0)
  {
    for (int i = 0; i < skeleton_sizes.size(); i++)
    {
      // skeleton_sizes_scan[i] = skeleton_sizes_scan[i-1] + skeleton_sizes[i-1];
      cout << "skeleton scan[" << i << "]: " << skeleton_sizes[i] << ", " << skeleton_sizes_scan[i] << " total size " << total_skeleton_size << "\n";
    }
  }
  */
  
  direct_source_inds.resize(tree->letNodeList.size());
  direct_charge_inds.resize(tree->letNodeList.size());
  
  approx_source_inds.resize(tree->letNodeList.size());
  approx_charge_inds.resize(tree->letNodeList.size());
  
  // open mp won't reduce a member variable
  // long local_num_kernels = 0;
  
// #pragma omp parallel for reduction(+:local_num_kernels)
#pragma omp parallel for
  for (int i = 0; i < tree->letNodeList.size(); i++)
  {
    
    fks_ompNode* node = tree->letNodeList[i];
    
    // if we're a leaf
    if (node && node->leftNode == NULL)
    {
      
      direct_source_inds[i].resize(node->leaf_point_gids.size());
      direct_charge_inds[i].resize(node->leaf_point_gids.size());
      
      for (int j = 0; j < node->leaf_point_gids.size(); j++)
      {
        
        direct_source_inds[i][j] = tree->pos(node->leaf_point_gids[j]);
        direct_charge_inds[i][j] = direct_source_inds[i][j];
        
      }
      
      // local_num_kernels += direct_source_inds[i].size() * direct_node_to_target[i].size();
      
    } // is it a leaf?
    
    if (node && node->skeletons)
    {
      
      assert(skeleton_sizes[i] == node->skeletons->numof_points);
      
      approx_source_inds[i].resize(node->skeletons->numof_points);
      approx_charge_inds[i].resize(node->skeletons->numof_points);
        
      for (int j = 0; j < node->skeletons->numof_points; j++)
      {
        // these are the indices into the final charge vector
        // it goes after all the local charges, which is the last term
        approx_source_inds[i][j] = tree->pos(node->skeletons->gids[j]);
        
        int ind = j + skeleton_sizes_scan[i] + tree->inProcData->numof_points;
        approx_charge_inds[i][j] = ind;  
        skeleton_charges[j + skeleton_sizes_scan[i]] = node->skeletons->charges[j];

      }

      // local_num_kernels += approx_source_inds[i].size() * approx_node_to_target[i].size();
      
    } // does it have a skeleton?
    
  } // loop over nodes
  
  // save local var
  // num_downward_kernel_evals = local_num_kernels;
  
  // now merge the lists
  // data table is already merged by LET construction

  if (_DEBUG_INT_LISTS_)
    cout << "Rank: " << my_mpi_rank << " original skeleton charges size: " << skeleton_charges.size() << "\n";
  
} // CreateSourceInteractionLists



template <class TKernel>
void AskitAlg<TKernel>::UpdateTestMergedInteractionLists()
{
  int rank, size;
  MPI_Comm_rank(tree->comm, &rank);
  MPI_Comm_size(tree->comm, &size);

  // cout << "\n\n START OF UPDATETESTMERGEDINTERACTIONLISTS\n\n\n";
//
//   print_set(tree->my_set_skeletons, MPI_COMM_WORLD);
//
//   if (my_mpi_rank == 3)
//   {
//     cout << "approx source inds before: \n";
//     for (int i = 0; i < approx_source_inds.size(); i++)
//     {
//       cout << "i = " << i << ": ";
//       for (int j = 0; j < approx_source_inds[i].size(); j++)
//       {
//         cout << approx_source_inds[i][j] << ", ";
//       }
//       cout << "\n";
//     }
//   }
//
  
  // Need member variables:
  // old_let_size (number of nodes in LET before we added nodes for test points)
  // old_inprocdata_size (number of points in tree->inProcData before we added
  // neighbors and skeletons for test points)
  // old_charge_table_size
  
  ///////////////// UPDATED VERSION ///////////////////////////////////////
  
  // we start by copying the training point lists
  test_approx_source_inds = approx_source_inds;
  test_approx_charge_inds = approx_charge_inds;
  
  test_direct_source_inds = direct_source_inds;
  test_direct_charge_inds = direct_charge_inds;
  
  
  int num_new_nodes = tree->letNodeList.size() - approx_source_inds.size();
  // cout << "Rank " << my_mpi_rank << " added " << num_new_nodes << " nodes\n";
  // for (int i = approx_source_inds.size(); i < tree->letNodeList.size(); i++)
  // {
  //   cout << "Added ngid: " << tree->letNodeList[i]->global_node_id << "\n";
  // }
  
  // update the index sizes here if we have more entries in letNodeList
  // This will only add space onto the end if we picked up more entries in 
  // letNodeList
  test_approx_source_inds.resize(tree->letNodeList.size());
  test_approx_charge_inds.resize(tree->letNodeList.size());
  
  test_direct_source_inds.resize(tree->letNodeList.size());
  test_direct_charge_inds.resize(tree->letNodeList.size());
  
  // The charges for any new skeletons we picked up
  vector<double> extra_skeleton_charges;

  // iterators for the new skeleton maps
  set<triple<long, long, int> >::iterator skel_begin = tree->testing_set_skeletons.begin();
  set<triple<long, long, int> >::iterator skel_end = tree->testing_set_skeletons.end();

  set<triple<long, long, int> >::iterator it;

  long new_total_skeleton_size = total_skeleton_size;

  skeleton_sizes.resize(tree->letNodeList.size());
  skeleton_sizes_scan.resize(tree->letNodeList.size());
  

  // the updated total number of point charges -- this has been set before
  // we enter this function
  long total_num_points = tree->inProcData->numof_points;

  // Doing this in serial for now
  for (it = skel_begin; it != skel_end; it++)
  {
    
    // get the index into the letNodeList
    map<long, int>::iterator map_it = tree->letNodeMap.find( it->first );
    assert(map_it != tree->letNodeMap.end());
    int i = map_it->second;
    
    fks_ompNode* node = tree->letNodeList[i];

    skeleton_sizes[i] = node->skeletons->numof_points;
    // Now, it's no longer a scan, but it is the correct location for the 
    // skeleton charges in the big table
    skeleton_sizes_scan[i] = new_total_skeleton_size;

    // Get a copy of the charges for charge_table
    extra_skeleton_charges.insert(extra_skeleton_charges.end(), 
      node->skeletons->charges.begin(), node->skeletons->charges.end());
      
    // now, we can update the source and charge inds    
    test_approx_source_inds[i].resize(node->skeletons->numof_points);
    test_approx_charge_inds[i].resize(node->skeletons->numof_points);
      
    for (int j = 0; j < node->skeletons->numof_points; j++)
    {
  
      // we're going to attach the new point charges and skeleton charges to the
      // end of the existing lists. Now, the charge table will be:
      // original point charges, original skeleton charges, new point charges,
      // new skeleton charges
    
      test_approx_source_inds[i][j] = tree->pos(node->skeletons->gids[j]);
      test_approx_charge_inds[i][j] = j + skeleton_sizes_scan[i] + total_num_points;
      
    }
    
    // update the counter
    new_total_skeleton_size += node->skeletons->numof_points;
    
  } // iterate over new skeletons
  
  // update the global skeleton size counter
  total_skeleton_size = new_total_skeleton_size;

  
  // now, handle the new leaf nodes
  set<triple<long, long, int> >::iterator leaf_begin = tree->testing_set_leaves.begin();
  set<triple<long, long, int> >::iterator leaf_end = tree->testing_set_leaves.end();

  for (it = leaf_begin; it != leaf_end; it++)
  {  
    
    // this is the bad leaf node
    // bool printme = (it->first == 654 && my_mpi_rank == 1);
    
    // get the index into the letNodeList
    map<long, int>::iterator map_it = tree->letNodeMap.find( it->first );
    assert(map_it != tree->letNodeMap.end());
    int i = map_it->second;
    
    fks_ompNode* node = tree->letNodeList[i];
  
    test_direct_source_inds[i].resize(node->leaf_point_gids.size());
    test_direct_charge_inds[i].resize(node->leaf_point_gids.size());

    // What if we had the skeleton, but now we get the whole leaf
    // In this case, some of the points will be in the old section, some will
    // be in the new
    
    // Idea to fix: reset all of the charge indices here
    // i.e. just iterate through all the skeleton charge indices, subtract out
    // the old num source points and add the new num source points
    // Then, the charge ind here is just the lid, and the charge ind for 
    // the new skeleton points is just the same as in the training case

    for (int j = 0; j < node->leaf_point_gids.size(); j++)
    {
      int lid = tree->pos(node->leaf_point_gids[j]);
    
      test_direct_source_inds[i][j] = lid; // the source points haven't changed
      // we need to go past the end of the old charge table, then its the id of the point minus the ones we already accounted for
      // This is equal to lid + num_old_skeleton_charges
      
      // Possible case: a node wasn't part of the LET.  
      // However: all (or some) of the points owned by it were part of the LET
      // So, we check if the coordinates (and charge), are in the old part 
      // of the charge table or the new part
      if (lid < old_inprocdata_size)
      {
        test_direct_charge_inds[i][j] = lid;
      }  
      else
      {   
        test_direct_charge_inds[i][j] = lid - old_inprocdata_size + old_charge_table_size;
      }
      
      // if (printme)
      // {
      //   cout << "lid: " << lid << ", charge_ind: " << test_direct_charge_inds[i][j] << ", charge: " << tree->inProcData->charges[lid] << "\n";
      // }
    
    }
    
  } // iterate over new leaves  
  
  
  // we're attaching all the new charges (point and skeleton) at the end of 
  // charge table so that we don't invalidate any of the old charge indices
  charge_table.insert(charge_table.end(), tree->inProcData->charges.begin() + old_inprocdata_size, tree->inProcData->charges.end());
  charge_table.insert(charge_table.end(), extra_skeleton_charges.begin(), extra_skeleton_charges.end());

  test_node_source_inds = test_direct_source_inds;
  test_node_source_inds.insert(test_node_source_inds.end(), test_approx_source_inds.begin(), test_approx_source_inds.end());

  test_charge_inds = test_direct_charge_inds;
  test_charge_inds.insert(test_charge_inds.end(), test_approx_charge_inds.begin(), test_approx_charge_inds.end());

  test_target_inds = direct_test_node_to_target;
  test_target_inds.insert(test_target_inds.end(), approx_test_node_to_target.begin(), approx_test_node_to_target.end());
  

  // cout << "test_target_inds.size() " << test_target_inds.size() << "\n";
  //
  // if (my_mpi_rank == 3)
  // {
  //   cout << "approx source inds after: \n";
  //   for (int i = 0; i < approx_source_inds.size(); i++)
  //   {
  //     cout << "i = " << i << ": ";
  //     for (int j = 0; j < approx_source_inds[i].size(); j++)
  //     {
  //       cout << approx_source_inds[i][j] << ", ";
  //     }
  //     cout << "\n";
  //   }
  // }
  //
  //
  //
  // cout << "\n\n END OF UPDATETESTMERGEDINTERACTIONLISTS\n\n\n";
  // print_set(tree->my_set_skeletons, MPI_COMM_WORLD);


} // UpdateTestMergedInteractionLists


// updates the part of the charge table corresponding to skeletons
// call this after exchange let
template<class TKernel>
void AskitAlg<TKernel>::UpdateChargeTable()
{

  if (_DEBUG_SKELETON_SCAN_)
  {
    cout << "rank " << my_mpi_rank << ": Beginning of UpdateChargeTable check.\n";

    for (int i = 0; i < tree->letNodeList.size(); i++)
    {
    
      fks_ompNode* node = tree->letNodeList[i];
    
      if (node && node->skeletons)
      {
      
        if (skeleton_sizes[i] != node->skeletons->numof_points)
        {
          cout << "Rank " << my_mpi_rank << ": failing on node " << i << ". size: " << skeleton_sizes[i] << ", numof_points: " << node->skeletons->numof_points << "\n";
          assert(skeleton_sizes[i] == node->skeletons->numof_points);
        }
      } // does it have a skeleton?
    
    } // loop over nodes
  
  } // if debug skeleton scan

  int num_self_charges = tree->inProcData->charges.size();
  long old_skeleton_size = old_charge_table_size - old_inprocdata_size;
  long new_skeleton_size = total_skeleton_size - old_skeleton_size;
  long num_new_inprocdata = tree->inProcData->numof_points - old_inprocdata_size;
  
  if (_DEBUG_UPDATE_CHARGES)
    cout << "rank: " << my_mpi_rank << ", scan end: " << *(skeleton_sizes_scan.end()-1) << " , size end: " << *(skeleton_sizes.end()-1) << "\n";
  
  // The total amount of space we'll need
  charge_table.resize(tree->inProcData->charges.size() + total_skeleton_size);

  // First, we have to copy the original (before test point) charges
  memcpy(charge_table.data(), tree->inProcData->charges.data(), old_inprocdata_size * sizeof(double));

  // Now, we copy the testing point charges, if there are any
  memcpy(charge_table.data() + old_charge_table_size, 
    tree->inProcData->charges.data() + old_inprocdata_size, num_new_inprocdata * sizeof(double));

  // Now, we need to get the points out of the skeletons
  
  // cout << "rank: " << my_mpi_rank << " memcpy done\n";

  // we iterate over all nodes with skeletons, then update their charges 
  // in the table
  // This assumes nothing changes position between iterations
#pragma omp parallel for 
  for (int i = 0; i < tree->letNodeList.size(); i++)
  {
    
    fks_ompNode* node = tree->letNodeList[i];
    
    if (node && node->skeletons)
    {

      int ind;
      
      // In this case, we have a training node, so the charges go into place now
      if (skeleton_sizes_scan[i] + node->skeletons->numof_points <= training_skeleton_size_cutoff)
      {
        ind = skeleton_sizes_scan[i] + old_inprocdata_size;
      }
      else
      {
        // in this case, the charges need to be added at the very end
        // the skeleton_sizes_scan still accounts for this
        ind = skeleton_sizes_scan[i] + num_self_charges;
      }

      //cout<<"skeleton_sizes = "<<skeleton_sizes[i]<<", node->skeletons = "<<node->skeletons->numof_points<<endl;
      assert(skeleton_sizes[i] == node->skeletons->numof_points);
      for (int j = 0; j < node->skeletons->numof_points; j++)
      {
        charge_table[ ind + j ] = node->skeletons->charges[j]; 
      }
      
    } // does it have a skeleton?
    
  } // loop over nodes
  
} // UpdateChargeTable


#ifdef USE_KS
extern "C" {
  #include <ks.h>
}
#include <omp_dgsks_list.hpp>
#endif


// Approximates the potential for all points using the tree
template<class TKernel>
std::vector<double> AskitAlg<TKernel>::ComputeAll()
{

    int rank;
    MPI_Comm_rank(tree->comm, &rank);

    vector<double>& data_table = tree->inProcData->X;
    
    vector<double> potentials(N, 0.0);
    
    if (do_fmm)
      potentials.resize(N + total_skeleton_size, 0.0);
    
    double eval_start = omp_get_wtime();

#ifdef USE_KS
    
    // if(my_mpi_rank == 0) cout<<"USE_KS is defined "<<endl;

    if (_DEBUG_INT_LISTS_)
      cout << "Rank " << my_mpi_rank << " calling " << training_target_inds.size() << " interactions on " << charge_table.size() << "charges.\n";
    
    bool use_efficient = true;
    
    ks_t ker;
    if (kernel_params.type == ASKIT_LAPLACE)
    {
      if (my_mpi_rank == 0)
        cout << "Calling efficient Laplace kernel.\n";
      ker.type = KS_LAPLACE;
    }
    else if (kernel_params.type == ASKIT_POLYNOMIAL)
    {
      use_efficient = false;
    }
    else if (kernel_params.do_variable_bandwidth)
    {
      cout << "Doing variable bandwidth Gaussian.\n";
      cout << "variable_h size: " << kernel_params.variable_h.size() << "\n";
      cout << "data_table size / dim: " << data_table.size() / dim << "\n";
      ker.type = KS_GAUSSIAN_VAR_BANDWIDTH;
      ker.h = kernel_params.variable_h.data();
    }
    else  // we default to Gaussian for now
    {
      ker.type = KS_GAUSSIAN;
      ker.scal = -1.0 / ( 2.0 * kernels[ 0 ]->h * kernels[ 0 ]->h );
    }    

    // We do this because there isn't support for the efficient polynomial
    // kernel yet
    if (use_efficient) {
      omp_dgsks_list_separated_u_symmetric(
          &ker,
          dim,
          potentials,
          potential_map,
          data_table.data(),
          ( data_table.size() / dim ), 
          training_target_inds,
          node_source_inds,
          charge_table.data(),
          charge_inds
          );
      }
      else {
        ComputeAllNodeInteractions(data_table, data_table, charge_table, 
          potential_map, node_source_inds, training_target_inds, 
          charge_inds, potentials);
        }

#else    
    ComputeAllNodeInteractions(data_table, data_table, charge_table, 
      potential_map, node_source_inds, training_target_inds, 
      charge_inds, potentials);
#endif

    // we'll just store the total evaluation time in the near field slot for 
    // convenience    
    evaluation_time = omp_get_wtime() - eval_start;

    // For the FMM, we have to pass down and then discard the skeleton 
    // potentials
    if (do_fmm)
    {
    
      double pass_down_start = omp_get_wtime();
      // Handles updating the potentials with the node-node interactions 
      // computed above

      PassPotentialsDown(potentials);

      pass_potentials_down_time = omp_get_wtime() - pass_down_start;

      // cout << "u[0] after pass down: " << potentials[0] << "\n";
    
      // discard all but the last N potential values, the rest are just 
      // skeleton potentials
      potentials.resize(N);
    
    } // for the FMM version, we have to pass potentials down
    
    // we'll just store the total evaluation time in the near field slot for 
    // convenience    
    evaluation_time = omp_get_wtime() - eval_start;
   
    return potentials;

} // ComputeAll()


template <class TKernel>
void AskitAlg<TKernel>::AddTestPoints(long num_test_points, fksData* testData, const char* test_knn_filename,
    bool is_binary)
{

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Now, we read the KNN info as well
  vector< pair<double, long> > *test_kNN_rkdt = new vector< pair<double, long> >;

  assert(testData->dim == dim);

  if (is_binary) {
      knn::binread_knn(test_knn_filename, testData->gids,
        num_neighbors_per_point, test_kNN_rkdt);        

      //knn::binread_knn(askit_inputs.knn_info_file.c_str(), refData->gids[0],
      //  refData->gids.size(), num_neighbors_per_point, kNN_rkdt);
      //for (int i = 0; i < std::min(5, num_neighbors_per_point); i++)  // BXC
      //{
      //  cout << "test neighbor pair: " << (*test_kNN_rkdt)[i].first << ", " << (*test_kNN_rkdt)[i].second << "\n";
      //}   // BXC

  }
  else {
    knn::dlmread_knn(test_knn_filename, testData->gids,
      num_neighbors_per_point, test_kNN_rkdt);
  }

  // Shuffle points and knn, update the LET

  // cout << "Rank " << my_mpi_rank << " distributing test set.\n";

  double dist_test_set_start = omp_get_wtime();
  tree->DistributeTestingPoints(testData, test_kNN_rkdt);
  dist_test_set_time = omp_get_wtime() - dist_test_set_start;

  // fill in the counters
  N_test = testData->numof_points;
  global_N_test = num_test_points;

  MPI_Barrier(MPI_COMM_WORLD);  // BXC
  if(my_mpi_rank == 0) cout << "Rank " << my_mpi_rank << " updating LET with test points.\n";   // BXC

  // print_data(tree->inProcData, MPI_COMM_WORLD);

  approx_test_target_to_node.resize(tree->inProcTestData->numof_points);
  direct_test_target_to_node.resize(tree->inProcTestData->numof_points);

  double update_test_let_start = omp_get_wtime();

  vector<vector<long> > direct_test_target_to_node_gid(tree->inProcTestData->numof_points);
  vector<vector<long> > approx_test_target_to_node_gid(tree->inProcTestData->numof_points);

  tree->UpdateTestLET(direct_test_target_to_node_gid, approx_test_target_to_node_gid, pruning_num_neighbors);
  
  // now, convert the node global ids to local ids
#pragma omp parallel for 
  for (long i = 0; i < tree->inProcTestData->numof_points; i++)
  {
  
    direct_test_target_to_node[i].resize(direct_test_target_to_node_gid[i].size());
    approx_test_target_to_node[i].resize(approx_test_target_to_node_gid[i].size());
  
    // loop over node global ids, replace with index into letNodeList
    for (int j = 0; j < direct_test_target_to_node_gid[i].size(); j++)
    {
      map<long, int>::iterator map_it = tree->letNodeMap.find( direct_test_target_to_node_gid[i][j] );
      assert(map_it != tree->letNodeMap.end());
      int node_lid = map_it->second;
      direct_test_target_to_node[i][j] = node_lid;        
    }

    // do the same for the approx indices
    for (int j = 0; j < approx_test_target_to_node_gid[i].size(); j++)
    {
      map<long, int>::iterator map_it = tree->letNodeMap.find( approx_test_target_to_node_gid[i][j] );
      assert(map_it != tree->letNodeMap.end());
      int node_lid = map_it->second;
      approx_test_target_to_node[i][j] = node_lid;        
    }
  
  }  // loop over target points
  
  update_test_let_time = omp_get_wtime() - update_test_let_start;

  // print_data(tree->inProcData, MPI_COMM_WORLD);

  MPI_Barrier(tree->comm);  // BXC
  if(my_mpi_rank == 0) cout << "Rank " << my_mpi_rank << " finished AddTestPoints()\n";  // BXC

  double test_list_start = omp_get_wtime();

  // need to resize the node to target lists
  approx_test_node_to_target.resize(tree->letNodeList.size());
  direct_test_node_to_target.resize(tree->letNodeList.size());

  InvertInteractionList(approx_test_target_to_node, approx_test_node_to_target);
  InvertInteractionList(direct_test_target_to_node, direct_test_node_to_target);

  UpdateTestMergedInteractionLists();

  test_list_blocking_time = omp_get_wtime() - test_list_start;

  MPI_Barrier(tree->comm);    // BXC
  if(my_mpi_rank == 0) cout << "Rank " << my_mpi_rank << " finished inverting interaction lists.\n";  // BXC

} // AddTestPoints



template <class TKernel>
vector<double> AskitAlg<TKernel>::ComputeAllTestPotentials()
{
  
  // IMPORTANT: needs to be called after AddTestPoints()
  
  // cout << "Allocating potentials with " << tree->inProcTestData->numof_points << " points\n";
  vector<double> potentials(tree->inProcTestData->numof_points, 0.0);
  
  // Compute and report the total number of interactions
  long num_kernel_evals = 0;
  
  // cout << "\nPrinting all interaction counts: \n";
#pragma omp parallel for reduction(+:num_kernel_evals)
  for (int i = 0; i < test_target_inds.size(); i++)
  {
    // cout << "(" << training_target_inds[i].size() << " x " << node_source_inds[i].size() << ")\n";
    num_kernel_evals += test_target_inds[i].size() * test_node_source_inds[i].size();
  }

  if (my_mpi_rank == 0)
    cout << "\nTotal rank 0 TEST kernel evaluations: " << num_kernel_evals << "\n\n";


  MPI_Barrier(tree->comm);
  
  vector<double>& source_table = tree->inProcData->X;
  vector<double>& target_table = tree->inProcTestData->X;

  double eval_start = omp_get_wtime();

#ifdef USE_KS
  
  if (_DEBUG_INT_LISTS_)
    cout << "Rank " << my_mpi_rank << " calling " << test_target_inds.size() << " interactions on " << charge_table.size() << "charges.\n";
  
  ks_t ker;

  bool use_efficient = true;  
  
  if (kernel_params.type == ASKIT_LAPLACE)
  {
    if (my_mpi_rank == 0)
      cout << "Calling efficient Laplace kernel.\n";
    ker.type = KS_LAPLACE;
  }
  else if (kernel_params.type == ASKIT_POLYNOMIAL)
  {
    use_efficient = false;
  }
  else if (kernel_params.do_variable_bandwidth)
  {
    ker.type = KS_GAUSSIAN_VAR_BANDWIDTH;
    ker.h = kernel_params.variable_h.data();
  }
  else  // we default to Gaussian for now
  {
    ker.type = KS_GAUSSIAN;
    ker.scal = -1.0 / ( 2.0 * kernels[ 0 ]->h * kernels[ 0 ]->h );
  }    
  
  if (use_efficient) {
    omp_dgsks_list_unsymmetric(
      &ker,
      dim,
      potentials,
      ( target_table.size() / dim ), 
      target_table.data(),
      test_target_inds,
      ( source_table.size() / dim), 
      source_table.data(),
      test_node_source_inds,
      charge_table.data(),
      test_charge_inds
      );
    }
    else {
      ComputeAllNodeInteractions(target_table, source_table, charge_table, 
        test_target_inds, test_node_source_inds, 
        test_target_inds, test_charge_inds, potentials);      
    }
#else    
  ComputeAllNodeInteractions(target_table, source_table, charge_table, 
    test_target_inds, test_node_source_inds, 
    test_target_inds, test_charge_inds, potentials);
#endif

  // we'll just store the total evaluation time in the near field slot for 
  // convenience    
  test_evaluation_time = omp_get_wtime() - eval_start;

  MPI_Barrier(MPI_COMM_WORLD);
  
  //if(my_mpi_rank == 0) cout << "Rank " << my_mpi_rank << " Finished ComputeAll()\n";

  return potentials;
  
} // ComputeAllTestPotentials


template<class TKernel>
void AskitAlg<TKernel>::ComputeAllNodeInteractions(vector<double>& target_coordinates,  
    vector<double>& source_coordinates,
    vector<double>& charges,
    vector<vector<int> >& potential_ind_list,
    vector<vector<int> >& source_ind_list,
    vector<vector<int> >& target_ind_list,
    vector<vector<int> >& charge_ind_list,
    vector<double>& potentials_out)
{
  
  // This code will carry out the evaluations using the simple 
  // (non-optimized version)

/*  
  int bad_node_list_ind;

  if (my_mpi_rank == 1) 
  {
    long bad_node_gid = 654;

    map<long, int>::iterator map_it = tree->letNodeMap.find(bad_node_gid);
    if (map_it == tree->letNodeMap.end())
      bad_node_list_ind = -1;
    else
      bad_node_list_ind = map_it->second;
  }
  */
  std::vector<std::vector<double> > thread_potentials(omp_get_max_threads());
  for (int i = 0; i < thread_potentials.size(); i++)
  {
    thread_potentials[i].resize(potentials_out.size(), 0.0);
  }
  
  if (_DEBUG_INT_LISTS_)
  {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    for (int r = 0; r < size; r++)
    {
      MPI_Barrier(MPI_COMM_WORLD);
      if (r == my_mpi_rank)
      {
        cout << "Rank " << my_mpi_rank << " computing " << potential_ind_list.size() << " potentials, ";
        cout << source_ind_list.size() << " sources, " << target_ind_list.size() << " targets, and ";
        cout << charge_ind_list.size() << " charges.\n";
      }
    }
  }
  // iterate over source nodes and compute each interaction
#pragma omp parallel for
  for (int i = 0; i < source_ind_list.size(); i++)
  {
    
    int my_thread_id = omp_get_thread_num();
    
    vector<int>& this_potential_inds = potential_ind_list[i];
    std::vector<int>& this_source_inds = source_ind_list[i];
    std::vector<int>& this_target_inds = target_ind_list[i];
    std::vector<int>& this_charge_inds = charge_ind_list[i];
    
    /*
    if (i == bad_node_list_ind && my_mpi_rank == 1)
    {
      cout << "Rank " << my_mpi_rank << " calling interaction on potentials: " << this_potential_inds.size();
      cout << ", sources: " << this_source_inds.size() << ", targets " << this_target_inds.size() << ", charges: " << this_charge_inds.size() << "\n";

      cout << "inProcData charges: \n";
      for (int j = 0; j < source_ind_list[i].size(); j++)
      {
        cout << tree->inProcData->charges[this_source_inds[j]] << ", ";
      }
      cout << "\n charge table charges: \n";
      for (int j = 0; j < source_ind_list[i].size(); j++)
      {        
        cout << "(" << this_charge_inds[j] << ", " << charges[this_charge_inds[j]] << "), "; 
      }
      cout << "\n\n";


    }
    */
    // this will do the computation and accumulate the results
    // Currently, all coordinates live in local_table, since we don't support
    // the LET yet
    ComputeNodeInteraction(target_coordinates.data(), source_coordinates.data(), charges.data(),
          this_potential_inds, this_source_inds, this_target_inds, this_charge_inds, 
          thread_potentials[my_thread_id]);

  } // compute the interactions
  
  // now, reduce the separate u vectors
#pragma omp parallel for
  for (int i = 0; i < potentials_out.size(); i++)
  {
    
    for (int j = 0; j < thread_potentials.size(); j++)
    {
      potentials_out[i] += thread_potentials[j][i];
    }
  
  } // loop over entries of u
  
} // ComputeNodeInteractions


template<class TKernel>
void AskitAlg<TKernel>::ComputeNodeInteraction(double* target_coordinate_table, 
  double* source_coordinate_table,
  double* charge_table, 
  std::vector<int>& potential_inds,
  std::vector<int>& source_inds, std::vector<int>& target_inds, 
  std::vector<int>& charge_inds, std::vector<double>& potentials)
{
  
  int my_thread_id = omp_get_thread_num();
  
  TKernel& kernel = *(kernels[my_thread_id]);
  std::vector<double>& kernel_space = kernel_mat_space[my_thread_id];
  
  // collect the coordinates and charges
  std::vector<double> target_coords(target_inds.size() * dim);
  std::vector<double> source_coords(source_inds.size() * dim);
  std::vector<double> source_charges(charge_inds.size());

  bool printme = false;
  int target_ind = -1;

  // collect target coords  
  for (int i = 0; i < target_inds.size(); i++)
  {

    // if (target_inds[i] == 0)
    // {
    //   printme = true;
    //   target_ind = i;
    // }

    for (int d = 0; d < dim; d++)
    {
      target_coords[d + i*dim] = target_coordinate_table[d + target_inds[i]*dim];
    }
  }

  // if (printme)
  // {
  //   cout << "Source inds: ";
  // }

  // collect source coords and charges
  for (int i = 0; i < source_inds.size(); i++)
  {
    // if (printme)
    //   cout << "(" << source_inds[i] << ", " << charge_table[charge_inds[i]] << "), ";
    
    for (int d = 0; d < dim; d++)
    {
      source_coords[d + i*dim] = source_coordinate_table[d + source_inds[i]*dim];
    }
    source_charges[i] = charge_table[charge_inds[i]];
  }
  
  // if (printme)
  //   cout << "\n";

  // temporary storage for the output
  std::vector<double> u(target_coords.size());


  // do the computation
  // if (my_mpi_rank == 0)
  // {
  //   std::cout << "doing " << target_coords.size() / dim << " x " << source_coords.size() / dim << " computation\n";
  //   cout << "\nCharges:\n";
  //   for (int i = 0; i < source_inds.size(); i++)
  //   {
  //     cout << source_charges[i] << ", ";
  //   }
  //   cout << "\n\n";
  // }

  ComputeKernelMatvec(kernel, kernel_space,
    target_coords.begin(), target_coords.end(),
    source_coords.begin(), source_coords.end(),
    source_charges.begin(), u.begin(), dim, source_inds);

  // assemble the results into potentials
  //std::cout << "Computed potentials: ";
  for (int i = 0; i < target_inds.size(); i++)
  {
    // if (printme && target_inds[i] == target_ind)
    // {
    //   std::cout << "\nu = " << u[i] << "\n";
    //   cout << "target_inds[i]: " << target_inds[i] << ", " << "potential_inds[i]:" << potential_inds[i] << "\n\n";
    // }
    
    potentials[potential_inds[i]] += u[i];
  }
  // std::cout << "\n\n";
  
} // ComputeNodeInteraction          


//////////////////////// Error estimation //////////////////////////////


template<class TKernel>
std::vector<double> AskitAlg<TKernel>::ComputeDirect(vector<double>& target_coordinates)
{

  //cout << "Rank " << my_mpi_rank << " Calling ComputeDirect\n";
  // Compute the direct potentials for these coordinates
  int num_queries = target_coordinates.size() / dim;
  
  MPI_Barrier(MPI_COMM_WORLD);

  std::vector<double> final_potentials(num_queries, 0.0);

  // Start the timer
  //time_t direct_start = time(NULL);
  double direct_start = omp_get_wtime();

  std::vector<double> potentials(num_queries, 0.0);
  vector<double> temp_potentials(num_queries);

  TKernel& my_kernel = *(kernels[0]);
  std::vector<double>& my_K = kernel_mat_space[0];

  // sources and charges are the same for all threads and aren't written to
  std::vector<double>& source_coords = tree->inProcData->X;

  // need to make sure that we only compute interactions with the points that
  // we originally owned -- other processes will do the rest
  int num_sources = N;
  
  // Problem: if N is very large, then we can run out of memory here (after 
  // neighbors and skeletonization)
  // Solution: split into several chunks and compute separately
  int chunk_size = 100000;
  int num_chunks = num_sources / chunk_size;

  // cout << "num chunks: " << num_chunks << ", num_sources: " << num_sources << "\n";
  
  // Go to plus 1 to account for the case where there is only one chunk
  for (int chunk_ind = 0; chunk_ind < num_chunks+1; chunk_ind++)
  {

    std::vector<double>::iterator source_begin = source_coords.begin() + chunk_size * chunk_ind * dim;
    std::vector<double>::iterator source_end;
    
    if (chunk_ind == num_chunks)
    {
      // last batch, make sure we end at the right point
      source_end = source_coords.begin() + num_sources*dim;
    }
    else {
      source_end = source_coords.begin() + chunk_size * (chunk_ind + 1) * dim;
    }
    
    // There is a corner case where the chunk size divides the number of sources exactly, this covers it
    if ((source_end - source_begin) == 0)
      continue;
    
    std::vector<double>::iterator charges = tree->inProcData->charges.begin() + chunk_size * chunk_ind;

    vector<int> source_inds(source_end - source_begin);
    for (int i = 0; i < source_inds.size(); i++)
    {
      source_inds[i] = chunk_ind * chunk_size + i;
    }
    
    // cout << "Chunk " << chunk_ind << ". From: " << (chunk_size * chunk_ind * dim) << " to " << (source_end - source_begin) << "\n";

    ComputeKernelMatvec(my_kernel, my_K, target_coordinates.begin(), target_coordinates.end(),
      source_begin, source_end, charges, temp_potentials.begin(), dim, source_inds);

#pragma omp parallel for 
    for (int i = 0; i < num_queries; i++)
    {
      // cout << "temp potential: " << temp_potentials[i] << "\n";
      potentials[i] += temp_potentials[i];  
    } // sum over potentials

  } // loop over chunks of source points
  
  MPI_Barrier(MPI_COMM_WORLD);
    
  // now, reduce all the results on the root process
  MPI_Reduce(potentials.data(), final_potentials.data(), num_queries,
             MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);

  // End the timer and record the result

  //exact_comp_time = difftime(time(NULL), direct_start);
  exact_comp_time = omp_get_wtime() - direct_start;

  // cout << "Rank " << my_mpi_rank << " finished ComputeDirect\n";

  return final_potentials;

} // ComputeDirect


template<class TKernel>
vector<double> AskitAlg<TKernel>::ComputeDistNN(vector<long>& my_test_gids, vector<int>& my_test_lids, 
  vector<int>& num_test_lids, vector<int>& displ)
{
  
  // Each rank will compute the potentials for its gids, then gather them on 
  // rank 0

  // cout << "Rank " << my_mpi_rank << " Calling ComputeDistNN\n";
  
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int total_queries = displ[size-1] + num_test_lids[size-1];
  
  // Compute the direct potentials for these coordinates
  int num_queries = my_test_lids.size();
  
  MPI_Barrier(MPI_COMM_WORLD);

  vector<pair<long, double> > final_unsorted_potentials(total_queries);
  std::vector<double> final_potentials(total_queries, 0.0);
  vector<pair<long, double> > my_potentials(num_queries);

#pragma omp parallel for if (num_queries > 1000)
  for (int i = 0; i < num_queries; i++)
  {
  
    int my_thread_id = omp_get_thread_num();
  
    // Collect local query coordinates
    std::vector<double> query_coords(dim);

    int lid = my_test_lids[i];
    
    query_coords.assign(tree->inProcData->X.begin()+lid*dim,
      tree->inProcData->X.begin()+(lid+1)*dim);
    
    // Get the nearest neighbor info for th query 
    std::vector<std::pair<double, long> > query_nn_gids;

    query_nn_gids.assign(tree->inProcKNN->begin()+lid*num_neighbors_per_point,
      tree->inProcKNN->begin() + (lid+1)*num_neighbors_per_point);
  
    // Collect neighbors info
    std::vector<double> ref_coords(dim * num_neighbors_per_point);
    std::vector<double> charges(num_neighbors_per_point);

    int num_sources = num_neighbors_per_point;
    //std::cout << "computing NN for " << num_sources << " neighbors\n";
    vector<int> source_inds(num_sources);

    for (int j = 0; j < num_sources; j++)
    {
      // Get the charges and coordinates
      int slid = tree->pos(query_nn_gids[j].second);
      charges[j] = tree->inProcData->charges[slid];
      for (int d = 0; d < dim; d++)
      {
        ref_coords[j*dim + d] = tree->inProcData->X[slid * dim + d];
      }
      source_inds[j]  = slid;
      
    } // loop over sources
 
    std::vector<double>& K = kernel_mat_space[my_thread_id];
    
    double my_potential = ComputeKernelDotProd(*(kernels[my_thread_id]), K, query_coords,
      ref_coords.begin(), ref_coords.end(), charges.begin(), dim, source_inds);
    
    my_potentials[i] = make_pair(my_test_gids[i], my_potential);
      
  } // loop over queries

  MPI_Barrier(MPI_COMM_WORLD);
  
  MPI_Datatype msgtype;
  MPI_Type_contiguous(sizeof(pair<long, double>), MPI_BYTE, &msgtype);
  MPI_Type_commit(&msgtype);
  MPI_Gatherv(my_potentials.data(), num_queries, msgtype, 
    final_unsorted_potentials.data(), num_test_lids.data(), displ.data(), 
    msgtype, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
    
  if (my_mpi_rank == 0)
  {
    // Now, sort them and extract the potentials
    sort(final_unsorted_potentials.begin(), final_unsorted_potentials.end());
    for(int i = 0; i < final_unsorted_potentials.size(); i++)
    {
      // cout << "final potential: " << final_unsorted_potentials[i].first << ": " << final_unsorted_potentials[i].second << "\n";
      final_potentials[i] = final_unsorted_potentials[i].second;
    }
    // cout << "\n\n";
  }
  
  // cout << "Rank " << my_mpi_rank << " finished ComputeDistNN\n";
  
  return final_potentials;
  
}


// Computes NN potentials in the case that queries are test points
template<class TKernel>
vector<double> AskitAlg<TKernel>::ComputeTestDistNN(vector<long>& my_test_gids, vector<int>& my_test_lids, 
  vector<int>& num_test_lids, vector<int>& displ)
{
  
  // Each rank will compute the potentials for its gids, then gather them on 
  // rank 0
  
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int total_queries = displ[size-1] + num_test_lids[size-1];
  
  // Compute the direct potentials for these coordinates
  int num_queries = my_test_lids.size();
  
  // MPI_Barrier(MPI_COMM_WORLD);

  // for (int r = 0; r < size; r++)
  // {
  //   MPI_Barrier(MPI_COMM_WORLD);
  //
  //   if (my_mpi_rank == r)
  //   {
  //
  //     cout << "Rank " << my_mpi_rank << " doing " << num_queries << ", lids: " << my_test_lids.size() << ", gids: " << my_test_gids.size() << " out of total " << tree->inProcTestData->numof_points << "\n";
  //     for (int i = 0; i < num_queries; i++)
  //     {
  //       cout << "lid " << my_test_lids[i] << ", gid " << my_test_gids[i] << "\n";
  //     }
  //     cout << "\n";
  //   }
  //
  //   MPI_Barrier(MPI_COMM_WORLD);
  // }
  //
  
  
  vector<pair<long, double> > final_unsorted_potentials(total_queries);
  std::vector<double> final_potentials(total_queries, 0.0);
  vector<pair<long, double> > my_potentials(num_queries);

#pragma omp parallel for if (num_queries > 1000)
  for (int i = 0; i < num_queries; i++)
  {
  
    int my_thread_id = omp_get_thread_num();
  
    // Collect local query coordinates
    std::vector<double> query_coords(dim);

    int lid = my_test_lids[i];

    // if (my_mpi_rank == 1)
//       cout << "rank " << my_mpi_rank << " lid " << my_test_lids[i] << ", gid " << my_test_gids[i] << "\n";

    query_coords.assign(tree->inProcTestData->X.begin()+lid*dim,
      tree->inProcTestData->X.begin()+(lid+1)*dim);
    
    // Get the nearest neighbor info for th query 
    // pair is <global id, morton > 
    std::vector<std::pair<long, long> > query_nn_gids;

    query_nn_gids.assign(tree->inProcTestKNN->begin()+lid*num_neighbors_per_point,
      tree->inProcTestKNN->begin() + (lid+1)*num_neighbors_per_point);

    // if (my_mpi_rank == 1)
//       cout << "rank " << my_mpi_rank << " lid " << my_test_lids[i] << ", gid " << my_test_gids[i] << "getting references.\n";
//
    // Collect neighbors info
    std::vector<double> ref_coords(dim * num_neighbors_per_point);
    std::vector<double> charges(num_neighbors_per_point);

    int num_sources = num_neighbors_per_point;
    //std::cout << "computing NN for " << num_sources << " neighbors\n";
    vector<int> source_inds(num_sources);

    for (int j = 0; j < num_sources; j++)
    {
      // Get the charges and coordinates
      int slid = tree->pos(query_nn_gids[j].first);
      // if (my_mpi_rank == 1)
//  cout << "neighbor local id: " << slid << ", global id " << query_nn_gids[j].second << "\n";
      charges[j] = tree->inProcData->charges[slid];
      for (int d = 0; d < dim; d++)
      {
        ref_coords[j*dim + d] = tree->inProcData->X[slid * dim + d];
      }
      source_inds[j] = slid;
    } // loop over sources
 
    std::vector<double>& K = kernel_mat_space[my_thread_id];
    
    double my_potential = ComputeKernelDotProd(*(kernels[my_thread_id]), K, query_coords,
      ref_coords.begin(), ref_coords.end(), charges.begin(), dim, source_inds);
    
    my_potentials[i] = make_pair(my_test_gids[i], my_potential);
      
  } // loop over queries

  // cout << "rank " << my_mpi_rank << " finished loop over queries\n";

  MPI_Barrier(MPI_COMM_WORLD);
  
  MPI_Datatype msgtype;
  MPI_Type_contiguous(sizeof(pair<long, double>), MPI_BYTE, &msgtype);
  MPI_Type_commit(&msgtype);
  MPI_Gatherv(my_potentials.data(), num_queries, msgtype, 
    final_unsorted_potentials.data(), num_test_lids.data(), displ.data(), 
    msgtype, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
    
  if (my_mpi_rank == 0)
  {
    // Now, sort them and extract the potentials
    sort(final_unsorted_potentials.begin(), final_unsorted_potentials.end());
    for(int i = 0; i < final_unsorted_potentials.size(); i++)
    {
      // cout << "final potential: " << final_unsorted_potentials[i].first << ": " << final_unsorted_potentials[i].second << "\n";
      final_potentials[i] = final_unsorted_potentials[i].second;
    }
    // cout << "\n\n";
  }
  
  return final_potentials;
  
}



template<class TKernel>
vector<double> AskitAlg<TKernel>::CollectPotentials(vector<double>& potentials, 
  vector<long>& my_test_gids, vector<int>& my_test_lids, vector<int>& num_test_lids, vector<int>& displ)
{
  
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // cout << "Rank " << my_mpi_rank << " Calling CollectPotentials\n";
  
  int total_queries = displ[size-1] + num_test_lids[size-1];
  
  // Gather the potentials for each gid
  vector<pair<long, double> > my_potentials(my_test_lids.size());
  vector<pair<long, double> > final_unsorted_potentials(total_queries);
  vector<double> final_potentials(total_queries, 0.0);
  int num_queries = my_test_lids.size();
  
  if (_DEBUG_INT_LISTS_)
    cout << "Rank " << my_mpi_rank << ": gathering " << my_test_lids.size() << " out of " << total_queries << " potentials.\n";
  
  for (int i = 0; i < num_queries; i++)
  {
    int lid = my_test_lids[i];
    my_potentials[i] = make_pair(my_test_gids[i], potentials[lid]);
    // if (my_mpi_rank == 0)
    //   cout << "gid: " << my_test_gids[i] << ", lid: " << lid << ", potential: " << potentials[lid] << "\n";
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  MPI_Datatype msgtype;
  MPI_Type_contiguous(sizeof(pair<long, double>), MPI_BYTE, &msgtype);
  MPI_Type_commit(&msgtype);

  MPI_Gatherv(my_potentials.data(), num_queries, msgtype, 
    final_unsorted_potentials.data(), num_test_lids.data(), displ.data(), 
    msgtype, 0, MPI_COMM_WORLD);
  
  MPI_Barrier(MPI_COMM_WORLD);
    
  if (my_mpi_rank == 0)
  {
    // Now, sort them and extract the potentials
    sort(final_unsorted_potentials.begin(), final_unsorted_potentials.end());
    for(int i = 0; i < final_unsorted_potentials.size(); i++)
    {
      // cout << "final potential: " << final_unsorted_potentials[i].first << ": " << final_unsorted_potentials[i].second << "\n";
      final_potentials[i] = final_unsorted_potentials[i].second;
    }
    // cout << "\n\n";
  }
  
  // cout << "Rank " << my_mpi_rank << " finished CollectPotentials\n";
  
  return final_potentials;
  
} // collect potentials


///////////////// FMM Functions /////////////////////////////////


// Here, we go through the interaction lists for each target point and merge 
// what we can across nodes
template<class TKernel>
void AskitAlg<TKernel>::ComputeFMMInteractionLists(vector<vector<int> >& approx_target_to_node, 
  vector<vector<int> >& source_node_to_target_node)
{

  // for each target node, the list of source nodes that it interacts with 
  // approximately
  vector<vector<int> > target_node_to_source_node(tree->letNodeList.size());
  // for each source node, the list of target nodes it interacts with 
  // approximately
  source_node_to_target_node.resize(tree->letNodeList.size());

  int max_global_tree_level = tree->depth_let - 1;
  
  // cout << "max global level: " << tree->depth_let << "\n";

  // This is the number of leaves owned by this MPI rank
  int num_leaves = (1 << (tree->depth_omp - 1));

  // stores the interaction lists (list of source nodes) for each target node
  // on the current level (current iteraction of outer loop)
  vector<vector<int> > this_level_node_to_node(num_leaves);
  
  // stores the interaction lists of the level below the current one
  vector<vector<int> > child_level_node_to_node(num_leaves);
  
  // cout << "Rank " << my_mpi_rank << ": depth_omp " << tree->depth_omp << ", depth_mpi " << tree->depth_mpi << "\n";
  
  // iterate over targets and compute the size of the leaf interaction lists
  long local_before = 0;
#pragma omp parallel for reduction(+:local_before)
  for (int i = 0; i < N; i++)
  {
    local_before += approx_target_to_node[i].size();
  }
  
  leaf_list_size_before = local_before;
  
  // iterate over all levels of the tree from leaf to root
  for (int level_ind = max_global_tree_level;
       level_ind >= min_skeleton_level && level_ind > tree->depth_mpi - 1;
       level_ind--)
  {

    // cout << "Rank " << my_mpi_rank << " on level " << level_ind << " of ComputeFMMInteractionLists.\n";

    // The global ids at this level are the total number above the current 
    // level plus the number owned by ranks before ours
    long nodes_on_this_level = (1 << level_ind) / comm_size;
    
    long gid_start = (1 << level_ind) - 1 + nodes_on_this_level * my_mpi_rank;

    // this is how many we have
    long gid_end = gid_start + nodes_on_this_level;
    
    // cout << "Rank " << my_mpi_rank << " gid_start " << gid_start << ", gid_end " << gid_end << "\n";
    
#pragma omp parallel for
    for (long node_gid = gid_start; node_gid < gid_end; node_gid++)
    {
      
      // if (node_gid == 2142)
      //   cout << "Found node with target 0\n";
      
      map<long, int>::iterator it = tree->letNodeMap.find( node_gid );
      assert(it != tree->letNodeMap.end());

      int node_lid = it->second;
      fks_ompNode* node = tree->letNodeList[node_lid];
      assert(node != NULL); // make sure we found it

      // If the node failed to skeletonize in the adaptive level restriction
      // case, then we won't form any skeleton potentials for it
      // However, in this case we need to make sure we output the lists for 
      // its children
      if (node->skeletons->cant_prune)
      {
        
        if (node->leftNode) {
          
          double add_to_map_start = omp_get_wtime();
        
          int nodes_on_child_level = (1 << node->leftNode->level) / comm_size;
        
          // get the lists of the two children
          // these should already be sorted from the iteration over their level
          long left_ind = node->leftNode->global_node_id - (1 << node->leftNode->level) + 1 - nodes_on_child_level * my_mpi_rank;
          vector<int>& left_list = child_level_node_to_node[left_ind];
        
          long right_ind = node->rightNode->global_node_id - (1 << node->rightNode->level) + 1 - nodes_on_child_level * my_mpi_rank;
          vector<int>& right_list = child_level_node_to_node[right_ind];
        
          // now, remove anything in the intersection from each list, and store
          // whatever remains in the final output list
          // add anything that remains in the left list to the output lists
          if (left_list.size() > 0)
          {

            // now, left list has been properly shrunk
            vector<int>& source_node_inds = left_list;
          
      
            map<long, int>::iterator it = tree->letNodeMap.find( node->leftNode->global_node_id );
            assert(it != tree->letNodeMap.end());
            int left_lid = it->second;

  #pragma omp critical (add_to_target_nodes_for_source)
            {
              target_node_to_source_node[left_lid].insert(target_node_to_source_node[left_lid].end(),
                source_node_inds.begin(), source_node_inds.end());
            } // critical block 

          } // if there is anything left in the left list

          // add anything that remains in the right list to the output lists
          if (right_list.size() > 0)
          {

            // now, left list has been properly shrunk
            vector<int>& source_node_inds = right_list;

            map<long, int>::iterator it = tree->letNodeMap.find( node->rightNode->global_node_id );
            assert(it != tree->letNodeMap.end());
            int right_lid = it->second;

#pragma omp critical (add_to_target_nodes_for_source)
            {
              target_node_to_source_node[right_lid].insert(target_node_to_source_node[right_lid].end(),
                source_node_inds.begin(), source_node_inds.end());
            } // critical block 

          } // if there is anything left in the right list
          
          double add_to_map_end = omp_get_wtime();
        
#pragma omp atomic
          fmm_add_to_map_time += add_to_map_end - add_to_map_start;
        } // if the node is not a leaf
          
        continue; // don't do anything else for this node
      
      } // a node we can't prune in the adaptive level restriction scheme


      // index on this level
      int node_ind = node_gid - gid_start;

      // vector<int>& this_list = this_level_node_to_node[node_ind];
      vector<int> this_list;
      
      // just to store the output of the set_intersection call
      vector<int> intersection_output;

      vector<int>::iterator intersection_end;

      // Need: list of targets in the node
      if (node->leftNode == NULL)
      {

        // initialize the list to be the first point's interaction list
        // this will copy, can I avoid this? 
        this_list = approx_target_to_node[tree->pos(node->leaf_point_gids[0])];
        if (!merge_aggressive)
          sort(this_list.begin(), this_list.end());
        
        for (int i = 1; i < node->leaf_point_gids.size(); i++)
        {
          int lid = tree->pos(node->leaf_point_gids[i]);
          // make sure we're looking at target points that exist
          assert(lid >= 0 && lid < N);
          
          vector<int> list_i = approx_target_to_node[lid];

          if (merge_aggressive)
          {
           
            double merge_lists_agg_start = omp_get_wtime();
            MergeTargetListsFMMFull(this_list, list_i, intersection_output);
            double merge_lists_agg_end = omp_get_wtime();
            
#pragma omp atomic
            merge_fmm_lists_aggressive_time += merge_lists_agg_end - merge_lists_agg_start;
                        
          }
          else
          {
            double merge_lists_basic_start = omp_get_wtime();
            sort(list_i.begin(), list_i.end());
            MergeTargetListsFMMBasic(this_list, list_i, intersection_output);
            double merge_lists_basic_end = omp_get_wtime();
#pragma omp atomic 
            merge_fmm_lists_basic_time += merge_lists_basic_end - merge_lists_basic_start;
          }
          
          this_list = intersection_output;
        
        } // loop over all points in the leaf

        // now, this_list contains the list of all source nodes that can be 
        // pruned by every target point in this node
        
        // need to remove any source node that is still present from the 
        // lists for each point -- we didn't necessarily catch all of these 
        // the first time through all of the points
        if (this_list.size() > 0) {

          for (int i = 0; i < node->leaf_point_gids.size(); i++)
          {
          
            int lid = tree->pos(node->leaf_point_gids[i]);
        
            vector<int>& list_i = approx_target_to_node[lid];
            
            if (merge_aggressive)
            {
          
              // we'll just call the same function again to make sure the 
              // individual point lists are fully expanded
              
              double merge_lists_agg_start = omp_get_wtime();
              MergeTargetListsFMMFull(list_i, this_list, intersection_output);
              double merge_lists_agg_end = omp_get_wtime();
            
  #pragma omp atomic
              merge_fmm_lists_aggressive_time += merge_lists_agg_end - merge_lists_agg_start;
            
              
              // we don't care about the output, just what happens to list_i
              // shouldn't change this_list at all
              
              // need to set this again for the next iteration
              this_list = intersection_output;
              
            }
            else {
                
              double merge_basic_set_difference_start = omp_get_wtime();
              sort(list_i.begin(), list_i.end());

              intersection_output.resize(list_i.size());
              intersection_end = set_difference(list_i.begin(), list_i.end(), this_list.begin(), this_list.end(), intersection_output.begin());

              // save the set difference to the interaction list
              list_i.assign(intersection_output.begin(), intersection_end);
              double merge_basic_set_difference_end = omp_get_wtime();
              
#pragma omp atomic
              merge_basic_set_difference_time += merge_basic_set_difference_end - merge_basic_set_difference_start;
          
            }
            
          } // loop over target points to compress lists
        
          // assign the list of node interactions so we can pass it up
          this_level_node_to_node[node_ind] = this_list;

        } // if there are any node-to-node prunes

      } // this node is a leaf
      else { // not a leaf
        
        int nodes_on_child_level = (1 << node->leftNode->level) / comm_size;
        
        // get the lists of the two children
        // these should already be sorted from the iteration over their level
        long left_ind = node->leftNode->global_node_id - (1 << node->leftNode->level) + 1 - nodes_on_child_level * my_mpi_rank;
        vector<int>& left_list = child_level_node_to_node[left_ind];
        
        long right_ind = node->rightNode->global_node_id - (1 << node->rightNode->level) + 1 - nodes_on_child_level * my_mpi_rank;
        vector<int>& right_list = child_level_node_to_node[right_ind];
        
        if (merge_aggressive)
        {

            double merge_lists_agg_start = omp_get_wtime();
            MergeTargetListsFMMFull(left_list, right_list, this_level_node_to_node[node_ind]);
            double merge_lists_agg_end = omp_get_wtime();
          
#pragma omp atomic
            merge_fmm_lists_aggressive_time += merge_lists_agg_end - merge_lists_agg_start;
          
        }
        else
        {
          double merge_lists_start = omp_get_wtime();
          MergeTargetListsFMMBasic(left_list, right_list, this_level_node_to_node[node_ind]);
          double merge_lists_end = omp_get_wtime();
#pragma omp atomic
          merge_fmm_lists_basic_time += merge_lists_end - merge_lists_start;
        }
        
        double add_to_map_start = omp_get_wtime();
        
        // now, remove anything in the intersection from each list, and store
        // whatever remains in the final output list
        // add anything that remains in the left list to the output lists
        if (left_list.size() > 0)
        {

          // now, left list has been properly shrunk
          vector<int>& source_node_inds = left_list;
          
      
          map<long, int>::iterator it = tree->letNodeMap.find( node->leftNode->global_node_id );
          assert(it != tree->letNodeMap.end());
          int left_lid = it->second;

#pragma omp critical (add_to_target_nodes_for_source)
          {
            target_node_to_source_node[left_lid].insert(target_node_to_source_node[left_lid].end(),
              source_node_inds.begin(), source_node_inds.end());
          } // critical block 

        } // if there is anything left in the left list

        // add anything that remains in the right list to the output lists
        if (right_list.size() > 0)
        {

          // now, left list has been properly shrunk
          vector<int>& source_node_inds = right_list;

          map<long, int>::iterator it = tree->letNodeMap.find( node->rightNode->global_node_id );
          assert(it != tree->letNodeMap.end());
          int right_lid = it->second;

#pragma omp critical (add_to_target_nodes_for_source)
          {
            target_node_to_source_node[right_lid].insert(target_node_to_source_node[right_lid].end(),
              source_node_inds.begin(), source_node_inds.end());
          } // critical block 

        } // if there is anything left in the right list
       
        // in this case, we need to write out things in our list as well, because there is no next iteration to do it in
        if (level_ind == max(min_skeleton_level, tree->depth_mpi))
        {
        
          vector<int>& source_node_inds = this_level_node_to_node[node_ind];
          map<long, int>::iterator it = tree->letNodeMap.find( node->global_node_id );
          assert(it != tree->letNodeMap.end());
          int this_node_lid = it->second;

#pragma omp critical (add_to_target_nodes_for_source)
          {
            target_node_to_source_node[this_node_lid].insert(target_node_to_source_node[this_node_lid].end(),
              source_node_inds.begin(), source_node_inds.end());
          } // critical block 

        } // last level, need to add to lists
        double add_to_map_end = omp_get_wtime();
        
#pragma omp atomic
        fmm_add_to_map_time += add_to_map_end - add_to_map_start;
        
      } // not a leaf node

    } // loop over nodes in level
    
    // pass the lists up to the next level
    this_level_node_to_node.swap(child_level_node_to_node);
  
  } // loop over levels of tree
  
  // iterate over targets and compute the size of the leaf interaction lists
  long local_after = 0;
  
#pragma omp parallel for reduction(+:local_after)
  for (int i = 0; i < N; i++)
  {
    local_after += approx_target_to_node[i].size();
  }
  
  leaf_list_size_after = local_after;
  
  // For now, we don't worry about a distributed version -- we just stop at 
  // omp root

  
  // Now, we need to invert the target_node_to_source_node list
  // to obtain source_node_to_target_node
  InvertInteractionList(target_node_to_source_node, source_node_to_target_node);

} // ComputeFMMInteractionLists


template<class TKernel>
void AskitAlg<TKernel>::PassPotentialsDown(vector<double>& u)
{

  // The skeletonization step did not update the entries of letNodeList with 
  // the proj matrix and permutation -- however, we will need these here   
  if (tree->root_omp->skeletons && tree->root_omp->skeletons->numof_points > 0)
  {
    
    fks_mpiNode *curr_mpi = tree->root_mpi;
    while(curr_mpi->fks_kid != NULL)
      curr_mpi = curr_mpi->fks_kid;
    
    map<long, int>::iterator it = tree->letNodeMap.find(tree->root_omp->global_node_id);
    assert(it != tree->letNodeMap.end());
    
    fks_ompNode* list_node = tree->letNodeList[it->second];
    
    // cout << "MPI leaf proj: " << curr_mpi->skeletons->proj.size() << "\n";
    
    list_node->skeletons->proj = curr_mpi->skeletons->proj;
    list_node->skeletons->skeleton_perm = curr_mpi->skeletons->skeleton_perm;
    
  }

  
  double internal_start = omp_get_wtime();

  // pass from root to level just above leaves
  for (int level_ind = max(min_skeleton_level, tree->depth_mpi - 1); level_ind < tree->depth_let - 1; level_ind++)
  {
    
    // cout << "Rank " << my_mpi_rank << " passing down at level " << level_ind << "\n";
    
    long num_nodes_per_rank = (1 << level_ind) / comm_size;
      
    long gid_start = (1 << level_ind) - 1 + num_nodes_per_rank * my_mpi_rank;

    // this is how many we have
    long gid_end = gid_start + num_nodes_per_rank;
  
#pragma omp parallel for 
    for (long node_gid = gid_start; node_gid < gid_end; node_gid++)
    {

      map<long, int>::iterator it = tree->letNodeMap.find(node_gid);
      assert(it != tree->letNodeMap.end());
      fks_ompNode* node = tree->letNodeList[it->second];

      // If we didn't skeletonize this node, then it shouldn't have any 
      // skeleton potentials, so we go to the next node
      if (!node->skeletons || node->skeletons->cant_prune || node->skeletons->numof_points == 0)
        continue;

      vector<double>::iterator parent_u = u.begin() + N + skeleton_sizes_scan[it->second];
      
      it = tree->letNodeMap.find(node->leftNode->global_node_id);
      assert(it != tree->letNodeMap.end());
      vector<double>::iterator left_u = u.begin() + N + skeleton_sizes_scan[it->second];

      it = tree->letNodeMap.find(node->rightNode->global_node_id);
      assert(it != tree->letNodeMap.end());
      vector<double>::iterator right_u = u.begin() + N + skeleton_sizes_scan[it->second];

      // sum of skeleton sizes 
      int num_child_skel = node->leftNode->skeletons->numof_points + node->rightNode->skeletons->numof_points;
      int num_skel = node->skeletons->numof_points;
      int num_cols = num_child_skel - num_skel;
      int num_left = node->leftNode->skeletons->numof_points;
      int num_right = node->rightNode->skeletons->numof_points;
      
      // cout << "Rank " << my_mpi_rank << " computing for left: " << num_left << " right: " << num_right;
      // cout << ", parent_u: " << parent_u - u.begin() << ", left u: " << left_u - u.begin() << ", right_u: " << right_u - u.begin() << ", u total: " << u.size();
      // cout << ", num_skel: " << num_skel << ", num_cols: " << num_cols << "\n";
      
      // check that we actually did any compression
      if (num_skel < num_child_skel)
      {

        // cout << "Rank " << my_mpi_rank << " Pass down for node " << node->node_morton << " at level " << node->level << " with num skel " << num_skel << "\n";
        
        // P is num_skel x (num_child_skel - num_skel)
        vector<double>& P = node->skeletons->proj;
        // vector<int>& perm = node->skeletons->skeleton_perm;
      
        // cout << "Rank " << my_mpi_rank << " P: " << P.size() << " left P: " << node->leftNode->skeletons->proj.size() << "\n";
      
        vector<double> u_temp(num_cols);

        // Apply the proj matrix to the charges of the non-skeleton points
        double oned = 1.0;
        double zerod = 0.0;
        int onei = 1;
        
        // cout << "Rank " << my_mpi_rank << " Doing matvec\n";
        
        // cout << "Rank " << my_mpi_rank << " level " << node->level << ", num_skel: " << num_skel << ", num_cols: " << num_cols << ", P.size(): " << P.size() << ", parent_u: " << parent_u - u.begin() << ", u_temp: " << u_temp.size() << "\n";
        cblas_dgemv("T", &num_skel, &num_cols, &oned, 
          P.data(), &num_skel, &(*parent_u), &onei, &zerod, u_temp.data(), &onei);

        // Now, need to fill in the potentials for the children
        // Get the values out of left_u and right u that don't correspond to 
        // the skeleton points of this node

        // for all child potentials, if the child skeleton is in the parent skeleton, we add the parent potential of the same skeleton point
        // If the child skeleton is not in the parent skeleton, then we add u_temp -- but do we need to permute this somehow?

        // cout << "Rank " << my_mpi_rank << " permuting charges\n";

#ifndef JUST_PASSDOWN_MATVECS
        for (int i = 0; i < num_skel; i++)
        {
          int child_ind = node->skeletons->skeleton_perm[i];
        
          if (child_ind < num_left)
          {
            left_u[child_ind] += parent_u[i];
          }
          else {
            right_u[child_ind - num_left] += parent_u[i];
          }
        
        } // loop over parent skeletons that are in children

        for (int i = num_skel; i < num_child_skel; i++)
        {
        
          int child_ind = node->skeletons->skeleton_perm[i];
        
          if (child_ind < num_left)
          {
            left_u[child_ind] += u_temp[i - num_skel];
          }
          else {
            right_u[child_ind - num_left] += u_temp[i - num_skel];
          }
        
        } // loop over parent skeletons which are not in children
#endif
      } // did we compress at all
      else { // no compression 

#ifndef JUST_PASSDOWN_MATVECS        
        for (int i = 0; i < num_child_skel; i++)
        {
          
          if (i < num_left)
            left_u[i] += parent_u[i];
          else
            right_u[i - num_left] += parent_u[i];
          
        } // loop over skeletons
#endif
        
      } // no compression
      
    } // loop over node inds
    
  } // loop over levels

  double internal_time = omp_get_wtime() - internal_start;
  // cout << "Internal pass down time: " << internal_time << endl;

  // now, at the leaf level, apply the potentials to the points in the leaf
  // to get the final potentials 

  double leaf_time_start = omp_get_wtime();

  long leaf_start = (1 << (tree->depth_let - 1) ) - 1 + num_omp_leaves*my_mpi_rank;
  long leaf_end = leaf_start + num_omp_leaves;

  // cout << "Rank " << my_mpi_rank << " push down at leaf level.\n";

  for (long node_gid = leaf_start; node_gid < leaf_end; node_gid++)
  {

    map<long, int>::iterator it = tree->letNodeMap.find(node_gid);
    fks_ompNode* node = tree->letNodeList[it->second];
    
    vector<double>::iterator parent_u = u.begin() + N + skeleton_sizes_scan[it->second];
    
    int num_children = node->leaf_point_gids.size();
    int num_skel = node->skeletons->numof_points;
    
    int num_cols = num_children - num_skel;
    
    // check if we did any compression
    if (num_skel < num_children) 
    {
    
      // cout << "Rank " << my_mpi_rank << " Pass down for node " << node->node_morton << " at level " << node->level << " with num skel: " << num_skel << "\n";
    
      // P is num_skel x (num_cols - num_skel)
      vector<double>& P = node->skeletons->proj;
      // vector<int>& perm = node->skeletons->skeleton_perm;
    
      vector<double> u_temp(num_cols);

      // cout << "Rank " << my_mpi_rank << " level " << node->level << ", num_skel: " << num_skel << ", num_cols: " << num_cols << ", P.size(): " << P.size() << ", parent_u: " << parent_u - u.begin() << ", u_temp: " << u_temp.size() << "\n";
      
      // Apply the proj matrix to the charges of the non-skeleton points
      double oned = 1.0;
      double zerod = 0.0;
      int onei = 1;
      cblas_dgemv("T", &num_skel, &num_cols, &oned, 
        P.data(), &num_skel, &(*parent_u), &onei, &zerod, u_temp.data(), &onei);

      // This doesn't fix it
      // Idea: store the locals before hand in the skeletonization phase
#ifndef JUST_PASSDOWN_MATVECS
      
      for (int i = 0; i < num_skel; i++)
      {
        u[node->skeletons->local_ids[i]] += *(parent_u + i);
      }
      for (int i = num_skel; i < num_children; i++)
      {
        u[node->skeletons->local_ids[i]] += u_temp[i-num_skel];
      }
      
      
        /* 
        // improved leaf
      for (int i = 0; i < num_skel; i++)
      {
      
        int child_ind = node->skeletons->skeleton_perm[i];
        int lid = tree->pos(node->leaf_point_gids[child_ind]);
      
        u[lid] += *(parent_u + i);
      
      }
      for (int i = num_skel; i < num_children; i++)
      {
      
        int child_ind = node->skeletons->skeleton_perm[i];
        int lid = tree->pos(node->leaf_point_gids[child_ind]);
      
        u[lid] += u_temp[i-num_skel];
      
      }
        */
        
      /*
      for (int i = 0; i < num_children; i++)
      {
        int child_ind = node->skeletons->skeleton_perm[i];
        int lid = tree->pos(node->leaf_point_gids[child_ind]);
      
        // if (lid == 0)
        // {
        //   if (i < num_skel)
        //     cout << "writing to u[0]: " << u[lid] << " += " << *(parent_u + i) << "\n";
        //   else
        //     cout << "writing to u[0]: " << u[lid] << " += " << u_temp[i - num_skel] << "\n";
        // }
      
        if (i < num_skel)
          u[lid] += *(parent_u + i);
        else
          u[lid] += u_temp[i-num_skel];
      } // loop over parent skeletons that are in children
      */
#endif
    } // if we compressed
    else {
      // no compression
      
#ifndef JUST_PASSDOWN_MATVECS
      for (int i = 0; i < num_children; i++)
      {
        int lid = tree->pos(node->leaf_point_gids[i]);
        
        // if (lid == 0)
        //   cout << "node: " << node_gid << " writing to u[0]: " << u[lid] << " += " << *(parent_u + i) << "\n";
       
        u[lid] += *(parent_u + i);
      }
#endif
      
    } // no compression
    
  } // loop over leaf nodes
  
  double leaf_time = omp_get_wtime() - leaf_time_start;
  // cout << "Leaf pass down time: " << leaf_time << endl;

} // PassPotentialsDown



// on input: list1 and list2 are lists of nodes sorted by pre-order traversal; 
// list_out is blank
// on exit: list_out will contain the merged list, list1 and list2 will have
// removed and updated things from the merged list
template<class TKernel>
void AskitAlg<TKernel>::MergeTargetListsFMMBasic(vector<int>& list1, 
  vector<int>& list2, vector<int>& list_out)
{

  list_out.resize(max(list1.size(), list2.size()));
  
  vector<int>::iterator it;
  
  it = set_intersection(list1.begin(), list1.end(), list2.begin(), list2.end(), 
    list_out.begin());

  // now, list_out is the intersection (still sorted correctly)
  list_out.resize(it - list_out.begin());

  // Now, we need to remove items in list_out from list1 and list2
  vector<int> difference(list1.size());
  it = set_difference(list1.begin(), list1.end(), list_out.begin(), list_out.end(), difference.begin());
  list1.assign(difference.begin(), it);

  difference.resize(list2.size());
  it = set_difference(list2.begin(), list2.end(), list_out.begin(), list_out.end(), difference.begin());
  list2.assign(difference.begin(), it);

}


// Lists are pairs <Morton ID, level> 
// This version splits nodes fully in an effort to merge as much as possible
template<class TKernel>
void AskitAlg<TKernel>::MergeTargetListsFMMFull(vector<int>& list1, 
  vector<int>& list2, vector<int>& list_out)
{

  double mtl_full_start = omp_get_wtime();

  vector<int> list1_copy = list1;
  vector<int> list2_copy = list2;
  
  list1.clear();
  list2.clear();

  list_out.clear();
  list_out.reserve(min(list1_copy.size(), list2_copy.size()));
  
  while(list1_copy.size() > 0 && list2_copy.size() > 0)
  {
    
    int list1_ind = list1_copy[0];
    fks_ompNode* list1_node = tree->letNodeList[list1_ind];
    pair<long, int> list1_val = make_pair(list1_node->node_morton, list1_node->level);
    
    int list2_ind = list2_copy[0];
    fks_ompNode* list2_node = tree->letNodeList[list2_ind];
    pair<long, int> list2_val = make_pair(list2_node->node_morton, list2_node->level);
    
    if (list1_ind == list2_ind)
    {
      // the same node appears in both lists, so we remove it from the lists
      // and place it in the output list
      
      list_out.push_back(list1_ind);

      list1_copy.erase(list1_copy.begin());
      list2_copy.erase(list2_copy.begin());
      
    }
    else if (isAncestor(list1_val, list2_val))
    {
      
      // list1's node is an ancestor of list2's node -- so split list1's node
      vector<pair<long, int> > list1_split = SplitNode(list1_val, list2_val.second - list1_val.second);
      // now, we just replace the node with its split version
      
      // Now, we need to figure out the local ids of these new nodes and put 
      // them in the list
      vector<int> new_node_ids(list1_split.size());
      for (int i = 0; i < list1_split.size(); i++)
      {
        
        long node_gid = tree->morton_to_gid(list1_split[i].first, list1_split[i].second);
        
        map<long, int>::iterator it = tree->letNodeMap.find( node_gid );
        assert(it != tree->letNodeMap.end());
        int node_lid = it->second;

        new_node_ids[i] = node_lid;
        
      }

      list1_copy.erase(list1_copy.begin());
      list1_copy.insert(list1_copy.begin(), new_node_ids.begin(), new_node_ids.end());
      
      // we'll handle the merging in the next iteration, since the index is 
      // still in the same place
      
      // end_point = min(list1_copy.size(), list2_copy.size());
      
    }
    else if (isAncestor(list2_val, list1_val))
    {
      
      // list1's node is an ancestor of list2's node -- so split list1's node
      vector<pair<long, int> > list2_split = SplitNode(list2_val, list1_val.second - list2_val.second);
      
      // Now, we need to figure out the local ids of these new nodes and put 
      // them in the list
      vector<int> new_node_ids(list2_split.size());
      for (int i = 0; i < list2_split.size(); i++)
      {
        
        long node_gid = tree->morton_to_gid(list2_split[i].first, list2_split[i].second);
        
        map<long, int>::iterator it = tree->letNodeMap.find( node_gid );
        assert(it != tree->letNodeMap.end());
        int node_lid = it->second;

        new_node_ids[i] = node_lid;
        
      }
      
      // now, we just replace the node with its split version
      list2_copy.erase(list2_copy.begin());
      list2_copy.insert(list2_copy.begin(), new_node_ids.begin(), new_node_ids.end());
      
      // we'll handle the merging in the next iteration, since the index is 
      // still in the same place
      
      // end_point = min(list1_copy.size(), list2_copy.size());
      
    }
    else {
      
      // in this case, the nodes are just distinct
      // We iterate past the one that is earlier in the traversal
      
      // Not the right comparison: need to write a function and compare the 
      // bits corrresponding to the higher level
      // They have to be different in these bits since there is no ancestor 
      // relationship
      // if (list1_ind < list2_ind)
      if (LessThanTreeOrder(list1_val, list2_val))
      {
        list1.push_back(list1_ind);
        list1_copy.erase(list1_copy.begin());
      }
      else {
        list2.push_back(list2_ind);
        list2_copy.erase(list2_copy.begin());
      }
      
    } // nodes are disjoint
    
  } // while we still have entries to process

  // Add back in whatever is left  
  if (list1_copy.size() > 0)
  {
    list1.insert(list1.end(), list1_copy.begin(), list1_copy.end());
  }
  
  if (list2_copy.size() > 0)
  {
    list2.insert(list2.end(), list2_copy.begin(), list2_copy.end());
  }
  double mtl_full_end = omp_get_wtime();
  
#pragma omp atomic 
  merge_tree_list_full_time += mtl_full_end - mtl_full_start;

} // MergeTargetInteractionListsFMM

template<class TKernel>
bool AskitAlg<TKernel>::LessThanTreeOrder(pair<long, int>& node1, pair<long, int>& node2)
{
  
  double lto_start = omp_get_wtime();
  
  // should just be able to compare MID because we assume that there is no 
  // ancestor relationship for now
  
  // Compare the rightmost (closest to the root) bit where the two 
  // MID's differ 

  // This string is 1 in all the positions where the two differ  
  long mid_xor = node1.first ^ node2.first;
  
  // Now, we compute the position at which the first set bit occurs
  // I think we are assuming that mid_xor is not zero here -- we have this 
  // if neither is an ancestor of the other
  int position = log2(mid_xor & ~(mid_xor-1));
  
  long val1 = node1.first & (1 << position);
  long val2 = node2.first & (1 << position);

  double lto_end = omp_get_wtime();
  
#pragma omp atomic
  less_than_tree_order_time += lto_end - lto_start;
  
  return val1 < val2;
  
}


// pair< Morton ID, level>
template<class TKernel>
bool AskitAlg<TKernel>::isAncestor(pair<long, int>& par, pair<long, int>& child)
{
  
  double ia_start = omp_get_wtime();
  
  // returns true if par is an ancestor of child, false otherwise
  bool res = false;
  
  if (par.second < child.second)
  {
    int level = par.second + 1;
    long mask = ~( (~0) << level);
    
    res = ((par.first & mask) == (child.first & mask));
    
  }
  
  double ia_end = omp_get_wtime();
  
#pragma omp atomic
  is_ancestor_time += ia_end - ia_start;
  
  return res;
  
} // isAncestor

// Performs num_splits splits of the given node and returns them in the vector
// Make sure to return the vector sorted in pre-order 
template<class TKernel>
vector<pair<long, int> > AskitAlg<TKernel>::SplitNode(pair<long, int>& val, int num_splits)
{
  
  double sn_start = omp_get_wtime();
  
  vector<pair<long, int> > output(1 << num_splits);
  
  // need all bit combinations from location val.second + 1 to val.second + 1 + 
  // num_splits
  
  int child_level = val.second + num_splits;
  
  for (int i = 0; i < (1 << num_splits); i++)
  {

    // set the bits to i
    long mask = (i << (val.second + 1));
    long mid = val.first | mask; 

    output[i] = make_pair(mid, child_level);
    
  } // loop over levels between parent and child

  double sn_end = omp_get_wtime();

#pragma omp atomic
  split_node_time += sn_end - sn_start;
  
  return output;
  
} // SplitNode




