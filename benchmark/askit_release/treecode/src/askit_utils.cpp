
#include "askit_utils.hpp"

#include <queue>

#define _OUTPUT_DATA_READ_ true

// The two inputs are the MID, level pairs for two tree nodes
// This is true if node1 comes before node2 in a postorder traversal
bool askit::LessThanTreeOrder(const pair<long, int>& node1, const pair<long, int>& node2)
{
  
  // Compare the rightmost (closest to the root) bit where the two 
  // MID's differ 


  // Need to check ancestor relationships here -- make it postorder
  if (isAncestor(node1, node2))
  {
    return false;
  }
  else if (isAncestor(node2, node1))
  {
    return true;
  }
  // handle the edge case where they have the same MID
  else if (node1.first == node2.first)
  {
    return node1.second < node2.second;
  }

  // This string is 1 in all the positions where the two differ  
  // It can't be zero, since the MID are checked if they are equal above
  long mid_xor = node1.first ^ node2.first;
  
  // Now, we compute the position at which the first set bit occurs
  int position = mid_xor & ~(mid_xor-1);

  long val1 = node1.first & position;
  long val2 = node2.first & position;

  return val1 < val2;
  
} // LessThanTreeOrder

// returns true if par is an ancestor of child, false otherwise
bool askit::isAncestor(const pair<long, int>& par, const pair<long, int>& child)
{
  
  bool res = false;
  
  if (par.second < child.second)
  {
    int level = par.second + 1;
    long mask = ~( (~0) << level);
    
    res = ((par.first & mask) == (child.first & mask));
  }
  
  return res;
  
} // isAncestor



void askit::ConstructTreeList(fks_ompNode* node, std::vector<fks_ompNode*>& list, 
    int& max_level)
{

  max_level = 0;

  int idx = 0;
  fks_ompNode *curr = node;
  queue<fks_ompNode *> myqueue;
  myqueue.push(curr);
  while( !myqueue.empty() ) {
    // dequeue the front node
    curr = myqueue.front();
    myqueue.pop();
    list.push_back(curr);
    // letNodeMap.insert(make_pair(curr->global_node_id, idx));
    // I'll use this to reference the list later -- Bill
    // I don't think this is used anywhere else
    curr->lnid = idx;
    idx++;

    if (curr->level > max_level)
      max_level = curr->level;
    
    // enqueue left child
    if(curr->leftNode != NULL)
      myqueue.push(curr->leftNode);

    // enqueue right child
    if(curr->rightNode != NULL)
      myqueue.push(curr->rightNode);
  }


  //
  //
  // int level = node->level;
  //
  // int my_rank;
  // MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  //
  // //std::cout << "Rank " << my_rank << " constructing tree list at level " << level << " local id "<<node->lnid<< "\n";
  //
  // // If we haven't seen a node this deep, add space for this level
  // if (list.size() < (1 << (level+1)))
  // {
  //   // initialize to NULL since the bottom level may not be complete
  //   list.resize(1 << (level+1), NULL);
  // }
  //
  // // Update the max level counter
  // if (level > max_level)
  //   max_level = level;
  //
  // // std::cout << "Tree list for node " << node->lnid << "\n";
  //
  // // use the local node id to index the list
  // long this_ind = (1 << level) - 1 + node->lnid;
  // list[this_ind] = node;
  // // in the single MPI rank case, the local ids of nodes are not set this
  // // way (because the tree construction never calls level_order_traversal())
  // // TODO: replace ConstructTreeList with level_order_traversal called on
  // // the omp root
  // // This is important to the test_askit_main.exe tests -- not sure why right now
  // node->lnid = this_ind;
  //
  // // handle children as well
  // if (node->leftNode != NULL)
  // {
  //   ConstructTreeList(node->leftNode, list, max_level);
  //   ConstructTreeList(node->rightNode, list, max_level);
  // }

} // ConstructTreeList

// Return true if the MID belongs to the node
bool askit::NodeOwnsPoint(fks_ompNode* node, long point_mid, int global_level)
{
  
  int level = node->level + global_level;
  
  // The root owns everything
  if (level == 0)
    return true;
  
  // Bo's code builds MID's from the right, so we care about the rightmost
  // level bits
  // one in the bits we care about, zero elsewhere
  long mask = (1 << (level+1)) - 1;
  
  long node_id = node->node_morton;
  
  // The node owns the point if the rightmost level bits are equal
  bool res = (node_id & mask) == (point_mid & mask);
  
  return res;
  
}



// Inverts the contents of target_to_node.
// 
// target_to_node is a list of nodes for each target.  Inverting it means that
// we compute a list of targets for each node, which is output in 
// node_to_target
void askit::InvertInteractionList(vector<vector<int> >& target_to_node, vector<vector<int> >& node_to_target)
{
  
  int num_targets = target_to_node.size();

  std::vector<std::vector<std::vector<int> > > thread_node_to_target(omp_get_max_threads());
  for (int i = 0; i < omp_get_max_threads(); i++)
  {
    thread_node_to_target[i].resize(node_to_target.size());
  }
  
  // loop over all query lids
#pragma omp parallel for  
  for (int i = 0; i < num_targets; i++)
  {
    int tid = omp_get_thread_num();
    std::vector<int>& this_target_to_node = target_to_node[i];
    std::vector<std::vector<int> >& my_node_to_target = thread_node_to_target[tid];
    // loop over nodes for this query lid
    for (int j = 0; j < this_target_to_node.size(); j++)
    {
      my_node_to_target[this_target_to_node[j]].push_back(i);
    } // loop over nodes interacting with targets

  } // parallel loop over target points
 
  // Now, we merge the lists for each thread
#pragma omp parallel for
  for (int i = 0; i < node_to_target.size(); i++)
  {
    // loop over the separate storage for each node
    for (int j = 0; j < thread_node_to_target.size(); j++)
    {
      node_to_target[i].insert(node_to_target[i].end(), 
        thread_node_to_target[j][i].begin(), thread_node_to_target[j][i].end());
    }
    
  } // parallel loop over node interactions
 
} // InvertInteractionList



askit::fksData* askit::ReadDistData(string& data_filename, string& charge_filename, 
  long glb_N, int d, bool is_binary_file)
{
  
  fksData* refData = new fksData;
  
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  int local_N;

  const char* ptrInputFile = data_filename.c_str();

  if(is_binary_file) {
      knn::mpi_binread(ptrInputFile, glb_N, d, local_N, refData->X, MPI_COMM_WORLD);
      refData->dim = d;
      refData->numof_points = local_N;
      
      if (_OUTPUT_DATA_READ_) {
        cout << "Rank " << rank << " read " << local_N << " with " << d << " features.\n";
        for (int i = 0; i < d; i++)
        {
          cout << refData->X[i] << ",";
        }
        cout << "\n";
      }
  }
  else {
      knn::mpi_dlmread(ptrInputFile, glb_N, d, refData->X, MPI_COMM_WORLD, false);
      local_N = refData->X.size() / d;
      refData->dim = d;
      refData->numof_points = local_N;
      
      if (_OUTPUT_DATA_READ_)
        cout << "Rank " << rank << " read " << local_N << " with " << d << " features.\n";
  
  }

  // Set up the global ID's
  long my_N = (long)local_N;
  long gid_offset;

  MPI_Scan( &my_N, &gid_offset, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD );
  gid_offset -= my_N;
  refData->gids.resize(local_N);

#pragma omp parallel for
  for(int i = 0; i < local_N; i++) {
      refData->gids[i] = gid_offset + (long)i;
  }
  
  
  // Read in or generate the charges
  refData->charges.resize(local_N);

  if (charge_filename.compare("ones") == 0)
  {

    double oosqrt_n = 1.0/sqrt(glb_N);

    #pragma omp parallel for
    for (int i = 0; i < local_N; i++)
    {
      refData->charges[i] = oosqrt_n;
    }

  }
  else if (charge_filename.compare("rand") == 0)
  {
    double loc_sum = 0.0;
    double glb_sum = 0.0;
    
#pragma omp parallel for reduction(+:loc_sum)
    for (int i = 0; i < local_N; i++)
    {
      refData->charges[i] = rand() / (double)RAND_MAX;
      loc_sum += refData->charges[i] * refData->charges[i];
    }

    MPI_Allreduce(&loc_sum, &glb_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    glb_sum = sqrt(glb_sum);

#pragma omp parallel for
    for (int i = 0; i < local_N; i++)
    {
      refData->charges[i] /= glb_sum;
    }

  }
  else if (charge_filename.compare("norm") == 0)
  {

    generateNormal(local_N, 1, refData->charges.data(), MPI_COMM_WORLD);
      
  }
  else if (charge_filename.compare("zeros") == 0)
  {
    refData->charges.assign(refData->charges.size(), 0.0);
  }
  else if (charge_filename.compare("debug") == 0)
  {
#pragma omp parallel for
    for (int i = 0; i < local_N; i++)
    {
      refData->charges[i] = (double)refData->gids[i];
    }
  }
  else if (is_binary_file)
  {
    // read the charges as a 1D vector
    knn::mpi_binread(charge_filename.c_str(), glb_N, 1, local_N, refData->charges, MPI_COMM_WORLD);
    
    if (_OUTPUT_DATA_READ_)
    {
      cout << "Rank " << rank << " read " << local_N << " charges.\n";
      // print the first 10 charges as a sanity check
      for (int i = 0; i < 10; i++)
      {
        cout << refData->charges[i] << ",";
      }
      cout << "\n";
    }
  }
  else {
    // not binary charges
    //long glb_numof_points = N;
    //knn::mpi_dlmread(charge_file, glb_numof_points, 1, refData->charges, comm, false);
    cout << "haven't implemented non-binary charge files\n";
  }
    
  return refData;
  
} // ReadDistData


// -- error_point_coordinates -- coordinates of all points used to estimate the 
// approximation error
// -- my_error_gids -- the global ids of error points that this MPI rank owns
// -- num_error_gids -- the number of error points per MPI rank
// -- error_gids -- the global ids of all the error check points 
void askit::CollectErrorCheckCoordinates(vector<double>& error_point_coordinates, 
  vector<long>& my_error_gids, vector<int>& num_error_gids, vector<long>& error_gids,
  fksData* refData, long global_N, int num_error_checks)
{
  
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  int d = refData->dim;
  
  error_point_coordinates.resize(num_error_checks * d);
  num_error_gids.resize(size);
  error_gids.resize(num_error_checks);
  
  long my_N = refData->numof_points;
  long gid_offset;
  MPI_Scan( &my_N, &gid_offset, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD );
  gid_offset -= my_N;  
  
  if (num_error_checks > 0)
  {
   
    vector<double> my_error_point_coordinates(num_error_checks * d);
   
    for (int i = 0; i < num_error_checks; i++)
    {
      error_gids[i] = (long)i * global_N / num_error_checks;
      int owner_rank = knn::home_rank(global_N, size, error_gids[i]);

      if(owner_rank == rank)
      {
        my_error_gids.push_back(error_gids[i]);
        // the local id is the global id minus the number of points that 
        // are owned by previous ranks
        int lid = error_gids[i] - gid_offset; 
        // cout << "lid " << lid << ", refid_offset " << refid_offset << "\n";
        memcpy(my_error_point_coordinates.data() + num_error_gids[rank]*d, 
          refData->X.data()+lid*d, d*sizeof(double));
      }
      num_error_gids[owner_rank]++;
      
    } // loop over error checks

    // now, every rank has the number and coordinates of the test points it 
    // owns -- we need to give all of these to every process
    // cout << "Computing displ\n";

    vector<int> recv_sizes(size);
    for (int i =0; i < size; i++)
    {
      recv_sizes[i] = num_error_gids[i] * d;
    }
    vector<int> displ(size);
    displ[0] = 0;
    for (int i = 1; i < size; i++)
    {
      displ[i] = displ[i-1] + recv_sizes[i-1];
    }
    
    // cout << "Doing AllgatherV\n";
    // now gather the coordinates

    // for (int r = 0; r < size; r++)
    // {
    //   MPI_Barrier(MPI_COMM_WORLD);
    //   if (r == rank)
    //   {
    //     cout << "Rank " << rank << " gathering " << num_error_gids[rank] << " points (" << error_point_coordinates.size() << " doubles.)\n";
    //   }
    // }

    MPI_Allgatherv(my_error_point_coordinates.data(), num_error_gids[rank]*d, MPI_DOUBLE, 
      error_point_coordinates.data(), recv_sizes.data(), displ.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    
  } // if we're checking error

} // CollectErrorCheckCoordinates



