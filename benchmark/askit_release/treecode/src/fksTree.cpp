#include "fksTree.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <queue>
#include <float.h>
#include <set>

#include "ompUtils.h"
#include "binTree.h"
#include "binQuery.h"
#include "blas.h"
#include "repartition.h"
#include "generator.h"
#include "clustering.h"
#include "direct_knn.h"
#include "rotation.h"
#include "parallelIO.h"
// #include "fileio.h"


using namespace std;
using namespace knn;
using namespace knn::repartition;
using namespace askit;

#define _DEBUG_ false
#define _OUTDEBUG_ false
#define _DIST_DEBUG_ false
#define _COLLECT_COORD_DEBUG_ false
#define _SHUFFLE_DEBUG_ false

#define _OUTPUT_ false
#define _DEBUG_TREE_ false
#define _DEBUG_LET_ false

#define _TREE_TIMING_ false
#define _DEBUG_TEST_POINT_ false

void askit::print_data(fksData *data, MPI_Comm comm)
{
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  MPI_Barrier(comm);

  int dim = data->dim;

  if(data->charges.size() == data->numof_points && data->mortons.size() == 0) {
    for(int k = 0; k < size; k++) {
      if(k == rank) {
        for(int i = 0; i < data->numof_points; i++) {
          cout<<"(rank "<<rank<<") "<<data->gids[i]<<": ("<<data->charges[i]<<") ";
          for(int j = 0; j < dim; j++)
            cout<<data->X[i*dim+j]<<" ";
          cout<<endl;
        }
        cout.flush();
      }
      MPI_Barrier(comm);
    }
  }
  else if(data->charges.size() == 0 && data->mortons.size() == data->numof_points) {
    for(int k = 0; k < size; k++) {
      if(k == rank) {
        for(int i = 0; i < data->numof_points; i++) {
          cout<<"(rank "<<rank<<") "<<data->gids[i]<<": ("<<data->mortons[i]<<") ";
          for(int j = 0; j < dim; j++)
            cout<<data->X[i*dim+j]<<" ";
          cout<<endl;
        }
        cout.flush();
      }
      MPI_Barrier(comm);
    }
  }
  else if(data->charges.size() == data->numof_points && data->mortons.size() == data->numof_points) {
    for(int k = 0; k < size; k++) {
      if(k == rank) {
        for(int i = 0; i < data->numof_points; i++) {
          cout<<"(rank "<<rank<<") "<<data->gids[i]
            <<": ("<<data->mortons[i]<<","<<data->charges[i]<<") ";
          for(int j = 0; j < dim; j++)
            cout<<data->X[i*dim+j]<<" ";
          cout<<endl;
        }
        cout.flush();
      }
      MPI_Barrier(comm);
    }
  }
  else {
    for(int k = 0; k < size; k++) {
      if(k == rank) {
        for(int i = 0; i < data->numof_points; i++) {
          cout<<"(rank "<<rank<<") "<<data->gids[i]<<": ";
          for(int j = 0; j < dim; j++)
            cout<<data->X[i*dim+j]<<" ";
          cout<<endl;
        }
        cout.flush();
      }
      MPI_Barrier(comm);
    }
  }
  if(rank == 0) cout<<endl;
}

void askit::print_data(fksData *data, map<long, int> &mymap, MPI_Comm comm)
{
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  MPI_Barrier(comm);

  int dim = data->dim;

  if(data->charges.size() == data->numof_points && data->mortons.size() == 0) {
    for(int k = 0; k < size; k++) {
      if(k == rank) {
        for(int i = 0; i < data->numof_points; i++) {
          cout<<"(rank "<<rank<<") "<<data->gids[i]<<": ("<<data->charges[i]<<") ";
          for(int j = 0; j < dim; j++)
            cout<<data->X[i*dim+j]<<" ";
          cout<<endl;
        }
        cout.flush();
      }
      MPI_Barrier(comm);
    }
  }
  else if(data->charges.size() == 0 && data->mortons.size() == data->numof_points) {
    for(int k = 0; k < size; k++) {
      if(k == rank) {
        for(int i = 0; i < data->numof_points; i++) {
          cout<<"(rank "<<rank<<") "<<data->gids[i]<<": ("<<data->mortons[i]<<") ";
          for(int j = 0; j < dim; j++)
            cout<<data->X[i*dim+j]<<" ";
          cout<<endl;
        }
        cout.flush();
      }
      MPI_Barrier(comm);
    }
  }
  else if(data->charges.size() == data->numof_points && data->mortons.size() == data->numof_points) {
    for(int k = 0; k < size; k++) {
      if(k == rank) {
        for(int i = 0; i < data->numof_points; i++) {
          cout<<"(rank "<<rank<<") "<<data->gids[i]
            <<": ("<<data->mortons[i]<<","<<data->charges[i]<<") ";
          for(int j = 0; j < dim; j++)
            cout<<data->X[i*dim+j]<<" ";
          cout<<endl;
        }
        cout.flush();
      }
      MPI_Barrier(comm);
    }
  }
  else {
    for(int k = 0; k < size; k++) {
      if(k == rank) {
        for(int i = 0; i < data->numof_points; i++) {
          cout<<"(rank "<<rank<<") "<<data->gids[i]<<": ";
          for(int j = 0; j < dim; j++)
            cout<<data->X[i*dim+j]<<" ";
          cout<<endl;
        }
        cout.flush();
      }
      MPI_Barrier(comm);
    }
  }

    for(int k = 0; k < size; k++) {
      if(k == rank) {
        for(map<long, int>::iterator it = mymap.begin(); it!= mymap.end(); it++) {
            cout<<"(rank "<<rank<<") "<<it->first<<" - "<<it->second<<endl;
        }
        cout.flush();
      }
      MPI_Barrier(comm);
    }

  if(rank == 0) cout<<endl;
}

void askit::print_data(binData *data, MPI_Comm comm)
{
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  MPI_Barrier(comm);

  int dim = data->dim;

  for(int k = 0; k < size; k++) {
    if(k == rank) {
      for(int i = 0; i < data->numof_points; i++) {
        cout<<"(rank "<<rank<<") "<<data->gids[i]<<": ";
        for(int j = 0; j < dim; j++)
          cout<<data->X[i*dim+j]<<" ";
        cout<<endl;
      }
      cout.flush();
    }
    MPI_Barrier(comm);
  }
  if(rank == 0) cout<<endl;
}

void askit::print_knn(vector<long> &queryIDs, vector< pair<double, long> > *knns, MPI_Comm comm)
{
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  MPI_Barrier(comm);

  int k = knns->size() / queryIDs.size();
  for(int r = 0; r < size; r++) {
    if(r == rank) {
      //cout<<"queryIDs.size = "<<queryIDs.size()<<", knns.size = "<<knns->size()<<endl;
      for(int i = 0; i < queryIDs.size(); i++) {
        cout<<"(rank "<<rank<<") "<<queryIDs[i]<<": ";
        for(int j = 0; j < k; j++)
          cout<<(*knns)[i*k+j].second<<" ";
        cout<<endl;
      }
      cout.flush();
    }
    MPI_Barrier(comm);
  }
  if(rank == 0) cout<<endl;
}

void askit::print_set(set< triple<long, long, int> > &myset, MPI_Comm comm)
{
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  MPI_Barrier(comm);

  for(int k = 0; k < size; k++) {
    if(k == rank) {
      for(set< triple<long, long, int> >::iterator it = myset.begin(); it != myset.end(); it++) {
        cout<<"(rank "<<rank<<") gid="<<it->first<<", morton="<<it->second<<", level="<<it->third<<endl;
      }
      cout.flush();
    }
    MPI_Barrier(comm);
  }
  if(rank == 0) cout<<endl;
}

void askit::print_tree_single(fks_ompNode *inNode)
{
    if(inNode == NULL)
        return;

    if(NULL == inNode->leftNode && NULL == inNode->rightNode) {
        //cout<<"node "<<inNode->lnid<<" ["<<inNode->node_morton<<"] "
        cout<<"node "<<inNode->global_node_id<<" ["<<inNode->node_morton<<"] "
            <<" @l("<<inNode->level<<")"
            <<": {"<<inNode->leaf_point_gids.size()<<"} ";
        for(int i = 0; i < inNode->leaf_point_gids.size(); i++)
            cout<<inNode->leaf_point_gids[i]<<" ";
        if(inNode->skeletons != NULL) {
            cout<<"#skel: ";
            for(int i = 0; i < inNode->skeletons->numof_points; i++)
                cout<<inNode->skeletons->gids[i]<<"("<<inNode->skeletons->charges[i]<<") ";
        }
        cout<<endl;
        return;
    }
    else {
        //cout<<"node "<<inNode->lnid<<" ["<<inNode->node_morton<<"] "
        cout<<"node "<<inNode->global_node_id<<" ["<<inNode->node_morton<<"] "
            <<" @l("<<inNode->level<<") ";
        if(inNode->skeletons != NULL) {
            cout<<"#skel: ";
            for(int i = 0; i < inNode->skeletons->numof_points; i++)
                cout<<inNode->skeletons->gids[i]<<"("<<inNode->skeletons->charges[i]<<") ";
        }
        cout<<endl;
        print_tree_single(inNode->leftNode);
        print_tree_single(inNode->rightNode);
    }
}

void askit::print_tree(fks_ompNode *inNode, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    for(int r = 0; r < size; r++) {
        if(r == rank) {
            cout<<" ----          rank "<<rank<<"         ---- "<<endl;
            print_tree_single(inNode);
        }
        cout.flush();
        MPI_Barrier(comm);
    }
    if(rank == 0) cout<<endl;
}


void askit::print_let_node_list(vector<fks_ompNode*>& list)
{
  
  int rank, size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  for(int r = 0; r < size; r++) {
      if(r == rank) {
          cout<<" ----          rank "<<rank<<"         ---- "<<endl;
          print_let_node_list_single(list);
      }
      cout.flush();
      MPI_Barrier(MPI_COMM_WORLD);
  }
  if(rank == 0) cout<<endl;
  
}


void askit::print_let_node_list_single(vector<fks_ompNode*>& list)
{

  for (int i = 0; i < list.size(); i++)
  {
    fks_ompNode* inNode = list[i];
    
    if (inNode->leftNode == NULL)
    {
      cout<<"node "<<inNode->global_node_id<<" ["<<inNode->node_morton<<"] "
          <<" @l("<<inNode->level<<")"
          <<": {"<<inNode->leaf_point_gids.size()<<"} ";
      for(int i = 0; i < inNode->leaf_point_gids.size(); i++)
          cout<<inNode->leaf_point_gids[i]<<" ";
      if(inNode->skeletons != NULL) {
          cout<<"#skel: ";
          for(int i = 0; i < inNode->skeletons->numof_points; i++)
              cout<<inNode->skeletons->gids[i]<<" ";
      }
      cout<<endl;
    }
    else {
      cout<<"node "<<inNode->global_node_id<<" ["<<inNode->node_morton<<"] "
          <<" @l("<<inNode->level<<") ";
      if(inNode->skeletons != NULL) {
          cout<<"#skel: ";
          for(int i = 0; i < inNode->skeletons->numof_points; i++)
              cout<<inNode->skeletons->gids[i]<<" ";
      }
      cout<<endl;
    }
  }
}



// ======== fks_mpiNode member functions =======
void fks_mpiNode::Insert(fks_mpiNode *in_parent, int maxp, int maxLevel, int minCommSize,
MPI_Comm inComm, binData *inData)
{
  int worldsize, worldrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldsize);

  // input checks
  int numof_kids = 2;
  assert( maxp > 1 );
  assert( maxLevel >= 0 && maxLevel <= options.max_max_treelevel);

  comm = inComm;
  int size, rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank); // needed to print information
  int dim = inData->dim;
  MPI_Allreduce(&(inData->numof_points), &Nglobal, 1, MPI_INT, MPI_SUM, comm);

  // set the number of points owned by this node
  num_points_owned = Nglobal;

  vector<double> &X = inData->X;
  vector<long> &gids= inData->gids;

  double start_t = omp_get_wtime();

  // Initializations
  int its_child_id = 0;
  int n_over_p = Nglobal / size;

  knn::repartition::loadBalance_arbitraryN(inData->X, inData->gids,
  inData->numof_points, inData->dim, inData->numof_points, comm);

  if(_TREE_TIMING_) {
    MPI_Barrier(comm);
    if(rank == 0) cout<<"fksTree->Insert(): loadbalance time "<<omp_get_wtime()-start_t<<endl;
    start_t = omp_get_wtime();
  }


  if(in_parent!=NULL)  {
    level = in_parent->level + 1;
    fks_parent = in_parent;
    its_child_id = chid;
  }

  // BASE CASE TO TERMINATE RECURSION
  if (size <= minCommSize || level == maxLevel || Nglobal <= maxp) {
    // This will get deleted in find_knn_for_leaves, so it needs to be allocated here
    data = new binData;
    data->Copy(inData);

    if(_TREE_TIMING_) {
        MPI_Barrier(comm);
        if(rank == 0) cout<<"fksTree->Insert(): copy leaf node time "<<omp_get_wtime()-start_t<<endl;
        start_t = omp_get_wtime();
    }


    if(_DEBUG_TREE_ && false) {
      MPI_Barrier(MPI_COMM_WORLD);
      if(worldrank == 0) cout<<"leaf data: "<<endl;
      print_data(inData, MPI_COMM_WORLD);
    }

    return;
  }   // end of base case

  int numof_clusters = 2;
  vector<int> point_to_cluster_membership(inData->numof_points);
  vector<int> local_numof_points_per_cluster(numof_clusters);
  vector<int> global_numof_points_per_cluster(numof_clusters);

  coord_mv = -1;
  proj.resize(dim);
  mtreeSplitter(&X[0], inData->numof_points, dim, &(proj[0]), median,
  &(point_to_cluster_membership[0]),
  &(local_numof_points_per_cluster[0]),
  &(global_numof_points_per_cluster[0]),
  comm);

  if(_TREE_TIMING_) {
    MPI_Barrier(comm);
    if(rank == 0) cout<<"fksTree->Insert(): mtree splitter time "<<omp_get_wtime()-start_t<<endl;
    start_t = omp_get_wtime();
  }

  int my_rank_color;
  rank_colors.resize(size);
  if(size % 2 == 0) {  // Even split
    my_rank_color = (rank < size/2) ? 0 : 1;
    for(int i = 0; i < size; i++)
      rank_colors[i] = (i < size/2) ? 0 : 1;
  }
  else {
    my_rank_color = (rank <= size/2) ? 0 : 1;
    for(int i = 0; i < size; i++)
      rank_colors[i] = (i <= size/2) ? 0 : 1;
  }

  pre_all2all(&(gids[0]), &(point_to_cluster_membership[0]), &(X[0]),
  (long)(inData->numof_points), dim);


  if(_TREE_TIMING_) {
    MPI_Barrier(comm);
    if(rank == 0) cout<<"fksTree->Insert(): pre_all2all time "<<omp_get_wtime()-start_t<<endl;
    start_t = omp_get_wtime();
  }


  int newN = tree_repartition_arbitraryN(inData->gids, inData->X,
  inData->numof_points, &(point_to_cluster_membership[0]),
  &(rank_colors[0]), dim, comm);
  inData->numof_points = newN;

  if(_TREE_TIMING_) {
    MPI_Barrier(comm);
    if(rank == 0) cout<<"fksTree->Insert(): tree repartition time "<<omp_get_wtime()-start_t<<endl;
    start_t = omp_get_wtime();
  }

  if(_DEBUG_TREE_ && false) {
    MPI_Barrier(MPI_COMM_WORLD);
    if(worldrank == 0) cout<<"after tree_repartition data: "<<endl;
    print_data(inData, MPI_COMM_WORLD);
  }


  //6. create new communicator
  MPI_Comm new_comm = MPI_COMM_NULL;
  if( MPI_SUCCESS != MPI_Comm_split( comm, my_rank_color, rank, &new_comm) )
    assert(NULL);
  assert(new_comm != MPI_COMM_NULL);

  //7. Create new node and insert new data
  fks_kid = new fks_mpiNode(its_child_id);
  if(my_rank_color == 0) {    // left child
    fks_kid->node_morton = ( node_morton & (~(1 << (level+1))) );
  }
  else {
    fks_kid->node_morton = ( node_morton | (1 << (level+1)) );
  }
  fks_kid->options.hypertree = options.hypertree;
  fks_kid->options.flag_r = options.flag_r;
  fks_kid->options.flag_c = options.flag_c;
  fks_kid->options.pruning_verbose = options.pruning_verbose;
  fks_kid->options.timing_verbose = options.timing_verbose;
  fks_kid->options.splitter = options.splitter;
  fks_kid->options.debug_verbose = options.debug_verbose;
  fks_kid->Insert(this, maxp, maxLevel, minCommSize, new_comm, inData);
};

fks_mpiNode::~fks_mpiNode()
{
  if(this->fks_parent != NULL) {  // if this is not root
    MPI_Barrier(comm);
    MPI_Comm_free(&comm);
  }

  if(NULL == this->fks_kid && NULL != this->data) {   // if leaf
    delete this->data;
    this->data = NULL;
    if(this->skeletons != NULL) {
      delete this->skeletons;
      this->skeletons = NULL;
    }
  }
  else {
    if(this->skeletons != NULL) {
      delete this->skeletons;
      this->skeletons = NULL;
    }
    delete this->fks_kid;
  }
}


// ======== fksTree member functions =========


// ------------------- parallel tree consturction ---------------
//
// ----------------------------------------------------------------

fksTree::~fksTree()
{
  if(inProcKNN != NULL) {
    delete inProcKNN;
    inProcKNN = NULL;
  }

  if(root_omp != NULL && root_let == NULL) {
    delete root_omp;
    root_omp = NULL;
  }

  if(root_let != NULL) {
    delete root_let;
    root_let = NULL;
  }

  if(root_mpi != NULL) {
    delete root_mpi;
    root_mpi = NULL;
  }

  // inProcData pionts to root_mpi --- > leaf->data, so it has already been released.
  if(inProcData != NULL) {
    delete inProcData;
    inProcData = NULL;
  }

  if(inProcTestData != NULL) {
    delete inProcTestData;
    inProcTestData = NULL;
  }

  if(inProcTestKNN != NULL) {
    delete inProcTestKNN;
    inProcTestKNN = NULL;
  }

}


int fksTree::pos(long gid)
{
  int lid = -1;
  map<long, int>::iterator it = inProcMap.find( gid );
  if(it != inProcMap.end() ) {
    lid = it->second;
    return lid;
  }
  return lid;
}


// charges should be sorted in order of global id
void fksTree::exchange_charges(int numof_points, double *charges,
int numof_request_points, long *request_gids,
double *request_charges,
MPI_Comm comm)
{
  int size, rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  long glb_numof_points, max_numof_points, dummy_numof_points;
  dummy_numof_points = numof_points;
  MPI_Allreduce(&dummy_numof_points, &max_numof_points, 1, MPI_LONG, MPI_MAX, comm);
  MPI_Allreduce(&dummy_numof_points, &glb_numof_points, 1, MPI_LONG, MPI_SUM, comm);

  int *send_count = new int [size];
  int *recv_count = new int [size];
  int *send_disp = new int [size];
  int *recv_disp = new int [size];

  if(_DEBUG_TREE_) {
    for(int r = 0; r < size; r++) {
      if(r == rank) {
        cout<<"(rank "<<rank<<") request_gids: ";
        for(int i = 0; i < numof_request_points; i++)
          cout<<request_gids[i]<<" ";
        cout<<endl;
      }
      cout.flush();
      MPI_Barrier(comm);
    }
  }

  pair<long, int> *clone_request_gids = new pair<long, int> [numof_request_points];
  //vector< pair<long, int> > clone_request_gids(numof_request_points);
  for(int i = 0; i < numof_request_points; i++) {
    clone_request_gids[i].first = request_gids[i];
    clone_request_gids[i].second = i;
  }
  //omp_par::merge_sort(clone_request_gids.begin(), clone_request_gids.end());
  omp_par::merge_sort(clone_request_gids, clone_request_gids+numof_request_points);

  if(_DEBUG_TREE_) {
    for(int r = 0; r < size; r++) {
      if(r == rank) {
        cout<<"(rank "<<rank<<") clone_request_gids: ";
        for(int i = 0; i < numof_request_points; i++)
          cout<<"("<<clone_request_gids[i].first<<"-"<<clone_request_gids[i].second<<") ";
        cout<<endl;
      }
      cout.flush();
      MPI_Barrier(comm);
    }
  }

  // .1 send request_gids to home process
  memset(send_count, 0, sizeof(int)*size);
  for(int i = 0; i < numof_request_points; i++) {
    int target_rank = knn::home_rank( glb_numof_points, size, clone_request_gids[i].first );
    send_count[ target_rank ]++;
  }
  MPI_Alltoall( send_count, 1, MPI_INT, recv_count, 1, MPI_INT, comm );
  int n_total_recv = recv_count[0];
  send_disp[0] = 0;
  recv_disp[0] = 0;
  for(int l = 1; l < size; l++) {
    n_total_recv += recv_count[l];
    send_disp[l] = send_disp[l-1] + send_count[l-1];
    recv_disp[l] = recv_disp[l-1] + recv_count[l-1];
  }
  long *recv_request_gids = new long [n_total_recv];
  long *send_request_gids = new long [numof_request_points];
  for(int i = 0; i < numof_request_points; i++)
    send_request_gids[i] = clone_request_gids[i].first;
  MPI_Alltoallv( send_request_gids, send_count, send_disp, MPI_LONG,
  recv_request_gids, recv_count, recv_disp, MPI_LONG, comm);

  if(_DEBUG_TREE_) {
    for(int r = 0; r < size; r++) {
      if(r == rank) {
        cout<<"(rank "<<rank<<") recv_request_gids: ";
        for(int i = 0; i < n_total_recv; i++)
          cout<<recv_request_gids[i]<<" ";
        cout<<endl;
      }
      cout.flush();
      MPI_Barrier(comm);
    }
  }

  // .2 prepare data
  double *tosend_request_charges = new double [n_total_recv];
  long offset;
  MPI_Scan(&dummy_numof_points, &offset, 1, MPI_LONG, MPI_SUM, comm);
  offset -= dummy_numof_points;
#pragma omp parallel if (n_total_recv > 1000)
  {
#pragma omp for
    for(int i = 0; i < n_total_recv; i++) {
      int loc_id = recv_request_gids[i] - offset;
      tosend_request_charges[i] = charges[loc_id];
    }
  }

  if(_DEBUG_TREE_) {
    for(int r = 0; r < size; r++) {
      if(r == rank) {
        cout<<"(rank "<<rank<<") tosend_request_charges: ";
        for(int i = 0; i < n_total_recv; i++)
          cout<<tosend_request_charges[i]<<" ";
        cout<<endl;
      }
      cout.flush();
      MPI_Barrier(comm);
    }
  }

  // .3 send requested charges
  double *recv_request_charges = new double [numof_request_points];
  for(int i = 0; i < size; i++) {
    int tmp = send_count[i];
    send_count[i] = recv_count[i];
    recv_count[i] = tmp;
  }
  send_disp[0] = 0;
  recv_disp[0] = 0;
  for(int l = 1; l < size; l++) {
    send_disp[l] = send_disp[l-1] + send_count[l-1];
    recv_disp[l] = recv_disp[l-1] + recv_count[l-1];
  }
  MPI_Alltoallv(tosend_request_charges, send_count, send_disp, MPI_DOUBLE,
  recv_request_charges, recv_count, recv_disp, MPI_DOUBLE, comm);

  if(_DEBUG_TREE_) {
    for(int r = 0; r < size; r++) {
      if(r == rank) {
        cout<<"(rank "<<rank<<") recv_request_charges: ";
        for(int i = 0; i < numof_request_points; i++)
          cout<<recv_request_charges[i]<<" ";
        cout<<endl;
      }
      cout.flush();
      MPI_Barrier(comm);
    }
  }

  // .4 copy charges back
  #pragma omp parallel for
  for(int i = 0; i < numof_request_points; i++)
    request_charges[clone_request_gids[i].second] = recv_request_charges[i];


  if(_DEBUG_TREE_) {
    for(int r = 0; r < size; r++) {
      if(r == rank) {
        cout<<"(rank "<<rank<<") request_charges: ";
        for(int i = 0; i < numof_request_points; i++)
          cout<<"("<<clone_request_gids[i].first<<" "<<request_charges[i]<<") ";
        cout<<endl;
      }
      cout.flush();
      MPI_Barrier(comm);
    }
  }





  delete [] clone_request_gids;
  delete [] recv_request_charges;
  delete [] tosend_request_charges;
  delete [] send_request_gids;
  delete [] recv_request_gids;
  delete [] send_count;
  delete [] recv_count;
  delete [] send_disp;
  delete [] recv_disp;

}


// this function is used to shuffle the values back to its original process accord. to global id
// i.e., values after shuffle, should be evenly distributed accd to their global id
void fksTree::shuffle_back(int numof_points, double *values, long *gids,
                           int shuffled_numof_points, double *shuffled_values, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    long dummy_glb_numof_points, dummy_numof_points = numof_points;
    MPI_Allreduce(&dummy_numof_points, &dummy_glb_numof_points, 1, MPI_LONG, MPI_SUM, comm);

    pair<long, double> *clone_values = new pair<long, double> [numof_points];
    for(int i = 0; i < numof_points; i++) {
        clone_values[i].first = gids[i];
        clone_values[i].second = values[i];
    }
    omp_par::merge_sort(clone_values, clone_values+numof_points);

    if(_SHUFFLE_DEBUG_ & 0) {
        for(int r = 0; r < size; r++) {
            if(r == rank) {
                cout<<"dummy_glb_numof_points = "<<dummy_glb_numof_points<<endl;
                for(int i = 0; i < numof_points; i++) {
                    cout<<"rank "<<rank<<": "<<clone_values[i].first<<" "<<clone_values[i].second<<" ";
                }
                cout<<endl;
            }
            MPI_Barrier(comm);
            cout.flush();
        }
    }

    int *send_count = new int [size];
    int *recv_count = new int [size];
    int *send_disp = new int [size];
    int *recv_disp = new int [size];

    memset(send_count, 0, sizeof(int)*size);
    for(int i = 0; i < numof_points; i++) {
        //int target_rank = knn::home_rank(glbN, size, clone_values[i].first);
        int target_rank = knn::home_rank(dummy_glb_numof_points, size, clone_values[i].first);
        send_count[ target_rank ]++;
    }
    MPI_Alltoall( send_count, 1, MPI_INT, recv_count, 1, MPI_INT, comm );
    int n_total_recv = recv_count[0];
    send_disp[0] = 0;
    recv_disp[0] = 0;
    for(int l = 1; l < size; l++) {
        n_total_recv += recv_count[l];
        send_disp[l] = send_disp[l-1] + send_count[l-1];
        recv_disp[l] = recv_disp[l-1] + recv_count[l-1];
    }

    if(_SHUFFLE_DEBUG_) {
        cout<<"rank "<<rank<<": n_total_recv = "<<n_total_recv
            <<", shuffled_numof_points = "<<shuffled_numof_points
            <<endl;
    }

    assert(n_total_recv == shuffled_numof_points);

    pair<long, double> *shuffled_pairs = new pair<long, double> [shuffled_numof_points];
    MPI_Datatype msgtype;
    MPI_Type_contiguous(sizeof(pair<long, double>), MPI_BYTE, &msgtype);
    MPI_Type_commit(&msgtype);
    MPI_Alltoallv(clone_values, send_count, send_disp, msgtype,
                  shuffled_pairs, recv_count, recv_disp, msgtype, comm);

    if(_SHUFFLE_DEBUG_) {
        if(rank == 0) cout<<endl<<endl;
        if(rank == 0) cout<<"before shuffle charges:"<<endl;
        MPI_Barrier(comm);
        for(int r = 0; r < size; r++) {
            if(r == rank) {
                cout<<"rank "<<rank<<": ";
                for(int i = 0; i < numof_points; i++) {
                    int target_rank = knn::home_rank(glbN, size, clone_values[i].first);
                    cout<<"("<<clone_values[i].first
                        <<", "<<target_rank<<") "
                        <<clone_values[i].second<<"  ";
                }
                cout<<endl;
                cout.flush();
            }
            MPI_Barrier(comm);
        }
        if(rank == 0) cout<<endl;

        if(rank == 0) cout<<"after shuffle charges:"<<endl;
        MPI_Barrier(comm);
        for(int r = 0; r < size; r++) {
            if(r == rank) {
                cout<<"rank "<<rank<<": ";
                for(int i = 0; i < shuffled_numof_points; i++) {
                    cout<<"("<<shuffled_pairs[i].first<<") "
                        <<shuffled_pairs[i].second<<" ";
                }
                cout<<endl;
                cout.flush();
            }
            MPI_Barrier(comm);
        }
        if(rank == 0) cout<<endl<<endl;
    }

    omp_par::merge_sort(shuffled_pairs, shuffled_pairs+shuffled_numof_points);
    for(int i = 0; i < shuffled_numof_points; i++)
        shuffled_values[i] = shuffled_pairs[i].second;

    delete [] send_count;
    delete [] recv_count;
    delete [] send_disp;
    delete [] recv_disp;
    delete [] clone_values;
    delete [] shuffled_pairs;
}


// this function is used to shuffle the values back to its original process accord. to global id
// i.e., values after shuffle, should be evenly distributed accd to their global id
void fksTree::shuffle_back(int numof_points, int dim, long *gids, double *values,
                           int shuffled_numof_points, double *shuffled_values, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    long dummy_glb_numof_points, dummy_numof_points = numof_points;
    MPI_Allreduce(&dummy_numof_points, &dummy_glb_numof_points, 1, MPI_LONG, MPI_SUM, comm);

    pair<long, int> *clone_idx = new pair<long, int> [numof_points];
    for(int i = 0; i < numof_points; i++) {
        clone_idx[i].first = gids[i];
        clone_idx[i].second = i;
    }
    omp_par::merge_sort(clone_idx, clone_idx+numof_points);

    if(_SHUFFLE_DEBUG_ & 0) {
        for(int r = 0; r < size; r++) {
            if(r == rank) {
                cout<<"dummy_glb_numof_points = "<<dummy_glb_numof_points<<endl;
                for(int i = 0; i < numof_points; i++) {
                    cout<<"rank "<<rank<<": "<<clone_idx[i].first<<" "<<clone_idx[i].second<<" ";
                }
                cout<<endl;
            }
            MPI_Barrier(comm);
            cout.flush();
        }
    }

    int *send_count = new int [size];
    int *recv_count = new int [size];
    int *send_disp = new int [size];
    int *recv_disp = new int [size];

    memset(send_count, 0, sizeof(int)*size);
    for(int i = 0; i < numof_points; i++) {
        int target_rank = knn::home_rank(dummy_glb_numof_points, size, clone_idx[i].first);
        send_count[ target_rank ]++;
    }
    MPI_Alltoall( send_count, 1, MPI_INT, recv_count, 1, MPI_INT, comm );
    int n_total_recv = recv_count[0];
    send_disp[0] = 0;
    recv_disp[0] = 0;
    for(int l = 1; l < size; l++) {
        n_total_recv += recv_count[l];
        send_disp[l] = send_disp[l-1] + send_count[l-1];
        recv_disp[l] = recv_disp[l-1] + recv_count[l-1];
    }

    if(_SHUFFLE_DEBUG_) {
        cout<<"rank "<<rank<<": n_total_recv = "<<n_total_recv
            <<", shuffled_numof_points = "<<shuffled_numof_points
            <<endl;
    }

    assert(n_total_recv == shuffled_numof_points);

    pair<long, int> *shuffled_gids = new pair<long, int> [shuffled_numof_points];
    MPI_Datatype msgtype;
    MPI_Type_contiguous(sizeof(pair<long, int>), MPI_BYTE, &msgtype);
    MPI_Type_commit(&msgtype);
    MPI_Alltoallv(clone_idx, send_count, send_disp, msgtype,
                  shuffled_gids, recv_count, recv_disp, msgtype, comm);
    for(int i = 0; i < shuffled_numof_points; i++) {
        shuffled_gids[i].second = i;
    }

    double *send_values = new double [numof_points*dim];
    for(int i = 0; i < numof_points; i++) {
        memcpy(send_values+i*dim, values+clone_idx[i].second*dim, sizeof(double)*dim);
    }

    double *shuffled_values_copy = new double [shuffled_numof_points*dim];
    for(int i = 0; i < size; i++) {
        send_count[i] *= dim;
        recv_count[i] *= dim;
        send_disp[i] *= dim;
        recv_disp[i] *= dim;
    }
    MPI_Alltoallv(send_values, send_count, send_disp, MPI_DOUBLE,
                  shuffled_values_copy, recv_count, recv_disp, MPI_DOUBLE, comm);

    omp_par::merge_sort(shuffled_gids, shuffled_gids+shuffled_numof_points);
    for(int i = 0; i < shuffled_numof_points; i++) {
        memcpy(shuffled_values+i*dim, shuffled_values_copy+shuffled_gids[i].second*dim, sizeof(double)*dim);
    }

    delete [] send_count;
    delete [] recv_count;
    delete [] send_disp;
    delete [] recv_disp;
    delete [] clone_idx;
    delete [] shuffled_gids;
    delete [] send_values;
    delete [] shuffled_values_copy;

}



// I cannot directly change inData because later I need to gather knn based on the original order
void fksTree::build(fksData *inData, void *ctx)
{
  fksCtx *user = (fksCtx*) ctx;

  fksData *inData_clone = new fksData();
  inData_clone->numof_points = inData->numof_points;
  inData_clone->dim = inData->dim;
  inData_clone->X.resize(inData->X.size());
  memcpy(inData_clone->X.data(), inData->X.data(), sizeof(double)*inData->X.size());
  inData_clone->gids.resize(inData->gids.size());
  memcpy(inData_clone->gids.data(), inData->gids.data(), sizeof(long)*inData->gids.size());
  inData_clone->charges.resize(inData->charges.size());
  memcpy(inData_clone->charges.data(), inData->charges.data(), sizeof(double)*inData->charges.size());

  int rank, size;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  long dummy_numof_points = inData->numof_points;
  MPI_Allreduce(&dummy_numof_points, &glbN, 1, MPI_LONG, MPI_SUM, comm);

  double start_t, total_t;

  if(_OUTPUT_ || _TREE_TIMING_) {
    MPI_Barrier(comm);
    if(rank == 0) cout<<"fksTree::build(): enter ... "<<endl;
    start_t = omp_get_wtime();
  }

  if(_DEBUG_TREE_) {
    if(rank == 0) cout<<"input inData: "<<endl;
    print_data( inData, comm );
  }

  int dim = inData->dim;

  // .1 build distributed memory tree
  root_mpi = new fks_mpiNode();
  root_mpi->options.hypertree = 1;        // point-wise data exchange, alltoall if set to 0
  root_mpi->options.splitter = "rsmt";    // use metric tree splitting
  root_mpi->options.flag_r = 0;           // do not rotate
  root_mpi->options.flag_c = 0;           // do not use in fksTree
  root_mpi->Insert(NULL, user->fks_mppn, user->fks_maxLevel, user->minCommSize, comm, inData_clone);

  if(_OUTPUT_ || _TREE_TIMING_) {
    MPI_Barrier(comm);
    if(rank == 0) cout<<"fksTree::build(): root_mpi done, t = "<<omp_get_wtime()-start_t<<endl;
    start_t = omp_get_wtime();
  }

  if(_DEBUG_TREE_) {
    MPI_Barrier(comm);
    if(rank == 0) cout<<"after distributed tree done, inData_clone: "<<endl;
    print_data( inData_clone, comm );
  }

  // .2  inProcData = root_mpi->...->leaf
  fks_mpiNode *curr = root_mpi;
  while(curr->fks_kid != NULL) curr = curr->fks_kid;
  numof_points_of_dist_leaf = curr->data->numof_points;
  inProcData = new fksData();
  inProcData->Copy(curr->data);
  inProcData->mortons.resize(inProcData->numof_points);
#pragma omp parallel for
  for(int i = 0; i < inProcData->numof_points; i++) {
    inProcData->mortons[i] = curr->node_morton;
  }
  // build inProcMap
  for(int i = 0; i < inProcData->numof_points; i++) {
    inProcMap.insert(make_pair(inProcData->gids[i], i));
  }

  depth_mpi = curr->level+1;

  // TODO: release memory of curr->data->X

  if(_DEBUG_TREE_) {
    MPI_Barrier(comm);
    if(rank == 0) cout<<"inProcData: "<<endl;
    print_data( inProcData, comm );
  }

  // .3 exchange charges of inProcData
  inProcData->charges.resize(inProcData->numof_points);
  exchange_charges(inData->numof_points, &(inData->charges[0]),
  inProcData->numof_points, &(inProcData->gids[0]), &(inProcData->charges[0]), comm);

  if(_OUTPUT_ || _TREE_TIMING_) {
    MPI_Barrier(comm);
    if(rank == 0) cout<<"fksTree::build(): exchange charges done, t = "<<omp_get_wtime()-start_t<<endl;
    start_t = omp_get_wtime();
  }

  if(_DEBUG_TREE_) {
    MPI_Barrier(comm);
    if(rank == 0) cout<<"after exchange charges, inData_clone: "<<endl;
    print_data( inData_clone, comm );

    MPI_Barrier(comm);
    if(rank == 0) cout<<"after exchange charges, inProcData: "<<endl;
    print_data( inProcData, comm );
  }


  // .4 build shared memory tree
  vector<long> *active_set = new vector<long>();
  active_set->resize(numof_points_of_dist_leaf);
  #pragma omp parallel for
  for(int i = 0; i < numof_points_of_dist_leaf; i++)
    (*active_set)[i] = (long)i;
  root_omp = new fks_ompNode();
  root_omp->node_morton = curr->node_morton;
  root_omp->insert(NULL, inProcData, active_set, curr->level, user->fks_mppn, user->fks_maxLevel-curr->level);

  fks_ompNode *curr_omp = root_omp;
  while(curr_omp->leftNode != NULL)
    curr_omp = curr_omp->leftNode;
  numof_points_of_share_leaf = curr_omp->leaf_point_gids.size();
  depth_omp = curr_omp->level+1;
  depth_let = depth_omp + depth_mpi - 1;

  if(_OUTPUT_ || _TREE_TIMING_) {
    MPI_Barrier(comm);
    if(rank == 0) cout<<"fksTree::build(): build shared memory tree done, t = "<<omp_get_wtime()-start_t<<endl;
    start_t = omp_get_wtime();
  }

  if(_DEBUG_TREE_) {
    MPI_Barrier(comm);
    if(rank == 0) cout<<"after shared memeory tree, inProcData: "<<endl;
    print_data( inProcData, comm );
  }

  delete inData_clone;

  // .5 it is stupid, but we need to put morton id into indata,
  // distribute inProcData->morton to inData->morton accordingly
  int *send_count = new int [size];
  int *recv_count = new int [size];
  int *send_disp = new int [size];
  int *recv_disp = new int [size];
  // pair<gid, morton>
  pair<long, long> *send_mortons = new pair<long, long> [numof_points_of_dist_leaf];
  #pragma omp parallel for
  for(int i = 0; i < numof_points_of_dist_leaf; i++) {
    send_mortons[i].first = inProcData->gids[i];
    send_mortons[i].second = inProcData->mortons[i];
  }
  omp_par::merge_sort(send_mortons, send_mortons+numof_points_of_dist_leaf);
  memset(send_count, 0, sizeof(int)*size);
  for(int i = 0; i < numof_points_of_dist_leaf; i++) {
    int target_rank = home_rank(glbN, size, send_mortons[i].first);
    send_count[ target_rank ]++;
  }
  MPI_Alltoall( send_count, 1, MPI_INT, recv_count, 1, MPI_INT, comm );
  int n_total_recv = recv_count[0];
  send_disp[0] = 0;
  recv_disp[0] = 0;
  for(int l = 1; l < size; l++) {
    n_total_recv += recv_count[l];
    send_disp[l] = send_disp[l-1] + send_count[l-1];
    recv_disp[l] = recv_disp[l-1] + recv_count[l-1];
  }

  assert(n_total_recv == inData->numof_points);

  pair<long, long> *recv_mortons = new pair<long, long> [n_total_recv];
  MPI_Datatype msgtype;
  MPI_Type_contiguous(sizeof(pair<long, long>), MPI_BYTE, &msgtype);
  MPI_Type_commit(&msgtype);
  MPI_Alltoallv(send_mortons, send_count, send_disp, msgtype,
  recv_mortons, recv_count, recv_disp, msgtype, comm);
  long offset;
  MPI_Scan(&dummy_numof_points, &offset, 1, MPI_LONG, MPI_SUM, comm);
  offset -= dummy_numof_points;
  inData->mortons.resize(inData->numof_points);
  for(int i = 0; i < n_total_recv; i++) {
    int loc_id = recv_mortons[i].first - offset;
    inData->mortons[loc_id] = recv_mortons[i].second;
  }

  delete [] send_mortons;
  delete [] recv_mortons;
  delete [] send_count;
  delete [] recv_count;
  delete [] send_disp;
  delete [] recv_disp;

  if(_OUTPUT_ || _TREE_TIMING_) {
    MPI_Barrier(comm);
    if(rank == 0) cout<<"fksTree::build(): move morton ids back done, t = "<<omp_get_wtime()-start_t<<endl;
    start_t = omp_get_wtime();
  }

  if(_DEBUG_TREE_) {
    MPI_Barrier(comm);
    if(rank == 0) cout<<"move morton ids back, inData: "<<endl;
    print_data( inData, comm );
  }

  long dummy_np = inProcData->numof_points, max_np, min_np, avg_np;
  MPI_Allreduce(&dummy_np, &max_np, 1, MPI_LONG, MPI_MAX, comm);
  MPI_Allreduce(&dummy_np, &min_np, 1, MPI_LONG, MPI_MIN, comm);
  MPI_Allreduce(&dummy_np, &avg_np, 1, MPI_LONG, MPI_SUM, comm);
  avg_np /= size;
  if(rank == 0 && _DEBUG_TREE_) {
      cout<<"after build tree, \tinProcData: \tmin = "<<min_np
          <<", \tmax = "<<max_np<<", \tavg = "<<avg_np<<endl;
  }

}


void fksTree::knn(fksData *inData, vector< pair<double, long> > *inKNN)
{
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (size > 1)
  {
    // cout << "calling exchange basic knn \n";
    exchange_basic_knn(inData, inKNN);
    // cout << "calling exchange knn data \n";
    exchange_knn_data(inData);
  }
  else {
    inProcKNN = inKNN; // just copies the pointer
    numof_neighbors_not_in_dist_leaf = 0; // no other nodes
  }
  // cout << "finished with knn\n";

  long dummy_np = inProcData->numof_points, max_np, min_np, avg_np;
  MPI_Allreduce(&dummy_np, &max_np, 1, MPI_LONG, MPI_MAX, comm);
  MPI_Allreduce(&dummy_np, &min_np, 1, MPI_LONG, MPI_MIN, comm);
  MPI_Allreduce(&dummy_np, &avg_np, 1, MPI_LONG, MPI_SUM, comm);
  avg_np /= size;
  if(rank == 0 && _DEBUG_TREE_) {
      cout<<"after gather knn, \tinProcData: \tmin = "<<min_np
          <<", \tmax = "<<max_np<<", \tavg = "<<avg_np<<endl;
  }

}


void fksTree::exchange_basic_knn(fksData *inData, vector< pair<double, long> > *inKNN)
{
  int rank, size;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  long glb_numof_points, dummy_numof_points;
  dummy_numof_points = inData->numof_points;
  MPI_Allreduce(&dummy_numof_points, &glb_numof_points, 1, MPI_LONG, MPI_SUM, comm);
  int k = inKNN->size() / inData->numof_points;

  int *send_count = new int [size];
  int *recv_count = new int [size];
  int *send_disp = new int [size];
  int *recv_disp = new int [size];

  // 0. sort request gids
  pair<long, int> *request_gids = new pair<long, int> [numof_points_of_dist_leaf];
  for(int i = 0; i < numof_points_of_dist_leaf; i++) {
    request_gids[i].first = inProcData->gids[i];
    request_gids[i].second = i;
  }
  omp_par::merge_sort(request_gids, request_gids+numof_points_of_dist_leaf);

  // .1 send request_gids to home process
  memset(send_count, 0, sizeof(int)*size);
  for(int i = 0; i < numof_points_of_dist_leaf; i++) {
    int target_rank = knn::home_rank( glb_numof_points, size, request_gids[i].first );
    send_count[ target_rank ]++;
  }
  MPI_Alltoall( send_count, 1, MPI_INT, recv_count, 1, MPI_INT, comm );
  int n_total_recv = recv_count[0];
  send_disp[0] = 0;
  recv_disp[0] = 0;
  for(int l = 1; l < size; l++) {
    n_total_recv += recv_count[l];
    send_disp[l] = send_disp[l-1] + send_count[l-1];
    recv_disp[l] = recv_disp[l-1] + recv_count[l-1];
  }
  long *recv_request_gids = new long [n_total_recv];
  long *send_request_gids = new long [numof_points_of_dist_leaf];
  for(int i = 0; i < numof_points_of_dist_leaf; i++)
    send_request_gids[i] = request_gids[i].first;
  MPI_Alltoallv( send_request_gids, send_count, send_disp, MPI_LONG,
  recv_request_gids, recv_count, recv_disp, MPI_LONG, comm);

  // .2 prepare data
  pair<double, long> *tosend_request_knns = new pair<double, long> [n_total_recv*k];
  long offset;
  MPI_Scan(&dummy_numof_points, &offset, 1, MPI_LONG, MPI_SUM, comm);
  offset -= dummy_numof_points;
  #pragma omp parallel if (n_total_recv > 1000)
  {
    #pragma omp for
    for(int i = 0; i < n_total_recv; i++) {
      int loc_id = recv_request_gids[i] - offset;
      for(int j = 0; j < k; j++) {
        tosend_request_knns[i*k+j] = (*inKNN)[loc_id*k+j];
      }
    }
  }

  // .3 send requested knn
  pair<double, long> *recv_request_knns = new pair<double, long> [numof_points_of_dist_leaf*k];
  for(int i = 0; i < size; i++) {
    int tmp = send_count[i];
    send_count[i] = recv_count[i]*k;
    recv_count[i] = tmp*k;
  }
  send_disp[0] = 0;
  recv_disp[0] = 0;
  for(int l = 1; l < size; l++) {
    send_disp[l] = send_disp[l-1] + send_count[l-1];
    recv_disp[l] = recv_disp[l-1] + recv_count[l-1];
  }
  MPI_Datatype msgtype;
  MPI_Type_contiguous(sizeof(pair<double, long>), MPI_BYTE, &msgtype);
  MPI_Type_commit(&msgtype);
  MPI_Alltoallv(tosend_request_knns, send_count, send_disp, msgtype,
  recv_request_knns, recv_count, recv_disp, msgtype, comm);

  // .4 copy charges back
  inProcKNN = new vector< pair<double, long> >;
  inProcKNN->resize(numof_points_of_dist_leaf*k);
#pragma omp parallel for
  for(int i = 0; i < numof_points_of_dist_leaf; i++) {
    for(int j = 0; j < k; j++)
      (*inProcKNN)[request_gids[i].second*k+j] = recv_request_knns[i*k+j];
  }

  delete [] recv_request_knns;
  delete [] tosend_request_knns;
  delete [] send_request_gids;
  delete [] recv_request_gids;
  delete [] request_gids;
  delete [] send_count;
  delete [] recv_count;
  delete [] send_disp;
  delete [] recv_disp;

}


void fksTree::exchange_knn_data(fksData *inData)
{
  int rank, size;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  long glb_numof_points, dummy_numof_points;
  dummy_numof_points = inData->numof_points;
  MPI_Allreduce(&dummy_numof_points, &glb_numof_points, 1, MPI_LONG, MPI_SUM, comm);
  int k = inProcKNN->size() / numof_points_of_dist_leaf;
  int dim = inData->dim;

  int *send_count = new int [size];
  int *recv_count = new int [size];
  int *send_disp = new int [size];
  int *recv_disp = new int [size];

  // .1 find unique nn gids
  vector<long> request_gids(inProcKNN->size());
#pragma omp parallel for
  for(int i = 0; i < inProcKNN->size(); i++)
    request_gids[i] = (*inProcKNN)[i].second;
  omp_par::merge_sort(request_gids.begin(), request_gids.end());
  vector<long>::iterator it = unique(request_gids.begin(), request_gids.end());
  request_gids.resize(it-request_gids.begin());

  // .2 remove gids already in proc
  vector<long> send_request_gids;
  send_request_gids.reserve(request_gids.size());
  for(int i = 0; i < request_gids.size(); i++) {
    if(pos(request_gids[i]) == -1) {
      send_request_gids.push_back(request_gids[i]);
    }
  }

  if(_DEBUG_TREE_) {
    if(rank == 0) cout<<"exchange knn: "<<endl;
    for(int r = 0; r < size; r++) {
      if(r == rank) {
        cout<<"send_request_gids: ";
        for(int i = 0; i < send_request_gids.size(); i++)
          cout<<send_request_gids[i]<<" ";
        cout<<endl;
      }
      cout.flush();
      MPI_Barrier(comm);
    }
  }

  // .3 send copy_request_gids to home process
  memset(send_count, 0, sizeof(int)*size);
  for(int i = 0; i < send_request_gids.size(); i++) {
    int target_rank = knn::home_rank( glb_numof_points, size, send_request_gids[i] );
    send_count[ target_rank ]++;
  }
  MPI_Alltoall( send_count, 1, MPI_INT, recv_count, 1, MPI_INT, comm );
  int n_total_recv = recv_count[0];
  send_disp[0] = 0;
  recv_disp[0] = 0;
  for(int l = 1; l < size; l++) {
    n_total_recv += recv_count[l];
    send_disp[l] = send_disp[l-1] + send_count[l-1];
    recv_disp[l] = recv_disp[l-1] + recv_count[l-1];
  }
  vector<long> recv_request_gids(n_total_recv);
  MPI_Alltoallv( send_request_gids.data(), send_count, send_disp, MPI_LONG,
  recv_request_gids.data(), recv_count, recv_disp, MPI_LONG, comm);


  if(_DEBUG_TREE_) {
    if(rank == 0) cout<<"exchange knn: "<<endl;
    for(int r = 0; r < size; r++) {
      if(r == rank) {
        cout<<"recv_request_gids: ";
        for(int i = 0; i < recv_request_gids.size(); i++)
          cout<<recv_request_gids[i]<<" ";
        cout<<endl;
      }
      cout.flush();
      MPI_Barrier(comm);
    }
  }

  // .4 prepare data
  double *tosend_request_knns = new double [n_total_recv*(dim+2)];
  long offset;
  MPI_Scan(&dummy_numof_points, &offset, 1, MPI_LONG, MPI_SUM, comm);
  offset -= dummy_numof_points;
#pragma omp parallel if (n_total_recv > 1000)
  {
#pragma omp for
    for(int i = 0; i < n_total_recv; i++) {
      int loc_id = recv_request_gids[i] - offset;
      tosend_request_knns[i*(dim+2)+0] = inData->charges[loc_id];
      tosend_request_knns[i*(dim+2)+1] = (double)inData->mortons[loc_id];
      memcpy( (tosend_request_knns+i*(dim+2)+2), &(inData->X[loc_id*dim]), sizeof(double)*dim );
    }
  }

  // .5 send requested knn
  double *recv_request_knns = new double [send_request_gids.size()*(dim+2)];
  for(int i = 0; i < size; i++) {
    int tmp = send_count[i];
    send_count[i] = recv_count[i]*(dim+2);
    recv_count[i] = tmp*(dim+2);
  }
  send_disp[0] = 0;
  recv_disp[0] = 0;
  for(int l = 1; l < size; l++) {
    send_disp[l] = send_disp[l-1] + send_count[l-1];
    recv_disp[l] = recv_disp[l-1] + recv_count[l-1];
  }
  MPI_Alltoallv(tosend_request_knns, send_count, send_disp, MPI_DOUBLE,
  recv_request_knns, recv_count, recv_disp, MPI_DOUBLE, comm);

  // .6 put nn into inProcData
  int tmpn = inProcData->numof_points;
  inProcData->X.resize( (send_request_gids.size()+tmpn)*dim );
  inProcData->gids.resize( (send_request_gids.size()+tmpn) );
  inProcData->mortons.resize( (send_request_gids.size()+tmpn) );
  inProcData->charges.resize( (send_request_gids.size()+tmpn) );
  inProcData->numof_points = send_request_gids.size()+tmpn;
#pragma omp parallel for
  for(int i = 0; i < send_request_gids.size(); i++) {
    inProcData->charges[tmpn+i] = recv_request_knns[i*(dim+2)+0];
    if (inProcData->charges[tmpn+i] > 200.0)
    {
      cout << "\n\n\n BAD CHARGE IN exchange_knn_data \n\n\n";
    }
    inProcData->mortons[tmpn+i] = recv_request_knns[i*(dim+2)+1];
    inProcData->gids[tmpn+i] = send_request_gids[i];
    memcpy( &(inProcData->X[(tmpn+i)*dim]), &(recv_request_knns[i*(dim+2)+2]), sizeof(double)*dim);
  }

  for(int i = 0; i < send_request_gids.size(); i++) {
    inProcMap.insert(make_pair(send_request_gids[i], tmpn+i));
  }
  numof_neighbors_not_in_dist_leaf = send_request_gids.size();

  delete [] recv_request_knns;
  delete [] tosend_request_knns;
  delete [] send_count;
  delete [] recv_count;
  delete [] send_disp;
  delete [] recv_disp;
}


bool fksTree::getLeafData(fks_ompNode *ompLeaf, fksData *leaf)
{
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  int nlocal = ompLeaf->leaf_point_gids.size();
  int dim = inProcData->dim;

  if(nlocal == 0) {
    cout<<"rank "<<rank<<" leaf_point_gids is empty"<<endl;
    return false;
  }

  leaf->dim = dim;
  leaf->numof_points = nlocal;
  leaf->X.resize( nlocal*dim );
  leaf->gids.resize( nlocal );
  leaf->mortons.resize( nlocal );
  leaf->charges.resize( nlocal );
  #pragma omp parallel for
  for(int i = 0; i < nlocal; i++) {
    int idx = pos(ompLeaf->leaf_point_gids[i]);
    leaf->gids[i] = inProcData->gids[idx];
    leaf->mortons[i] = inProcData->mortons[idx];
    leaf->charges[i] = inProcData->charges[idx];
    memcpy( &(leaf->X[i*dim]), &(inProcData->X[idx*dim]), sizeof(double)*dim );
  }

  return true;
}


// sampling_num_neighbors is the number we want
bool fksTree::getLeafexclNN(fks_ompNode *ompLeaf, fksData *exclNN, int sampling_num_neighbors)
{
  // IMPORTANT: I'm putting a hack in here to get the distances of the neighbors
  // along with the neighbors themselves.  Since we never care about the charges
  // of a leaf's exclusive nearest neighbors, I'm going to use charges to store
  // these -- Bill

  int rank;
  MPI_Comm_rank(comm, &rank);


  // this is the full k -- total_num_neighbors in the ASKIT class
  int k = inProcKNN->size() / numof_points_of_dist_leaf;
  // this the portion of the neighbors we'll use for pruning
  int pruning_num_neighbors = k - sampling_num_neighbors;

  // if (ompLeaf->node_morton == 0 && ompLeaf->level == 10)
  // {
  //
  //   cout << "k: " << k << ", samp k: " << sampling_num_neighbors << ", prune k: " << pruning_num_neighbors << "\n";
  //
  // }
  
  int nlocal = ompLeaf->leaf_point_gids.size();

  if(nlocal == 0)
  {
    cout<<"(exclknn) rank "<<rank<<": leaf_point_gids empty"<<endl;
    return false;
  }

  // -1. get nn for all leaf points
  vector<pair<double, long> > all_knn_ids(nlocal*sampling_num_neighbors);
  #pragma omp parallel for
  for(int i = 0; i < nlocal; i++) {
    int idx = pos(ompLeaf->leaf_point_gids[i]);
    for(int j = 0; j < sampling_num_neighbors; j++) {
      // skip over the neighbors used for pruning
      long tmpind = idx * (long)k + (long)(pruning_num_neighbors + j);
      all_knn_ids[i*sampling_num_neighbors+j] = (*inProcKNN)[tmpind];
    }
  }

  // IMPORTANT: I was using comp_second_first here to try to sort by distance
  // as well -- this fails for some reason I don't understand yet.
  
  // The problem with just calling this is that I will lose track of which
  // copy of the neighbor was closest (i.e. had the smallest distance)
  // I need this to know which ones to keep.
  
  // Using this one (from askit) seems to work
  // sort(all_knn_ids.begin(), all_knn_ids.end(), comp_second());
  sort(all_knn_ids.begin(), all_knn_ids.end(), comp_neighbor_unique_sort());
  vector<pair<double, long> >::iterator it = unique(all_knn_ids.begin(), all_knn_ids.end(), equ_second());
  all_knn_ids.resize(it - all_knn_ids.begin());

  // just printing this for the matlab experiments
  // if (ompLeaf->node_morton == 0 && ompLeaf->level == 10)
  // {
  //
  //   cout << "\nSampling Neighbors:\n";
  //   for (int i = 0; i < all_knn_ids.size(); i++)
  //   {
  //     cout << all_knn_ids[i].second << " ";
  //   }
  //   cout << "\n\n";
  //
  //   // cout << "\nAll Neighbor dists:\n";
  //   // for (int i = 0; i < all_knn_ids.size(); i++)
  //   // {
  //   //   cout << all_knn_ids[i].first << " ";
  //   // }
  //   // cout << "\n\n";
  //
  // }



  // Also need to exclude those in the first half of any point's list
  // -2. exclude those in leaf

  // We do this so that we can recover the old (1 defn of k) behavior easily
  // In this case, we'll set this value to 1 so that we only include the 
  // self neighbor -- this is the old behavior
  if (pruning_num_neighbors == 0)
    pruning_num_neighbors = 1;

  vector<pair<double, long> > leaf_ids(nlocal*pruning_num_neighbors);
#pragma omp parallel for
  for(int i = 0; i < nlocal; i++) {

    // we need to get the first k/2 neighbors for each point
    long gid = pos(ompLeaf->leaf_point_gids[i]);
    for (int j = 0; j < pruning_num_neighbors; j++)
    {
      long ind = gid * (long)k + (long)j;
      leaf_ids[i*pruning_num_neighbors + j] = (*inProcKNN)[ind];
      // leaf_ids[i*pruning_num_neighbors + j].first = 0;
    }
  }
  sort(leaf_ids.begin(), leaf_ids.end(), comp_second());
  // there are a lot of duplicates, so we'll remove them here
  it = unique(leaf_ids.begin(), leaf_ids.end(), equ_second());
  leaf_ids.resize(it - leaf_ids.begin());

  // just printing this for the matlab experiments
  // if (ompLeaf->node_morton == 0 && ompLeaf->level == 10)
  // {
  //
  //   cout << "\nPruning Neighbors:\n";
  //   for (int i = 0; i < leaf_ids.size(); i++)
  //   {
  //     cout << leaf_ids[i].second << " ";
  //   }
  //   cout << "\n\n";
  //
  // }


  // vector<pair<double, long> >::iterator it;
  vector<pair<double, long> > excl_ids(all_knn_ids.size()+1);
  // Remove the points owned by this leaf from the list of neighbors
  it = set_difference(all_knn_ids.begin(), all_knn_ids.end(),
  leaf_ids.begin(), leaf_ids.end(), excl_ids.begin(), comp_second());
  // Resize to the number we found
  excl_ids.resize(it-excl_ids.begin());


  // Now, we have to sort them by distance
  // comp first should be the default
  sort(excl_ids.begin(), excl_ids.end());

  // -3. copy data
  int nexcl = excl_ids.size();
  int dim = inProcData->dim;
  exclNN->dim = dim;
  exclNN->numof_points = nexcl;
  exclNN->X.resize( nexcl*dim );
  exclNN->gids.resize( nexcl );
  exclNN->mortons.resize( nexcl );
  exclNN->charges.resize( nexcl );

  // if (ompLeaf->node_morton == 0 && ompLeaf->level == 10)
  // {
  //
  //   cout << "\nFinal Sampling Neighbors:\n";
  //   for (int i = 0; i < excl_ids.size(); i++)
  //   {
  //     cout << excl_ids[i].second << " ";
  //   }
  //   cout << "\n\n";
  //
  // }


#pragma omp parallel for
  for(int i = 0; i < nexcl; i++) {
    int idx = pos(excl_ids[i].second);
    exclNN->gids[i] = inProcData->gids[idx];
    exclNN->mortons[i] = inProcData->mortons[idx];
    // charges hold the distances
    exclNN->charges[i] = excl_ids[i].first;
    // TODO: take this out, no longer needed
    memcpy( &(exclNN->X[i*dim]), &(inProcData->X[idx*dim]), sizeof(double)*dim );
  }

  return true;
}


// ------------------- distributed skeletonization ---------------
// currently, only works for the case p equals to *power of 2*
// ----------------------------------------------------------------
// max_size is the truncation size for neighbor lists
// sampling_num_neighbors -- the number of neighbors that will be used for 
// the sampling (in the split k case)
void fksTree::mergeNNList(fks_mpiNode *inNode, int max_size)
{

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // cout << "Rank " << rank << " calling mergeNNList with node " << inNode << "\n";
  // cout << "Rank " << rank << " fks_kid: " << inNode->fks_kid << "\n";

  if(inNode == NULL) return;

  int local_rank, local_size;
  MPI_Comm_rank(inNode->comm, &local_rank);
  MPI_Comm_size(inNode->comm, &local_size);

  double start_t;


  MPI_Request send_req, recv_req;
  MPI_Status stat;
  int send_tag = inNode->node_morton;
  int recv_tag = inNode->node_morton;

  if (_DIST_DEBUG_)
  {
    cout << "rank " << rank << " node: " << inNode << ", kid: " << inNode->fks_kid << "\n";
  }

  int kid_size = 0;
  MPI_Comm_size(inNode->fks_kid->comm, &kid_size);
  int dest_rank = 0;
  int source_rank = dest_rank + kid_size;
  

  int k = inProcKNN->size() / numof_points_of_dist_leaf;
  int dim = inProcData->dim;

  // if internal node
  // .1 collet nn id of both child
  // ** I would assume all the nn neighbors are stroed
  // ** in the first process of the child node

  // .1.1 message size

  int send_msg_size = inNode->fks_kid->excl_knn_of_this_node.size();
  int recv_msg_size = 0;
  
  // also need to collect the pruning list 
  vector<long> right_pruning_neighbor_list;
  int right_pruning_neighbor_list_size = inNode->fks_kid->pruning_neighbor_list.size();

  if(_DIST_DEBUG_) {
    for(int r = 0; r < size; r++) {
      if(r == rank) {
        cout<<"rank "<<rank<<", local_rank "<<local_rank<<": before send-recv msg size "
          <<" send_msg_size = "<<send_msg_size
            <<" recv_msg_size = "<<recv_msg_size
              <<" source_rank = "<<source_rank
                <<" dest_rank = "<<dest_rank
                  <<endl;
      }
      cout.flush();
      MPI_Barrier(comm);
    }
  }

  if(local_rank == source_rank) {
    MPI_Send(&send_msg_size, 1, MPI_INT, dest_rank, send_tag, inNode->comm);
  }
  if(local_rank == dest_rank) {
    MPI_Recv(&recv_msg_size, 1, MPI_INT, source_rank, recv_tag, inNode->comm, &stat);
  }

  if(local_rank == source_rank) {
    MPI_Send(&right_pruning_neighbor_list_size, 1, MPI_INT, dest_rank, send_tag, inNode->comm);
  }
  if(local_rank == dest_rank) {
    MPI_Recv(&right_pruning_neighbor_list_size, 1, MPI_INT, source_rank, recv_tag, inNode->comm, &stat);
  }

  if(_DIST_DEBUG_) {
    for(int r = 0; r < size; r++) {
      if(r == rank) {
        cout<<"rank "<<rank<<", local_rank "<<local_rank<<": after send-recv msg size "
          <<" send_msg_size = "<<send_msg_size
            <<" recv_msg_size = "<<recv_msg_size
              <<" source_rank = "<<source_rank
                <<" dest_rank = "<<dest_rank
                  <<endl;
      }
      cout.flush();
      MPI_Barrier(comm);
    }
  }

  // .1.2 collect nn of right child on local rank 0
  MPI_Datatype msgtype;
  MPI_Type_contiguous(sizeof(triple<long, long, double>), MPI_BYTE, &msgtype);
  MPI_Type_commit(&msgtype);

  vector< triple<long, long, double> > recv_buf;
  if(local_rank == source_rank) {
    MPI_Send(&(inNode->fks_kid->excl_knn_of_this_node[0]), send_msg_size, msgtype, dest_rank, send_tag, inNode->comm);
  }
  if(local_rank == dest_rank) {
    recv_buf.resize(recv_msg_size);
    MPI_Recv(&(recv_buf[0]), recv_msg_size, msgtype, source_rank, recv_tag, inNode->comm, &stat);
  }

  // exchanging the pruning list
  if(local_rank == source_rank) {
    MPI_Send(&(inNode->fks_kid->pruning_neighbor_list[0]), right_pruning_neighbor_list_size, MPI_LONG, dest_rank, send_tag, inNode->comm);
  }
  if(local_rank == dest_rank) {
    right_pruning_neighbor_list.resize(right_pruning_neighbor_list_size);
    MPI_Recv(&(right_pruning_neighbor_list[0]), right_pruning_neighbor_list_size, MPI_LONG, source_rank, recv_tag, inNode->comm, &stat);
  }
  
  if(_OUTPUT_) {
    MPI_Barrier(comm);
    start_t = omp_get_wtime();
  }

  if(_DIST_DEBUG_) {
    for(int r = 0; r < size; r++) {
      if(r == rank) {
        cout<<"rank "<<rank<<", local_rank "<<local_rank
            <<": recv_buf after send-recv: ";
        for(int i = 0; i < recv_buf.size(); i++)
          cout<<recv_buf[i].first<<"["<<recv_buf[i].second<<"] - "<<recv_buf[i].third<<",  ";
        cout<<endl;
      }
      cout.flush();
      MPI_Barrier(comm);
    }
  }

  // .2 remove duplicates in nn
  if(local_rank == dest_rank) {
    recv_buf.insert(recv_buf.end(), inNode->fks_kid->excl_knn_of_this_node.begin(), inNode->fks_kid->excl_knn_of_this_node.end());
    if(recv_buf.size() > 1) {
      omp_par::merge_sort(recv_buf.begin(), recv_buf.end(),
      triple<long, long, double>::firstLess);
      vector< triple<long, long, double> >::iterator it;
      it = unique(recv_buf.begin(), recv_buf.end(), triple<long, long, double>::firstEqual);
      recv_buf.resize(it - recv_buf.begin());
      
    }
      // now, remove elements of the pruning lists
      
    // merge the two pruning lists
    inNode->pruning_neighbor_list = inNode->fks_kid->pruning_neighbor_list;
    inNode->pruning_neighbor_list.insert(inNode->pruning_neighbor_list.end(), 
      right_pruning_neighbor_list.begin(), right_pruning_neighbor_list.end());
    
    // sort the merged pruning list and remove duplicates
    sort(inNode->pruning_neighbor_list.begin(), inNode->pruning_neighbor_list.end());
    vector<long>::iterator long_it = unique(inNode->pruning_neighbor_list.begin(), inNode->pruning_neighbor_list.end());
    inNode->pruning_neighbor_list.resize(long_it - inNode->pruning_neighbor_list.begin());

    // make a vector of matching type so we can use set difference
    vector<triple<long, long, double> > prune_list(inNode->pruning_neighbor_list.size());
    for (int i = 0; i < prune_list.size(); i++)
    {
      prune_list[i] = triple<long, long, double>(inNode->pruning_neighbor_list[i], 0, 0.0);
    }
    
    vector<triple<long, long, double> > diff_out(recv_buf.size());
    vector<triple<long, long, double> >::iterator it;
    
    it = set_difference(recv_buf.begin(), recv_buf.end(), prune_list.begin(), 
      prune_list.end(), diff_out.begin(), triple<long, long, double>::firstEqual);
    
    recv_buf.assign(diff_out.begin(), it);
    
  } // on the rank that's receiving the neighbors

  if(_DIST_DEBUG_) {
    for(int r = 0; r < size; r++) {
      if(r == rank) {
        cout<<"rank "<<rank<<", local_rank "<<local_rank
          <<", inNode morton "<<inNode->node_morton
            <<": recv_buf after merge and remove dup: ";
        for(int i = 0; i < recv_buf.size(); i++)
          cout<<recv_buf[i].first<<"["<<recv_buf[i].second<<"] - "<<recv_buf[i].third<<",  ";
        cout<<endl;
      }
      cout.flush();
      MPI_Barrier(comm);
    }
  }

  // .3 get excl nn of this node
  inNode->excl_knn_of_this_node.reserve(recv_buf.size());
  for(int i = 0; i < recv_buf.size(); i++) {
    if( !belong2Node(inNode->node_morton, inNode->level, recv_buf[i].second) ) {
      inNode->excl_knn_of_this_node.push_back(recv_buf[i]);
    }
  }

  // parallel merge sort seems to fail if the list is empty
  // and we only need to sort if we're going to truncate the list
  if (inNode->excl_knn_of_this_node.size() > max_size)
  {
    omp_par::merge_sort(inNode->excl_knn_of_this_node.begin(),
    inNode->excl_knn_of_this_node.end(), triple<long, long, double>::thirdLess);
    // std::sort(inNode->excl_knn_of_this_node.begin(),
//     inNode->excl_knn_of_this_node.end(), triple<long, long, double>::thirdLess);
    if(inNode->excl_knn_of_this_node.size() > max_size) {
      inNode->excl_knn_of_this_node.resize(max_size);
    }
  }
  
  vector<long> send_knn_ids;
  send_knn_ids.reserve( inNode->excl_knn_of_this_node.size() );
  for(int i = 0; i < inNode->excl_knn_of_this_node.size(); i++) {
    long target_gid = inNode->excl_knn_of_this_node[i].first;
    if( inProcMap.find(target_gid) == inProcMap.end() ) {
      send_knn_ids.push_back(target_gid);
    }
  }

  // .4 request data of those excl nn from right child
  send_msg_size = send_knn_ids.size();
  recv_msg_size = 0;
  source_rank = 0;
  dest_rank = source_rank + kid_size;

  if(local_rank == source_rank) {
    MPI_Send(&send_msg_size, 1, MPI_INT, dest_rank, send_tag, inNode->comm);
  }
  if(local_rank == dest_rank) {
    MPI_Recv(&recv_msg_size, 1, MPI_INT, source_rank, recv_tag, inNode->comm, &stat);
  }

  vector<long> request_knn_ids;
  if(local_rank == source_rank) {
    MPI_Send(&(send_knn_ids[0]), send_msg_size, MPI_LONG, dest_rank, send_tag, inNode->comm);
  }
  if(local_rank == dest_rank) {
    request_knn_ids.resize(recv_msg_size);
    MPI_Recv(&request_knn_ids[0], recv_msg_size, MPI_LONG, source_rank, recv_tag, inNode->comm, &stat);
  }

  dest_rank = 0;
  source_rank = dest_rank + kid_size;

  int span = dim + 3;
  // coord, charge, gid, morton
  double *recv_data = new double [send_msg_size*(dim+3)];
  double *send_data = new double [request_knn_ids.size()*(dim+3)];
  if(local_rank == source_rank) {
    send_data = new double [request_knn_ids.size()*(dim+3)];
    for(int i = 0; i < request_knn_ids.size(); i++) {
      int lid = pos(request_knn_ids[i]);
      if(lid >= 0) {
        memcpy(send_data+i*span, &(inProcData->X[lid*dim]), sizeof(double)*dim);
        send_data[i*span+dim] = inProcData->charges[lid];
        send_data[i*span+dim+1] = (double)inProcData->gids[lid];
        send_data[i*span+dim+2] = (double)inProcData->mortons[lid];
      }
    }
  }

  send_msg_size = request_knn_ids.size();
  recv_msg_size = send_knn_ids.size();
  if(local_rank == source_rank) {
    MPI_Send(send_data, send_msg_size*span, MPI_DOUBLE, dest_rank, send_tag, inNode->comm);
  }
  if(local_rank == dest_rank) {
    MPI_Recv(recv_data, recv_msg_size*span, MPI_DOUBLE, source_rank, recv_tag, inNode->comm, &stat);
  }


  if(_DIST_DEBUG_) {
    for(int r = 0; r < size; r++) {
      if(r == rank) {
        cout<<"rank "<<rank<<", local_rank "<<local_rank<<": recv_data "<<endl;
        for(int i = 0; i < recv_msg_size; i++) {
          cout<<"\t - "<<recv_data[i*span+dim+1]<<" ["<<recv_data[i*span+dim+2]<<"]"
            <<" ("<<recv_data[i*span+dim]<<") ";
          for(int j = 0; j < dim; j++)
            cout<<recv_data[i*span+j]<<" ";
          cout<<endl;
        }
      }
      cout.flush();
      MPI_Barrier(comm);
    }
  }

  // .5 write them into inProcData
  if(local_rank == dest_rank) {
    int offset = inProcData->numof_points;
    inProcData->numof_points = offset + recv_msg_size;
    inProcData->X.resize((offset+recv_msg_size)*dim);
    inProcData->charges.resize(offset+recv_msg_size);
    inProcData->mortons.resize(offset+recv_msg_size);
    inProcData->gids.resize(offset+recv_msg_size);
    for(int i = 0; i < recv_msg_size; i++) {
      memcpy( &(inProcData->X[(offset+i)*dim]), recv_data+i*span, sizeof(double)*dim);
      inProcData->charges[offset+i] = recv_data[i*span+dim];
      inProcData->gids[offset+i] = (long)recv_data[i*span+dim+1];
      inProcData->mortons[offset+i] = (long)recv_data[i*span+dim+2];
      inProcMap.insert(make_pair<long, int>(inProcData->gids[offset+i], offset+i));
    }
  }


  if(_DIST_DEBUG_) {
    askit::print_data(inProcData, comm);

    for(int r = 0; r < size; r++) {
      if(r == rank) {
        for(map<long, int>::iterator it2 = inProcMap.begin(); it2 != inProcMap.end(); it2++) {
          cout<<"\t "<<it2->first<<" - "<<it2->second<<endl;
        }
      }
      cout.flush();
      MPI_Barrier(comm);
    }
  }

  delete [] send_data;
  delete [] recv_data;

  MPI_Barrier(comm);
}


bool fksTree::getDistNodeExclNN(fks_mpiNode *inNode, fksData *exclNN)
{
  int n = inNode->excl_knn_of_this_node.size();
  int dim = inProcData->dim;

  exclNN->dim = dim;
  exclNN->numof_points = n;
  exclNN->X.resize(n*dim);
  exclNN->charges.resize(n);
  exclNN->mortons.resize(n);
  exclNN->gids.resize(n);

  for(int i = 0; i < n; i++) {
    int idx = pos(inNode->excl_knn_of_this_node[i].first);
    if(idx >= 0) {
      exclNN->gids[i] = inProcData->gids[idx];
      exclNN->mortons[i] = inProcData->mortons[idx];
      exclNN->charges[i] = inProcData->charges[idx];
      memcpy( &(exclNN->X[i*dim]), &(inProcData->X[idx*dim]), sizeof(double)*dim );
    }
    else {
      return false;
    }
  }

  return true;
}


// currently, I only try to sample points from its sibling node
// sample all other outside nodes is difficult to implement
// and it only works for p = power of 2
void fksTree::uniformSampleSibling(fks_mpiNode *inNode, int Ns, fksData *sampleData)
{
  int glb_rank, glb_size;
  MPI_Comm_rank(comm, &glb_rank);
  MPI_Comm_size(comm, &glb_size);

  if(inNode == NULL || inNode->fks_parent == NULL)    // if root, no sibling, then return
    return;

  int loc_par_rank, loc_par_size, loc_rank, loc_size;
  MPI_Comm_rank(inNode->fks_parent->comm, &loc_par_rank);
  MPI_Comm_size(inNode->fks_parent->comm, &loc_par_size);
  MPI_Comm_rank(inNode->comm, &loc_rank);
  MPI_Comm_size(inNode->comm, &loc_size);

  MPI_Status stat;

  // 1.1 exchange excl_knn_of_this_node
  int send_excl_knn = inNode->excl_knn_of_this_node.size();
  int recv_excl_knn = 0;
  int left2right_tag = 11111, right2left_tag = 22222;

  /*
  if(loc_par_rank == 0) {
    int partner_rank = loc_par_size / 2;
    MPI_Send(&send_excl_knn, 1, MPI_INT, partner_rank, left2right_tag, inNode->fks_parent->comm);
    MPI_Recv(&recv_excl_knn, 1, MPI_INT, partner_rank, right2left_tag, inNode->fks_parent->comm, &stat);
  }
  if(loc_par_rank == loc_par_size/2) {
    int partner_rank = 0;
    MPI_Send(&send_excl_knn, 1, MPI_INT, partner_rank, right2left_tag, inNode->fks_parent->comm);
    MPI_Recv(&recv_excl_knn, 1, MPI_INT, partner_rank, left2right_tag, inNode->fks_parent->comm, &stat);
  }
  */


  if(loc_par_rank == 0) {
    int partner_rank = loc_par_size / 2;
    MPI_Send(&send_excl_knn, 1, MPI_INT, partner_rank, left2right_tag, inNode->fks_parent->comm);
  }
  if(loc_par_rank == loc_par_size/2) {
    int partner_rank = 0;
    MPI_Recv(&recv_excl_knn, 1, MPI_INT, partner_rank, left2right_tag, inNode->fks_parent->comm, &stat);
  }

  if(loc_par_rank == 0) {
    int partner_rank = loc_par_size / 2;
    MPI_Recv(&recv_excl_knn, 1, MPI_INT, partner_rank, right2left_tag, inNode->fks_parent->comm, &stat);
  }
  if(loc_par_rank == loc_par_size/2) {
    int partner_rank = 0;
    MPI_Send(&send_excl_knn, 1, MPI_INT, partner_rank, right2left_tag, inNode->fks_parent->comm);
  }



  // 1.2 exchange Ns
  int glb_nsamples;

  /*
  if(loc_par_rank == 0) {
    int partner_rank = loc_par_size / 2;
    MPI_Send(&Ns, 1, MPI_INT, partner_rank, left2right_tag, inNode->fks_parent->comm);
    MPI_Recv(&glb_nsamples, 1, MPI_INT, partner_rank, right2left_tag, inNode->fks_parent->comm, &stat);
  }
  if(loc_par_rank == loc_par_size/2) {
    int partner_rank = 0;
    MPI_Send(&Ns, 1, MPI_INT, partner_rank, right2left_tag, inNode->fks_parent->comm);
    MPI_Recv(&glb_nsamples, 1, MPI_INT, partner_rank, left2right_tag, inNode->fks_parent->comm, &stat);
  }*/


  if(loc_par_rank == 0) {
    int partner_rank = loc_par_size / 2;
    MPI_Send(&Ns, 1, MPI_INT, partner_rank, left2right_tag, inNode->fks_parent->comm);
  }
  if(loc_par_rank == loc_par_size/2) {
    int partner_rank = 0;
    MPI_Recv(&glb_nsamples, 1, MPI_INT, partner_rank, left2right_tag, inNode->fks_parent->comm, &stat);
  }

  if(loc_par_rank == 0) {
    int partner_rank = loc_par_size / 2;
    MPI_Recv(&glb_nsamples, 1, MPI_INT, partner_rank, right2left_tag, inNode->fks_parent->comm, &stat);
  }
  if(loc_par_rank == loc_par_size/2) {
    int partner_rank = 0;
    MPI_Send(&Ns, 1, MPI_INT, partner_rank, right2left_tag, inNode->fks_parent->comm);
  }

  MPI_Bcast(&glb_nsamples, 1, MPI_INT, 0, inNode->comm);
  int divd = glb_nsamples / loc_size;
  int rem = glb_nsamples % loc_size;
  int loc_nsamples = loc_rank < rem ? (divd+1) : divd;

  // cout << "Rank " << glb_rank << " Receiving global num samples: " << glb_nsamples << "\n";

  if(_DIST_DEBUG_) {
    for(int r = 0; r < glb_size; r++) {
      if(r == glb_rank) {
        cout<<"glb_rank "<<glb_rank<<", loc_rank "<<loc_rank
            <<", loc_par_rank "<<loc_par_rank
            <<" send_excl_knn = "<<send_excl_knn
            <<" recv_excl_knn = "<<recv_excl_knn
            <<" glb_nsamples = "<<glb_nsamples
            <<" loc_nsamples = "<<loc_nsamples
            <<endl;
      }
      cout.flush();
      MPI_Barrier(comm);
    }
  }


  // 2. exchange exclIDs
  MPI_Bcast(&recv_excl_knn, 1, MPI_INT, 0, inNode->comm);
  vector<long> forbiddenIDs;
  forbiddenIDs.resize(recv_excl_knn);

  // cout << "Rank " << glb_rank << " forbidden size: " << recv_excl_knn << "\n";

  vector<long> tosend_fid(inNode->excl_knn_of_this_node.size());
  for(int i = 0; i < inNode->excl_knn_of_this_node.size(); i++)
    tosend_fid[i] = inNode->excl_knn_of_this_node[i].first;


  if(_DIST_DEBUG_) {
    for(int r = 0; r < glb_size; r++) {
      if(r == glb_rank) {
        cout<<"glb_rank "<<glb_rank<<", loc_rank "<<loc_rank<<", loc_par_rank "<<loc_par_rank
          <<": tosend_forbiddenIDs: ";
        for(int i = 0; i < tosend_fid.size(); i++)
          cout<<tosend_fid[i]<<" ";
        cout<<endl;
      }
      cout.flush();
      MPI_Barrier(comm);
    }
  }


  /*
  if(loc_par_rank == 0) {
    int partner_rank = loc_par_size / 2;
    MPI_Send(&(tosend_fid[0]), send_excl_knn, MPI_LONG, partner_rank, left2right_tag, inNode->fks_parent->comm);
    MPI_Recv(&(forbiddenIDs[0]), recv_excl_knn, MPI_LONG, partner_rank, right2left_tag, inNode->fks_parent->comm, &stat);
  }
  if(loc_par_rank == loc_par_size/2) {
    int partner_rank = 0;
    MPI_Send(&(tosend_fid[0]), send_excl_knn, MPI_LONG, partner_rank, right2left_tag, inNode->fks_parent->comm);
    MPI_Recv(&(forbiddenIDs[0]), recv_excl_knn, MPI_LONG, partner_rank, left2right_tag, inNode->fks_parent->comm, &stat);
  }
  */

  if(loc_par_rank == 0) {
    int partner_rank = loc_par_size / 2;
    // cout << "Rank " << glb_rank << " sending first value " << tosend_fid[0] << "\n";
    MPI_Send(tosend_fid.data(), send_excl_knn, MPI_LONG, partner_rank, left2right_tag, inNode->fks_parent->comm);
  }
  if(loc_par_rank == loc_par_size/2) {
    int partner_rank = 0;
    MPI_Recv(forbiddenIDs.data(), recv_excl_knn, MPI_LONG, partner_rank, left2right_tag, inNode->fks_parent->comm, &stat);
  }

 if(loc_par_rank == loc_par_size/2) {
    int partner_rank = 0;
    MPI_Send(tosend_fid.data(), send_excl_knn, MPI_LONG, partner_rank, right2left_tag, inNode->fks_parent->comm);
  }
  if(loc_par_rank == 0) {
    int partner_rank = loc_par_size / 2;
    MPI_Recv(forbiddenIDs.data(), recv_excl_knn, MPI_LONG, partner_rank, right2left_tag, inNode->fks_parent->comm, &stat);
  }

  MPI_Bcast(forbiddenIDs.data(), recv_excl_knn, MPI_LONG, 0, inNode->comm);

  if(_DIST_DEBUG_) {
    for(int r = 0; r < glb_size; r++) {
      if(r == glb_rank) {
        cout<<"glb_rank "<<glb_rank<<", loc_rank "<<loc_rank<<", loc_par_rank "<<loc_par_rank
          <<": forbiddenIDs: ";
        for(int i = 0; i < forbiddenIDs.size(); i++)
          cout<<forbiddenIDs[i]<<" ";
        cout<<endl;
      }
      cout.flush();
      MPI_Barrier(comm);
    }
  }


  // 3. sample points
  int dim = inProcData->dim;
  vector<double> samples_arr;     // coords, charge, gid, morton, ...
  int span = dim + 3;
  samples_arr.resize(loc_nsamples*span);
  int curr_ns = 0;
  int N = numof_points_of_dist_leaf;
  while(curr_ns < loc_nsamples) {
    int idx = floor( (double)N*( (double)rand()/(double)RAND_MAX) );
    long sgid = inProcData->gids[idx];
    if( !binary_search(forbiddenIDs.begin(), forbiddenIDs.end(), sgid) ) {
      samples_arr[curr_ns*span+dim] = inProcData->charges[idx];
      samples_arr[curr_ns*span+dim+1] = (double)inProcData->gids[idx];
      samples_arr[curr_ns*span+dim+2] = (double)inProcData->mortons[idx];
      memcpy( &(samples_arr[curr_ns*span]), &(inProcData->X[idx*dim]), sizeof(double)*dim );
      curr_ns++;
    }
  }


  if(_DIST_DEBUG_) {
    for(int r = 0; r < glb_size; r++) {
      if(r == glb_rank) {
        cout<<"glb_rank "<<glb_rank<<", loc_rank "<<loc_rank<<", loc_par_rank "<<loc_par_rank
          <<": samples: ";
        for(int i = 0; i < loc_nsamples; i++) {
          cout<<"\t"<<samples_arr[i*span+dim+1]<<" ("<<samples_arr[i*span+dim]<<"): ";
          for(int j = 0; j < dim; j++)
            cout<<samples_arr[i*span+j]<<" ";
          cout<<endl;
        }
        cout<<endl;
      }
      cout.flush();
      MPI_Barrier(comm);
    }
  }


  // 4. gather sample points into local rank 0 from inNode->fks_parent->comm
  // mpi_alltoallv
  vector<int> send_count(loc_par_size);
  vector<int> send_disp(loc_par_size);
  vector<int> recv_count(loc_par_size);
  vector<int> recv_disp(loc_par_size);

  memset(&(send_count[0]), 0, sizeof(int)*loc_par_size);
  if(loc_par_rank < loc_par_size/2)
    send_count[loc_par_size/2] = loc_nsamples;
  if(loc_par_rank >= loc_par_size/2)
    send_count[0] = loc_nsamples;

  divd = Ns / loc_size;
  rem = Ns % loc_size;
  memset(&(recv_count[0]), 0, sizeof(int)*loc_par_size);
  if(loc_par_rank == 0) {
    for(int i = 0; i < loc_size; i++)
      recv_count[loc_par_size/2+i] = i < rem ? (divd+1) : divd;
  }
  if(loc_par_rank == loc_par_size/2) {
    for(int i = 0; i < loc_size; i++)
      recv_count[i] = i < rem ? (divd+1) : divd;
  }

  for(int i = 0; i < loc_par_size; i++) {
    send_count[i] = span * send_count[i];
    recv_count[i] = span * recv_count[i];
  }
  send_disp[0] = 0;
  recv_disp[0] = 0;
  for(int i = 1; i < loc_par_size; i++) {
    send_disp[i] = send_disp[i-1] + send_count[i-1];
    recv_disp[i] = recv_disp[i-1] + recv_count[i-1];
  }

  vector<double> recv_arr;
  recv_arr.resize(Ns*span);
  MPI_Alltoallv( samples_arr.data(), &(send_count[0]), &(send_disp[0]), MPI_DOUBLE,
    recv_arr.data(), &(recv_count[0]), &(recv_disp[0]), MPI_DOUBLE, inNode->fks_parent->comm );


  if(_DIST_DEBUG_) {
    for(int r = 0; r < glb_size; r++) {
      if(r == glb_rank) {
        cout<<"glb_rank "<<glb_rank<<", loc_rank "<<loc_rank<<", loc_par_rank "<<loc_par_rank
          <<": recv_arr: ";
        for(int i = 0; i < Ns; i++) {
          cout<<"\t"<<recv_arr[i*span+dim+1]<<" ("<<recv_arr[i*span+dim]<<"): ";
          for(int j = 0; j < dim; j++)
            cout<<recv_arr[i*span+j]<<" ";
          cout<<endl;
        }
        cout<<endl;
      }
      cout.flush();
      MPI_Barrier(comm);
    }
  }

  // 5. make sampleData
  if(loc_rank == 0) {
    sampleData->numof_points = Ns;
    sampleData->dim = dim;
    sampleData->X.resize( Ns * dim );
    sampleData->charges.resize( Ns );
    sampleData->mortons.resize( Ns );
    sampleData->gids.resize( Ns );
    for(int i = 0; i < Ns; i++) {
      sampleData->charges[i] = recv_arr[i*span+dim];
      sampleData->gids[i] = (long)recv_arr[i*span+dim+1];
      sampleData->mortons[i] = (long)recv_arr[i*span+dim+2];
      memcpy( &(sampleData->X[i*dim]), &(recv_arr[i*span]), sizeof(double)*dim );
    }
  }

  //cout<<"rank "<<glb_rank<<" done make sampleData \n";
  MPI_Barrier(comm);

}


void fksTree::mergeSkeletonsOfKids(fks_mpiNode *inNode, fksData *ske)
{

  // Note from Bill:
  // Had to rewrite this. Now, we do a couple of exchanges. We have a left 
  // sibling (L), which receives a skeleton from the right sibling (R).
  // The new algorithm is: 
  // - R sends gids and effective charges to L
  // - L checks which gids it needs coordinates for and sends these gids to R
  // - R gathers coordinates, (original) charges, and mortons and sends these 
  // to L

  int glb_rank, glb_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &glb_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &glb_size);


  // for (int r = 0; r < glb_size; r++)
  // {
  //
  //   MPI_Barrier(MPI_COMM_WORLD);
  //   if (glb_rank == r)
  //   {
  //     cout << "Rank " << glb_rank << " entering mergeSkeletonsOfKids at level " << inNode->level << "\n";
  //     cout << "node child: " << inNode->fks_kid << ", parent: " << inNode->fks_parent << "\n\n";
  //   }
  //
  // }
    

  // if (inNode == NULL)
  // {
  //   cout << "Rank " << glb_rank << " NODE NULL IN MERGE SKELETONS OF KIDS.\n";
  // }

  // if(inNode == NULL || inNode->fks_kid == NULL) {
  //   // if leaf, I would assume the skeletonization was done
  //   // at the shared memory tree, so here the ->skeleton of
  //   // leaf node in the distributed tree remains NULL
  //   // i.e. the skeleton is just copied from the root_omp elsewhere
  //   // cout << "Rank " << glb_rank << " exiting mergeSkeletonsofKids early\n";
  //     cout << "Rank " << glb_rank << " exiting mergeSkeletonsofKids early\n";
  //     MPI_Barrier(MPI_COMM_WORLD);
  //     return;
  // }

  int loc_rank, loc_size;
  MPI_Comm_rank(inNode->comm, &loc_rank);
  // cout << "Rank " << glb_rank << " setting loc_rank " << loc_rank << "\n\n";
  MPI_Comm_size(inNode->comm, &loc_size);
  // cout << "Rank " << glb_rank << " setting loc_size " << loc_size << "\n\n";
  int dim = inProcData->dim;
  MPI_Status stat;

  // for (int r = 0; r < glb_size; r++)
//   {
//     MPI_Barrier(MPI_COMM_WORLD);
//     if (r == glb_rank)
//       cout << "Rank " << glb_rank << " computing flag\n";
//   }
//
  
  // cout << "Rank " << glb_rank << " setting flag with dim " << dim << "\n\n";
  
  
  // First, we need to find out if either me or my sibling has the cant prune 
  // flag set for the child's skeleton. If so, set this skeleton to cant prune
  // and return
  int my_flag = 0;
  if (loc_rank == loc_size / 2 || loc_rank == 0)
  {
    // need to check that the skeleton exists at all
    if (!inNode->fks_kid->skeletons || inNode->fks_kid->skeletons->cant_prune)
      my_flag = 1;
  }
  int res_flag;

  // do the communicators one at a time, see if that helps
  // for (int i = 0; i < glb_size / loc_size; i++)
  // {
  //   if (glb_rank / loc_size == i)
  //   {
  
  MPI_Allreduce(&my_flag, &res_flag, 1, MPI_INT, MPI_LOR, inNode->comm);
  //   }
  //   MPI_Barrier(MPI_COMM_WORLD);
  // }
  
  
  // MPI_Reduce(&my_flag, &res_flag, 1, MPI_INT, MPI_LOR, 0, inNode->comm);
  // if (loc_rank == 0)
  //   cout << "Rank " << glb_rank << " finished reduce.\n";
  //
  // MPI_Bcast(&res_flag, 1, MPI_INT, 0, inNode->comm);
  //
  // cout << "Rank " << glb_rank << " finished all reduce.\n";


  // for (int r = 0; r < glb_size; r++)
//   {
//     MPI_Barrier(MPI_COMM_WORLD);
//     if (r == glb_rank)
//       cout << "Rank " << glb_rank << " cant prune flag: " << res_flag << "\n";
//   }

  // Make sure everyone knows whether to stop or not
  if (res_flag)
  {
    if (inNode->skeletons)
      inNode->skeletons->cant_prune = true;
    return;
  }
  
  vector<double> recv_buff;   // coord, charge
  // for each point, we receive gid and charge
  int span = 2;

  int send_num_skeleton = 0, recv_num_skeleton = 0;
  int left2right_tag = 11111, right2left_tag = 22222;

  // 1.1 exchange skeleton size
  send_num_skeleton = (inNode->fks_kid->skeletons == NULL) ? 0 : inNode->fks_kid->skeletons->numof_points;
  if(loc_rank == 0) {
    int partner_rank = loc_size/2;
    MPI_Recv(&recv_num_skeleton, 1, MPI_INT, partner_rank, right2left_tag, inNode->comm, &stat);
  }
  if(loc_rank == loc_size/2) {
    int partner_rank = 0;
    MPI_Send(&send_num_skeleton, 1, MPI_INT, partner_rank, right2left_tag, inNode->comm);
  }

  // 1.2 collect skeletons
  recv_buff.resize(recv_num_skeleton * span);
  if(loc_rank == 0) {
    int partner_rank = loc_size/2;
    MPI_Recv(&(recv_buff[0]), recv_num_skeleton*span, MPI_DOUBLE, partner_rank, right2left_tag, inNode->comm, &stat);
  }
  if(loc_rank == loc_size/2) {
    int partner_rank = 0;
    vector<double> send_buff;
    send_buff.resize(send_num_skeleton*span);
    for(int i = 0; i < send_num_skeleton; i++) {
      send_buff[i*span] = (double)inNode->fks_kid->skeletons->gids[i];
      send_buff[i*span+1] = inNode->fks_kid->skeletons->charges[i];
    }
    MPI_Send(&(send_buff[0]), send_num_skeleton*span, MPI_DOUBLE, partner_rank, right2left_tag, inNode->comm);
  }
  
  // fill in the skeleton of this node
  int ns = 0;
  
  vector<double> gids_needed;
    
  if(loc_rank == 0) {
    ns = inNode->fks_kid->skeletons->numof_points + recv_num_skeleton;
    ske->numof_points = ns;
    ske->dim = dim;
    //ske->X.resize(ns*dim);
    ske->charges.resize(ns);
    ske->gids.resize(ns);
    
    // the points I already had
    for(int i = 0; i < inNode->fks_kid->skeletons->numof_points; i++) {
      //cout << "loop charges\n";
      ske->charges[i] = inNode->fks_kid->skeletons->charges[i];
      //cout << "loop gids\n";
      ske->gids[i] = inNode->fks_kid->skeletons->gids[i];
    }
    int offset = inNode->fks_kid->skeletons->numof_points;

    // now, we add the points we received to the skeleton
    // we also collect the gids of points for which we don't have coordinates
    
    for (int i = 0; i < recv_num_skeleton; i++)
    {
      
      long gid = (long)recv_buff[i*span];
      ske->gids[offset + i] = gid;
      ske->charges[offset + i] = recv_buff[i*span + 1];
      
      // check if we already have this gid
      if (pos(gid) < 0)
      {
        gids_needed.push_back((double)gid);
      }
      
    } // loop over received skeleton gids
    
  } // local rank 0 (i.e. left sibling or L)

  // Exchange the number of gids needed to send
  int send_num_gids_needed = gids_needed.size();
  int recv_num_gids_needed = 0;
  if(loc_rank == 0) {
    int partner_rank = loc_size/2;
    MPI_Send(&send_num_gids_needed, 1, MPI_INT, partner_rank, left2right_tag, inNode->comm);
  }
  if(loc_rank == loc_size/2) {
    int partner_rank = 0;
    MPI_Recv(&recv_num_gids_needed, 1, MPI_INT, partner_rank, left2right_tag, inNode->comm, &stat);
  }
  
  // now exchange the gids themselves
  vector<double> recv_gids(recv_num_gids_needed);
  if(loc_rank == 0) {
    int partner_rank = loc_size/2;
    MPI_Send(&(gids_needed[0]), send_num_gids_needed, MPI_DOUBLE, partner_rank, left2right_tag, inNode->comm);
  }
  if(loc_rank == loc_size/2) {
    int partner_rank = 0;
    MPI_Recv(&(recv_gids[0]), recv_num_gids_needed, MPI_DOUBLE, partner_rank, left2right_tag, inNode->comm, &stat);
  }
  
  // now we send the coorindates, original charge, and morton ID
  span = dim + 2;
  
  // rank 0 receives coodinates
  if (loc_rank == 0) {

    // we're the left sibling
    int partner_rank = loc_size / 2;

    vector<double> recv_coordinates(span * send_num_gids_needed);
    
    MPI_Recv(&(recv_coordinates[0]), send_num_gids_needed*span, MPI_DOUBLE, partner_rank, right2left_tag, inNode->comm, &stat);

    // now, put the coordinates in inProcData
    int nexist = inProcData->numof_points;
    inProcData->numof_points = nexist + send_num_gids_needed;
    inProcData->X.resize( (nexist + send_num_gids_needed ) * dim )  ;
    inProcData->mortons.resize( nexist + send_num_gids_needed )  ;
    inProcData->charges.resize( nexist + send_num_gids_needed)  ;
    inProcData->gids.resize( nexist + send_num_gids_needed )  ;

    for(int i = 0; i < send_num_gids_needed; i++) {
      
      long this_gid = (long)gids_needed[i];
      
      inProcData->gids[nexist + i] = this_gid;
      memcpy( &(inProcData->X[(nexist+i)*dim]), &(recv_coordinates[i*span]), sizeof(double)*dim);
      inProcData->charges[nexist+i] = recv_coordinates[i*span+dim];
      inProcData->mortons[nexist+i] = (long)recv_coordinates[i*span + dim+1];

      inProcMap.insert(make_pair(this_gid, nexist+i));

    }

  } // local rank 0 (receiving coorindates)
  if (loc_rank == loc_size / 2)
  {
    
    // we're the right sibling
    int partner_rank = 0;
    
    vector<double> send_coordinates(recv_num_gids_needed * span);
    
    for (int i = 0; i < recv_num_gids_needed; i++)
    {
      
      long gid = (long)recv_gids[i];
      int lid = pos(gid);
      assert(lid >= 0);
      
      // coorindates, charges, morton
      memcpy(&(send_coordinates[i*span]), &(inProcData->X[lid*dim]), sizeof(double)*dim);
      send_coordinates[i*span + dim] = inProcData->charges[lid];
      send_coordinates[i*span + dim + 1] = (double)inProcData->mortons[lid];
      
    }
    
    // now, send the info
    MPI_Send(&(send_coordinates[0]), recv_num_gids_needed*span, MPI_DOUBLE, partner_rank, right2left_tag, inNode->comm);
    
  } // local rank 1 (sending coordinates)

  // for (int r = 0; r < glb_size; r++)
  // {
  //   if (glb_rank == r)
  //     cout << "Rank " << glb_rank << " finished mergeSkeletonsOfKids\n";
  //   MPI_Barrier(comm);
  // }

} // merge skeletons





// ------------------- let construction ---------------------------
//
// ----------------------------------------------------------------

// direct interactions: global ids of source nodes which interact directly 
// with each target point
void fksTree::LET(int min_skeleton_global_level,
set< triple<long, long, int> > &set_leaves,
set< triple<long, long, int> > &set_skeletons, 
vector<pair<long, int> >& my_skeleton_frontier, int k,
vector<vector<long> >& direct_interactions,
vector<vector<long> >& approx_interactions)
{
  
  int worldsize, worldrank;
  MPI_Comm_rank(comm, &worldrank);
  MPI_Comm_size(comm, &worldsize);

  min_training_skeleton_level = min_skeleton_global_level;

  double start_t = omp_get_wtime();
  //int k = inProcKNN->size() / numof_points_of_dist_leaf;

  // Doing this because this gets called multiple times for update charges.
  // my_set_leaves.clear();
  // my_set_skeletons.clear();

  // adding k as a parameter to handle the fewer k in recursion case
  if (k == 0)
    k = inProcKNN->size() / numof_points_of_dist_leaf;

  long k_in_tree = inProcKNN->size() / numof_points_of_dist_leaf;

  if(_OUTPUT_) {
    MPI_Barrier(comm);
    if(worldrank == 0) cout<<"fksTree::let(): enter "<<omp_get_wtime()-start_t<<endl;
    start_t = omp_get_wtime();
  }
  
  double frontier_start = omp_get_wtime();
  
  // Communicate the skeletonization frontier
  MPI_Datatype frontier_type;
  MPI_Type_contiguous(sizeof(pair<long, int>), MPI_BYTE, &frontier_type);
  MPI_Type_commit(&frontier_type);

  // Collect the sizes of the frontiers
  int send_count = my_skeleton_frontier.size();
  vector<int> recv_count(worldsize);
  MPI_Allgather(&send_count, 1, MPI_INT, recv_count.data(), 1, MPI_INT, MPI_COMM_WORLD);
  
  vector<int> recv_disp(worldsize);
  recv_disp[0] = 0;
  int ntotal_recv = recv_count[0];
  for(int l = 1; l < worldsize; l++) {
    ntotal_recv += recv_count[l];
    recv_disp[l] = recv_disp[l-1] + recv_count[l-1];
  }
  
  // int MPI_Allgatherv  (void* sbuf, int scount, MPI_Datatype stype, void* rbuf, int* rcount, int* displs, MPI_Datatype rtype, MPI_Comm comm)
  
  skeleton_frontier.resize(ntotal_recv);
  MPI_Allgatherv(my_skeleton_frontier.data(), send_count, frontier_type, 
    skeleton_frontier.data(), recv_count.data(), recv_disp.data(), 
    frontier_type, MPI_COMM_WORLD);
  
  // Do we want to store the frontier outside for later? Might need it for test 
  // points?
    
  frontier_exchange_time = omp_get_wtime() - frontier_start;


  if (worldrank == 0)
  {
    cout << "\n\nSkeleton frontier: \n";
    for (int i = 0; i < skeleton_frontier.size(); i++)
    {
      cout << "( " << skeleton_frontier[i].first << ", " << skeleton_frontier[i].second << "), ";
    }
    cout << "\n\n";
  }
  



  double prune_by_knn_start = omp_get_wtime();

  //cout << "Pruning by knn\n";
  int nthreads = 1;
  #pragma omp parallel
  {
      nthreads = omp_get_num_threads();
  }
  vector< set< triple<long, long, int> > > set_leaves_multithreads;
  vector< set< triple<long, long, int> > > set_skeletons_multithreads;
  set_leaves_multithreads.resize(nthreads);
  set_skeletons_multithreads.resize(nthreads);
  #pragma omp parallel
  {
      int tid = omp_get_thread_num();
      long *mortons_knn = new long [k];
      #pragma omp for
      for(long i = 0; i < numof_points_of_dist_leaf; i++) {
          for(long j = 0; j < k; j++) {
              int lid = pos( (*inProcKNN)[i*k_in_tree+j].second );
              mortons_knn[j] = inProcData->mortons[lid];
          }
          pruneByKNN(0, 0, mortons_knn, k, min_skeleton_global_level,
                     set_leaves_multithreads[tid],
                     set_skeletons_multithreads[tid],
                     direct_interactions[i],
                     approx_interactions[i], skeleton_frontier);
      }
      delete [] mortons_knn;
  }
  // merge the copies for each thread
  for(int i = 0; i < nthreads; i++) {
    set_leaves.insert(set_leaves_multithreads[i].begin(), set_leaves_multithreads[i].end());
  }
  for(int i = 0; i < nthreads; i++) {
    set_skeletons.insert(set_skeletons_multithreads[i].begin(), set_skeletons_multithreads[i].end());
  }


  prune_by_knn_time = omp_get_wtime() - prune_by_knn_start;

  // Now, we need to exchange and update with any missing nodes in the 
  // adaptive rank algorithm
  
  // TODO: add a check if we're doing adaptive rank and use it here
  // update_adaptive_LET(set_skeletons, set_leaves);

  // BXC
  //cout<<"rank "<<worldrank
  //  <<" let prune by knn done "<<omp_get_wtime()-start_t
  //  <<" set_leaves = "<<set_leaves.size()
  //  <<" set_skeletons = "<<set_skeletons.size()
  //  <<endl;

  if(_DEBUG_LET_) {
      MPI_Barrier(MPI_COMM_WORLD);
      if(worldrank == 0) cout<<"let() set_leaves: "<<endl;
      print_set(set_leaves, MPI_COMM_WORLD);

      MPI_Barrier(MPI_COMM_WORLD);
      if(worldrank == 0) cout<<"let() set_skeletons: "<<endl;
      print_set(set_skeletons, MPI_COMM_WORLD);
  }

  //cout << "inserting root_omp\n";
  // .2 insert root_omp into root_let

  if (worldsize > 1)
  {
  
    root_let = new fks_ompNode();
    graft_omp_to_let();

    if(_DEBUG_TEST_POINT_) {
        if(worldrank == 0) cout<<"graft omp to let: "<<endl;
        MPI_Barrier(comm);
        print_tree(root_let, comm);
        if(worldrank == 0) cout<<endl<<endl;
        MPI_Barrier(comm);
    }


    if(_DEBUG_TEST_POINT_) {
        if(worldrank == 0) cout<<endl;
        for(int r = 0; r < worldsize; r++) {
            if(r == worldrank) {
                cout<<"set_leaves (node_gid, node_morton, node_level) on "<<worldrank<<": "<<endl;
                for(set< triple<long, long, int> >::iterator it = set_leaves.begin();
                        it != set_leaves.end(); it++) {
                    cout<<"("<<it->first<<","<<it->second<<","<<it->third<<")  ";
                }
                cout<<endl;

                cout<<"set_skeletons (node_gid, node_morton, node_level) on "<<worldrank<<": "<<endl;
                for(set< triple<long, long, int> >::iterator it = set_skeletons.begin();
                        it != set_skeletons.end(); it++) {
                    cout<<"("<<it->first<<","<<it->second<<","<<it->third<<")  ";
                }
                cout<<endl;
            }
            cout.flush();
            MPI_Barrier(comm);
        }
    }

    //cout << "Insert nodes\n";
    // .3 insert necessary let node accd. to morton id
    for(set< triple<long, long, int> >::iterator it = set_leaves.begin();
    it != set_leaves.end(); it++) {
      insert_node_by_morton(root_let, it->second, it->third);
    }
    for(set< triple<long, long, int> >::iterator it = set_skeletons.begin();
    it != set_skeletons.end(); it++) {
      insert_node_by_morton(root_let, it->second, it->third);
    }

    if(_DEBUG_TEST_POINT_) {
        if(worldrank == 0) cout<<"insert training node: "<<endl;
        MPI_Barrier(comm);
        print_tree(root_let, comm);
        if(worldrank == 0) cout<<endl<<endl;
        MPI_Barrier(comm);
    }

    //cout << "Make list array\n";
    // .4 make list array of let node
    level_order_traversal(root_let);

    if(_DEBUG_TEST_POINT_) {
          MPI_Barrier(comm);
          if(worldrank == 0) 
            cout<<endl;
          
          for(int r = 0; r < worldsize; r++) {
              if(r == worldrank) {
                  cout<<"letNodeList after let training (gid, lid, lid by map) "<<worldrank<<": "<<endl;
                  for(int i = 0; i < letNodeList.size(); i++) {
                      cout<<letNodeList[i]->global_node_id<<", "
                          <<letNodeList[i]->lnid<<", "
                          <<letNodeMap[letNodeList[i]->global_node_id]<<endl;
                  }
                  cout<<endl;
              }
              cout.flush();
              MPI_Barrier(comm);
          }
      }

    //cout << "Copy skeletons\n";
    // .5 put local skeleton from root_mpi into root_let
    fks_mpiNode *curr_mpi = root_mpi;
    while(curr_mpi != NULL) {
      long p2 = (int)pow(2.0, curr_mpi->level);
      long idx = p2 - 1 + worldrank / (worldsize/p2);
      if(curr_mpi->skeletons != NULL) {   // copy it to let
        fks_ompNode* curr = let_node(idx);
        curr->skeletons = new fksData();
        curr->skeletons->dim = inProcData->dim;
        curr->skeletons->numof_points = curr_mpi->skeletons->numof_points;
        curr->skeletons->charges.resize( curr_mpi->skeletons->charges.size() );
        curr->skeletons->gids.resize( curr_mpi->skeletons->gids.size() );
        memcpy( &(curr->skeletons->charges[0]), &(curr_mpi->skeletons->charges[0]),
        sizeof(double)*(curr_mpi->skeletons->charges.size()) );
        memcpy( &(curr->skeletons->gids[0]), &(curr_mpi->skeletons->gids[0]),
        sizeof(long)*(curr_mpi->skeletons->gids.size()) );
        // delete curr_mpi->skeletons
        //delete curr_mpi->skeletons;
        //curr_mpi->skeletons = NULL;
      }
      curr_mpi = curr_mpi->fks_kid;
    }

  } // more than one rank
  
  
  //std::cout << "LET() finished\n";

  // .6 store sets
  for(set< triple<long, long, int> >::iterator it = set_leaves.begin();
  it != set_leaves.end(); it++) {
    my_set_leaves.insert(*it);
  }
  for(set< triple<long, long, int> >::iterator it = set_skeletons.begin();
  it != set_skeletons.end(); it++) {
    my_set_skeletons.insert(*it);
    //my_set_skeletons.insert(triple<long, long, int>(it->first, it->second, it->third));
  }

}


void fksTree::graft_omp_to_let()
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    fks_mpiNode *curr_mpi = root_mpi;
    while(curr_mpi->fks_kid != NULL)
        curr_mpi = curr_mpi->fks_kid;
    fks_ompNode *tmp = insert_node_by_morton(root_let, curr_mpi->node_morton, curr_mpi->level);

    if(_DEBUG_LET_) {
        if(rank == 0) cout<<"first attempt insert root_let"<<endl;
        MPI_Barrier(comm);
        print_tree(root_let, comm);
    }

    root_omp->parent = tmp->parent;
    if(check_bit(tmp->node_morton, tmp->level) == 0) {
        tmp->parent->leftNode = root_omp;
    } else {
        tmp->parent->rightNode = root_omp;
    }

    // change lnid, level and gnid of shared tree
    modify_root_omp_id(root_omp);

    if(_DEBUG_LET_) {
        if(rank == 0) cout<<"graft root_omp on root_let"<<endl;
        MPI_Barrier(comm);
        print_tree(root_let, comm);
    }

}


void fksTree::modify_root_omp_id(fks_ompNode *inNode)
{
    if(inNode == NULL)
        return;

    inNode->level = inNode->parent->level+1;
    if(check_bit(inNode->node_morton, inNode->level) == 1) { // right node
        inNode->lnid = 2*inNode->parent->lnid+1;
        inNode->global_node_id = 2*inNode->parent->global_node_id+2;
    }
    else {
        inNode->lnid = 2*inNode->parent->lnid+0;
        inNode->global_node_id = 2*inNode->parent->global_node_id+1;
    }

    modify_root_omp_id(inNode->leftNode);
    modify_root_omp_id(inNode->rightNode);
}



fks_ompNode* fksTree::let_node(long node_gid)
{
  fks_ompNode* ptrnode = NULL;
  map<long, int>::iterator it = letNodeMap.find( node_gid );
  if(it != letNodeMap.end() ) {
    ptrnode = letNodeList[it->second];
    return ptrnode;
  }
  return ptrnode;
}



void fksTree::exchange_let(set< triple<long, long, int> > &set_leaves,
set< triple<long, long, int> > &set_skeletons)
{
  vector<int> skel_sizes;
  bool check_sizes = false;
  exchange_let(set_leaves, set_skeletons, skel_sizes, check_sizes);
}

void fksTree::exchange_let(set< triple<long, long, int> > &set_leaves,
set< triple<long, long, int> > &set_skeletons,
                      vector<int>& skeleton_sizes, bool check_sizes)
{
  int size, rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  long dummy_np = inProcData->numof_points, max_np, min_np, avg_np;
  MPI_Allreduce(&dummy_np, &max_np, 1, MPI_LONG, MPI_MAX, comm);
  MPI_Allreduce(&dummy_np, &min_np, 1, MPI_LONG, MPI_MIN, comm);
  MPI_Allreduce(&dummy_np, &avg_np, 1, MPI_LONG, MPI_SUM, comm);
  avg_np /= size;

  int dummy_nleaf = set_leaves.size(), max_nleaf, min_nleaf, avg_nleaf;
  MPI_Allreduce(&dummy_nleaf, &max_nleaf, 1, MPI_INT, MPI_MAX, comm);
  MPI_Allreduce(&dummy_nleaf, &min_nleaf, 1, MPI_INT, MPI_MIN, comm);
  MPI_Allreduce(&dummy_nleaf, &avg_nleaf, 1, MPI_INT, MPI_SUM, comm);
  avg_nleaf /= size;

  int dummy_nskeleton = set_skeletons.size(), max_nskeleton, min_nskeleton, avg_nskeleton;
  MPI_Allreduce(&dummy_nskeleton, &max_nskeleton, 1, MPI_INT, MPI_MAX, comm);
  MPI_Allreduce(&dummy_nskeleton, &min_nskeleton, 1, MPI_INT, MPI_MIN, comm);
  MPI_Allreduce(&dummy_nskeleton, &avg_nskeleton, 1, MPI_INT, MPI_SUM, comm);
  avg_nskeleton /= size;

  if(rank == 0) {
     cout<<"after let, \tnum leaves: \tmin = "<<min_nleaf
          <<", \tmax = "<<max_nleaf<<", \tavg = "<<avg_nleaf<<endl;
     cout<<"after let, \tnum skeletons: \tmin = "<<min_nskeleton
          <<", \tmax = "<<max_nskeleton<<", \tavg = "<<avg_nskeleton<<endl;
     cout<<"before exchange let (after skeletonization), \tinProcData: \tmin = "<<min_np
          <<", \tmax = "<<max_np<<", \tavg = "<<avg_np<<endl;
  }

  double start_t = omp_get_wtime();

  int dim = inProcData->dim;

  int *send_count = new int [size];
  int *send_disp = new int [size];
  int *recv_count = new int [size];
  int *recv_disp = new int [size];

  if (check_sizes)
  {
    cout << "rank " << rank << ": Beginning of exchange_let check.\n";

    for (int i = 0; i < letNodeList.size(); i++)
    {

      fks_ompNode* node = letNodeList[i];

      if (node && node->skeletons)
      {
      
        if (skeleton_sizes[i] != node->skeletons->numof_points)
        {
          cout << "Rank " << rank << ": failing on node " << i << ". size: " << skeleton_sizes[i] << ", numof_points: " << node->skeletons->numof_points << "\n";
          assert(skeleton_sizes[i] == node->skeletons->numof_points);
        }
      } // does it have a skeleton?

    } // loop over nodes

  } // if debug skeleton scan

  //cout << "requesting node gids\n";
  // .1 request_node_gids: triple<leaf_or_not, rank, node_gid>
  // set_leaves/skels <gid, morton, level>
  int p = 0;
  vector< triple<bool, int, long> > request_node_gids;
  request_node_gids.resize(set_leaves.size()+set_skeletons.size());
  // Set up request for each LET leaf
  for(set< triple<long, long, int> >::iterator it = set_leaves.begin(); it != set_leaves.end(); it++) {
    int myrank = morton_to_rank(it->second, it->third);
    if(myrank != rank) {
      request_node_gids[p].first = true;
      request_node_gids[p].second = myrank;
      request_node_gids[p].third = it->first;
      p++;
    }
  }
  // Set up request for each LET skeleton
  for(set< triple<long, long, int> >::iterator it = set_skeletons.begin(); it != set_skeletons.end(); it++) {
    int myrank = morton_to_rank(it->second, it->third);
    if(myrank != rank) {
      request_node_gids[p].first = false;
      request_node_gids[p].second = myrank;
      request_node_gids[p].third = it->first;
      p++;
    }
  }
  request_node_gids.resize(p);    // because delete those in proc
  // - sort by rank

  // cout << "Rank " << rank << " sorting " << p << " request node gids\n";

  // This doesn't seem to work if the list is empty
  if (p > 0)
    omp_par::merge_sort(request_node_gids.begin(), request_node_gids.end(), triple<bool, int, long>::secondLess);

  // - exchange node id

  // fill out send count for each rank
  memset(send_count, 0, sizeof(int)*size);
  for(int i = 0; i < request_node_gids.size(); i++)
    send_count[ request_node_gids[i].second ]++;

  // exchange send counts
  MPI_Alltoall( send_count, 1, MPI_INT, recv_count, 1, MPI_INT, comm );

  // Scan recv count
  int ntotal_recv = recv_count[0];
  send_disp[0] = 0;
  recv_disp[0] = 0;
  for(int l = 1; l < size; l++) {
    ntotal_recv += recv_count[l];
    send_disp[l] = send_disp[l-1] + send_count[l-1];
    recv_disp[l] = recv_disp[l-1] + recv_count[l-1];
  }

  // vector of nodes we'll receive
  vector< triple<bool, int, long> > recv_node_gids(ntotal_recv);

  // create a type for node triple
  MPI_Datatype let_node_type;
  MPI_Type_contiguous(sizeof(triple<bool, int, long>), MPI_BYTE, &let_node_type);
  MPI_Type_commit(&let_node_type);

  // exhange node info
  MPI_Alltoallv(&(request_node_gids[0]), send_count, send_disp, let_node_type,
                &(recv_node_gids[0]), recv_count, recv_disp, let_node_type, comm);

    if(_DEBUG_LET_) {
        if(rank == 0) cout<<endl;
        MPI_Barrier(comm);

        for(int r  = 0; r < size; r++) {
            if(r == rank) {
                cout<<"(rank "<<rank<<") : send_count: ";
                for(int i = 0; i < size; i++) {
                    cout<<send_count[i]<<" ";
                }
                cout<<endl;
            }
            cout.flush();
            MPI_Barrier(comm);
        }

        for(int r  = 0; r < size; r++) {
            if(r == rank) {
                for(int i = 0; i < request_node_gids.size(); i++) {
                    cout<<"(rank "<<rank<<") request_nodes: leaf ? "
                        <<request_node_gids[i].first
                        <<", home_rank "<<request_node_gids[i].second
                        <<", node_gid "<<request_node_gids[i].third
                        <<endl;
                }
            }
            cout.flush();
            MPI_Barrier(comm);
        }

        for(int r  = 0; r < size; r++) {
            if(r == rank) {
                cout<<"(rank "<<rank<<") : recv_count: ";
                for(int i = 0; i < size; i++) {
                    cout<<recv_count[i]<<" ";
                }
                cout<<"(rank "<<rank<<") : ntotal_recv = "<<ntotal_recv<<endl;
            }
            cout.flush();
            MPI_Barrier(comm);
        }

        for(int r  = 0; r < size; r++) {
            if(r == rank) {
                for(int i = 0; i < recv_node_gids.size(); i++) {
                    cout<<"(rank "<<rank<<") recv_request_nodes: leaf ? "
                        <<recv_node_gids[i].first
                        <<", home_rank "<<recv_node_gids[i].second
                        <<", node_gid "<<recv_node_gids[i].third
                        <<endl;
                }
            }
            cout.flush();
            MPI_Barrier(comm);
        }

    } // DEBUG_LET

    if (check_sizes)
    {
      cout << "rank " << rank << ": after step 1 of exchange_let check.\n";

      for (int i = 0; i < letNodeList.size(); i++)
      {
    
        fks_ompNode* node = letNodeList[i];
    
        if (node && node->skeletons)
        {
      
          if (skeleton_sizes[i] != node->skeletons->numof_points)
          {
            cout << "Rank " << rank << ": failing on node " << i << ". size: " << skeleton_sizes[i] << ", numof_points: " << node->skeletons->numof_points << "\n";
            assert(skeleton_sizes[i] == node->skeletons->numof_points);
          }
        } // does it have a skeleton?
    
      } // loop over nodes
  
    } // if debug skeleton scan


    MPI_Barrier(comm);
    if(rank == 0) cout<<"rank "<<rank<<" exchange_let.exchange_node_gid done "<<omp_get_wtime()-start_t<<endl;
    start_t = omp_get_wtime();


    // using a brace to free memory 
    {
    //cout << "exchanging points\n";
    // 2. exchange all request point gid, put them into let tree
    // format: np, gids, ..., np, gids, charges (if skeleton)...

    vector<int> node_rank(ntotal_recv);
    int prank = 0;
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < recv_count[i]; j++) {
            node_rank[prank++] = i;
        }
    }

    // figure out what points we need to send
    vector<double> send_request_point_info;
    send_request_point_info.reserve(recv_node_gids.size() * numof_points_of_share_leaf);
    memset(send_count, 0, sizeof(int)*size);

    for(int i = 0; i < recv_node_gids.size(); i++) {
      fks_ompNode *curr = let_node(recv_node_gids[i].third);
      // if its a leaf, send all its points
      if(recv_node_gids[i].first) {
        send_request_point_info.push_back( (double)curr->leaf_point_gids.size() );
        for(int j = 0; j < curr->leaf_point_gids.size(); j++) {
          send_request_point_info.push_back( (double)(curr->leaf_point_gids[j]) );
        }
        send_count[node_rank[i]] += (curr->leaf_point_gids.size()+1);
      }
      else {
        // if its a skeleton, send the points and charges
        //cout << "skel: " << curr->skeletons << "\n";
        int nskeletons = 0;
        if(curr->skeletons != NULL) nskeletons = curr->skeletons->numof_points;
        send_request_point_info.push_back( (double)nskeletons );
        if(curr->skeletons != NULL) {
            for(int j = 0; j < curr->skeletons->numof_points; j++)
                send_request_point_info.push_back( (double)curr->skeletons->gids[j] );
            for(int j = 0; j < curr->skeletons->numof_points; j++)
                send_request_point_info.push_back( curr->skeletons->charges[j] );
        }
        send_count[node_rank[i]] += (2*nskeletons+1);
      }
    }   // end for node_recv


    if(_DEBUG_LET_) {
        if(rank == 0) cout<<endl;
        MPI_Barrier(comm);

        for(int r  = 0; r < size; r++) {
            if(r == rank) {
                cout<<"(rank "<<rank<<") : send_count: ";
                for(int i = 0; i < size; i++) {
                    cout<<send_count[i]<<" ";
                }
                cout<<endl;
            }
            cout.flush();
            MPI_Barrier(comm);
        }

        for(int r  = 0; r < size; r++) {
            if(r == rank) {
                cout<<"(rank "<<rank<<") : send_request_point_info: ";
                for(int i = 0; i < send_request_point_info.size(); i++) {
                    cout<<send_request_point_info[i]<<" ";
                }
                cout<<endl;
            }
            cout.flush();
            MPI_Barrier(comm);
        }
    }

    //cout << "rank " << rank << " exchanging counts\n";
    MPI_Alltoall( send_count, 1, MPI_INT, recv_count, 1, MPI_INT, comm );
    ntotal_recv = recv_count[0];
    send_disp[0] = 0;
    recv_disp[0] = 0;
    // scan counts
    for(int l = 1; l < size; l++) {
      ntotal_recv += recv_count[l];
      send_disp[l] = send_disp[l-1] + send_count[l-1];
      recv_disp[l] = recv_disp[l-1] + recv_count[l-1];
    }

    // exchange point gids
    vector<double> recv_request_point_info(ntotal_recv);
    //cout << "rank " << rank << " exchanging coordinates\n";
    MPI_Alltoallv( &(send_request_point_info[0]), send_count, send_disp, MPI_DOUBLE,
    &(recv_request_point_info[0]), recv_count, recv_disp, MPI_DOUBLE, comm);


    if(_DEBUG_LET_) {
        for(int r  = 0; r < size; r++) {
            if(r == rank) {
                cout<<"(rank "<<rank<<") : recv_request_point_info ";
                for(int i = 0; i < recv_request_point_info.size(); i++) {
                    cout<<recv_request_point_info[i]<<" ";
                }
                cout<<endl;
            }
            cout.flush();
            MPI_Barrier(comm);
        }
    }


    if (check_sizes)
    {
      cout << "rank " << rank << ": before put into let tree of exchange_let check.\n";

      for (int i = 0; i < letNodeList.size(); i++)
      {
    
        fks_ompNode* node = letNodeList[i];
    
        if (node && node->skeletons)
        {
      
          if (skeleton_sizes[i] != node->skeletons->numof_points)
          {
            cout << "Rank " << rank << ": failing on node " << i << ". size: " << skeleton_sizes[i] << ", numof_points: " << node->skeletons->numof_points << "\n";
            assert(skeleton_sizes[i] == node->skeletons->numof_points);
          }
        } // does it have a skeleton?
    
      } // loop over nodes
  
    } // if debug skeleton scan



    //cout  << "rank " << rank << " putting in LET tree\n";
    // .2.2 put into let tree
    int offset = 0;
    for(int i = 0; i < request_node_gids.size(); i++) {
      fks_ompNode *curr_let = let_node(request_node_gids[i].third);
      long np = (long)recv_request_point_info[offset];
      offset++;
      if(request_node_gids[i].first) {
        curr_let->leaf_point_gids.resize(np);
        for(int j = 0; j < np; j++)
          curr_let->leaf_point_gids[j] = (long)recv_request_point_info[offset+j];
        offset += np;
      }
      else {
        //cout << "\nadding skeleton for node " << request_node_gids[i].third << "\n";
        if(curr_let->skeletons == NULL)
          curr_let->skeletons = new fksData();
        curr_let->skeletons->numof_points = np;
        //cout << "num points: " << np << "\n\n";
        curr_let->skeletons->gids.resize(np);
        for(int j = 0; j < np; j++)
          curr_let->skeletons->gids[j] = (long)recv_request_point_info[offset+j];
        offset += np;
        curr_let->skeletons->charges.resize(np);
        for(int j = 0; j < np; j++)
          curr_let->skeletons->charges[j] = recv_request_point_info[offset+j];
        offset += np;
      }
    } // loop over all requested gids

  }   // embrace vector to release memory


  if (check_sizes)
  {
    cout << "rank " << rank << ": after put into let tree of exchange_let check.\n";

    for (int i = 0; i < letNodeList.size(); i++)
    {
    
      fks_ompNode* node = letNodeList[i];
    
      if (node && node->skeletons)
      {
      
        if (skeleton_sizes[i] != node->skeletons->numof_points)
        {
          cout << "Rank " << rank << ": failing on node " << i << ". size: " << skeleton_sizes[i] << ", numof_points: " << node->skeletons->numof_points << "\n";
          assert(skeleton_sizes[i] == node->skeletons->numof_points);
        }
      } // does it have a skeleton?
    
    } // loop over nodes
  
  } // if debug skeleton scan


    MPI_Barrier(comm);
    if(rank == 0) cout<<"rank "<<rank<<" exchange_let.exchange_point_gid done "<<omp_get_wtime()-start_t<<endl;
    start_t = omp_get_wtime();


  // Now, we need to coordinates of points we didn't already have 
  //cout  << "rank " << rank << " finalizing request gids\n";
  // .3 finalize request_gids
  // fks_ompNode need to have point_gids as well
  vector<long> request_gids;
  int p2 = 0;
  // need this check because we might not have requested any nodes -- this comes
  // up in the test point evaluation
  if (request_node_gids.size() > 0) {
    for(int r = 0; r < size; r++) {
      vector<long> tmp;
      // added checks for self rank and to make sure we don't go past the end of the array
      while(request_node_gids[p2].second == r && r != rank && p2 < request_node_gids.size()) {
        fks_ompNode *curr_let = let_node(request_node_gids[p2].third);
        //cout << "rank: " << rank << " curr_let: " << curr_let << "\n";
        if(request_node_gids[p2].first) {
          for(int j = 0; j < curr_let->leaf_point_gids.size(); j++) {
            // check if we already have it
            if(pos(curr_let->leaf_point_gids[j]) < 0)
              tmp.push_back(curr_let->leaf_point_gids[j]);
          }
        }
        else {
          //cout << "rank: " << rank << " skel: " << curr_let->skeletons << "\n";
          //cout << "rank: " << rank << " request node gids: " << request_node_gids[p2].second << ", " << request_node_gids[p2].third << "\n";
          if (curr_let->skeletons != NULL) {
            for(int j = 0; j < curr_let->skeletons->gids.size(); j++) {
              if(pos(curr_let->skeletons->gids[j]) < 0)
                tmp.push_back(curr_let->skeletons->gids[j]);
            }
          } // check if the skeleton is NULL
        }
        p2++;
      } // while we're on this rank
      if (tmp.size() > 0) {
        omp_par::merge_sort(tmp.begin(), tmp.end());
        vector<long>::iterator it = unique(tmp.begin(), tmp.end());
        tmp.resize(it-tmp.begin());
        request_gids.insert(request_gids.end(), tmp.begin(), tmp.end());
      }
      send_count[r] = tmp.size();

    } // loop over ranks
  } // if there are any requested nodes
  else {
    // need this to avoid a null pointer in Alltoallv
    //request_gids.push_back(0);
    memset(send_count, 0, sizeof(int)*size);
  }

  //cout  << "rank " << rank << " sending requests\n";
  // 4. send request to other process
  MPI_Alltoall( send_count, 1, MPI_INT, recv_count, 1, MPI_INT, comm );
  ntotal_recv = recv_count[0];
  send_disp[0] = 0;
  recv_disp[0] = 0;
  for(int l = 1; l < size; l++) {
    ntotal_recv += recv_count[l];
    send_disp[l] = send_disp[l-1] + send_count[l-1];
    recv_disp[l] = recv_disp[l-1] + recv_count[l-1];
  }
  vector<long> recv_gids(ntotal_recv);

  // cout<<"rank "<<rank<<": recv_gids.size() = "<<recv_gids.size()<<endl;


  MPI_Alltoallv( &(request_gids[0]), send_count, send_disp, MPI_LONG,
  &(recv_gids[0]), recv_count, recv_disp, MPI_LONG, comm);

  // if (request_gids[0] == -1)
  // {
  //   request_gids.clear();
  // }


    MPI_Barrier(comm);
    if(rank == 0) cout<<"rank "<<rank<<" exchange_let.unique_point_gid done "<<omp_get_wtime()-start_t<<endl;
    start_t = omp_get_wtime();



  //cout << "rank " << rank << " distributing data\n";
  // 5. distribute data, charge, morton, coord
  for(int i = 0; i < size; i++)
    send_count[i] = recv_count[i]*(dim+2);
  vector<double> send_points(recv_gids.size()*(dim+2));
  for(int i = 0; i < recv_gids.size(); i++) {
    int idx = pos(recv_gids[i]);
    send_points[i*(dim+2)+0] = inProcData->charges[idx];
    send_points[i*(dim+2)+1] = inProcData->mortons[idx];
    memcpy( &(send_points[i*(dim+2)+2]), &(inProcData->X[idx*dim]), sizeof(double)*dim );
  }
  MPI_Alltoall( send_count, 1, MPI_INT, recv_count, 1, MPI_INT, comm );
  long n_total_recv = recv_count[0];
  send_disp[0] = 0;
  recv_disp[0] = 0;
  for(int l = 1; l < size; l++) {
    n_total_recv += recv_count[l];
    send_disp[l] = send_disp[l-1] + send_count[l-1];
    recv_disp[l] = recv_disp[l-1] + recv_count[l-1];
  }
  vector<double> recv_points(n_total_recv);
  MPI_Alltoallv( send_points.data(), send_count, send_disp, MPI_DOUBLE,
  recv_points.data(), recv_count, recv_disp, MPI_DOUBLE, comm);


    MPI_Barrier(comm);
    if(rank == 0) cout<<"rank "<<rank<<" exchange_let.exchange_point_coordinates done "<<omp_get_wtime()-start_t<<endl;
    start_t = omp_get_wtime();


  //cout << "rank " << rank << " inserting data in inProcData\n";
  // 6. put data in inProcData
  int nexist = inProcData->numof_points;
  inProcData->numof_points = nexist + request_gids.size();
  inProcData->X.resize( (nexist+request_gids.size())*dim )  ;
  inProcData->mortons.resize( nexist + request_gids.size() )  ;
  inProcData->charges.resize( nexist + request_gids.size() )  ;
  inProcData->gids.resize( nexist + request_gids.size() )  ;

    MPI_Barrier(comm);
    if(rank == 0) cout<<"rank "<<rank<<" exchange_let.put_point done "<<omp_get_wtime()-start_t<<endl;
    start_t = omp_get_wtime();


  if (_DEBUG_LET_)
    cout << "rank " << rank << " resizing in exchange_let. nexist " << nexist << ", new: " << request_gids.size() << "\n";

  for(int i = 0; i < request_gids.size(); i++) {
    inProcData->gids[nexist+i] = request_gids[i];
    inProcData->charges[nexist+i] = recv_points[i*(dim+2)+0];
    inProcData->mortons[nexist+i] = (long)recv_points[i*(dim+2)+1];
    memcpy( &(inProcData->X[(nexist+i)*dim]), &(recv_points[i*(dim+2)+2]), sizeof(double)*dim);
    inProcMap.insert(make_pair(request_gids[i], nexist+i));
  }


  delete [] send_count;
  delete [] send_disp;
  delete [] recv_count;
  delete [] recv_disp;

  //cout << "rank " << rank << " finished exchanging LET\n";

  dummy_np = inProcData->numof_points;  // max_np, min_np, avg_np;
  MPI_Allreduce(&dummy_np, &max_np, 1, MPI_LONG, MPI_MAX, comm);
  MPI_Allreduce(&dummy_np, &min_np, 1, MPI_LONG, MPI_MIN, comm);
  MPI_Allreduce(&dummy_np, &avg_np, 1, MPI_LONG, MPI_SUM, comm);
  avg_np /= size;
  if(rank == 0) {
    cout<<"after exchange let, \tinProcData: \tmin = "<<min_np
          <<", \tmax = "<<max_np<<", \tavg = "<<avg_np<<endl;
  }

} // exchange let


// this version is for calling during UpdateCharges
void fksTree::exchange_updated_let(set< triple<long, long, int> > &set_skeletons,
                      vector<int>& skeleton_sizes, bool check_sizes)
{

  int size, rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  int dim = inProcData->dim;

  int *send_count = new int [size];
  int *send_disp = new int [size];
  int *recv_count = new int [size];
  int *recv_disp = new int [size];

  // Should I just store these somewhere? They're kind of big though?


  // .1 request_node_gids: triple<leaf_or_not, rank, node_gid>
  // set_leaves/skels <gid, morton, level>
  int p = 0;
  // These are the nodes we're requesting
  vector< triple<bool, int, long> > request_node_gids;
  request_node_gids.resize(set_skeletons.size());
  // Set up request for each LET skeleton
  for(set< triple<long, long, int> >::iterator it = set_skeletons.begin(); it != set_skeletons.end(); it++) {
    int myrank = morton_to_rank(it->second, it->third);
    if(myrank != rank) {
      request_node_gids[p].first = false; // false indicates a skeleton
      request_node_gids[p].second = myrank;
      request_node_gids[p].third = it->first;
      p++;
    }
  }
  request_node_gids.resize(p);    // because delete those in proc
  // - sort by rank
  omp_par::merge_sort(request_node_gids.begin(), request_node_gids.end(), triple<bool, int, long>::secondLess);
  // - exchange node id

  // fill out send count for each rank
  memset(send_count, 0, sizeof(int)*size);
  for(int i = 0; i < request_node_gids.size(); i++)
    send_count[ request_node_gids[i].second ]++;

  // exchange send counts
  MPI_Alltoall( send_count, 1, MPI_INT, recv_count, 1, MPI_INT, comm );
  
  // Scan recv count 
  int ntotal_recv = recv_count[0];
  send_disp[0] = 0;
  recv_disp[0] = 0;
  for(int l = 1; l < size; l++) {
    ntotal_recv += recv_count[l];
    send_disp[l] = send_disp[l-1] + send_count[l-1];
    recv_disp[l] = recv_disp[l-1] + recv_count[l-1];
  }

  // vector of nodes we'll receive
  vector< triple<bool, int, long> > recv_node_gids(ntotal_recv);

  // create a type for node triple
  MPI_Datatype let_node_type;
  MPI_Type_contiguous(sizeof(triple<bool, int, long>), MPI_BYTE, &let_node_type);
  MPI_Type_commit(&let_node_type);
  
  // exhange node info
  MPI_Alltoallv(&(request_node_gids[0]), send_count, send_disp, let_node_type,
                &(recv_node_gids[0]), recv_count, recv_disp, let_node_type, comm);

  // cout << "Rank " << rank << " Node requests exchanged.\n";

    // using a brace to free memory 
  {

    //cout << "exchanging points\n";
    // 2. exchange all request point gid, put them into let tree
    // format: np, gids, ..., np, gids, charges (if skeleton)...

    vector<int> node_rank(ntotal_recv);
    int prank = 0;
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < recv_count[i]; j++) {
            node_rank[prank++] = i;
        }
    }
    
    // figure out what points we need to send
    vector<double> send_request_point_info;
    send_request_point_info.reserve(recv_node_gids.size() * numof_points_of_share_leaf);
    memset(send_count, 0, sizeof(int)*size);
    memset(recv_count, -1, sizeof(int) * size);
    
    for(int i = 0; i < recv_node_gids.size(); i++) {
      fks_ompNode *curr = let_node(recv_node_gids[i].third);

      // if its a skeleton, send the points and charges
      //cout << "skel: " << curr->skeletons << "\n";
      int nskeletons = 0;
      if(curr->skeletons != NULL) nskeletons = curr->skeletons->numof_points;

      assert(!curr->skeletons->cant_prune);

      if (_DEBUG_LET_) {
        if (nskeletons == 0)
        {
          cout << "Rank " << rank << " sending node global_node_id " << curr->global_node_id << ", i: " << i << ", recv_node_gids[i].third " << recv_node_gids[i].third << ", lid " << curr->lnid << ", morton " << curr->node_morton;
          cout << "\nSending " << nskeletons << " points.\n";
          assert(0 == 1);
        }

        // Going to print all of the nodes that rank 0 is packing on level 4
        // if (rank == 0 && curr->level == 4 && check_sizes)
//         {
//           cout << "Rank 0 sending node global_node_id " << curr->global_node_id << ", i: " << i << ", recv_node_gids[i].third " << recv_node_gids[i].third << ", lid " << curr->lnid << ", morton " << curr->node_morton;
//           cout << "\nSending " << nskeletons << " points, skeleton_sizes: " << skeleton_sizes[curr->lnid] << "\n";
//}
      }
      
      send_request_point_info.push_back( (double)nskeletons );
      if(curr->skeletons != NULL) {
          // for(int j = 0; j < curr->skeletons->numof_points; j++)
        //    send_request_point_info.push_back( (double)curr->skeletons->gids[j] );
          for(int j = 0; j < curr->skeletons->numof_points; j++)
              send_request_point_info.push_back( curr->skeletons->charges[j] );
      }
      // send_count[node_rank[i]] += (2*nskeletons+1);
      send_count[node_rank[i]] += (nskeletons + 1); // not sending gids any more
    
    }   // end for node_recv
    
    // cout << "Rank " << rank << " Exchanging counts of request fulfillment.\n";
    
    //cout << "rank " << rank << " exchanging counts\n";
    MPI_Alltoall( send_count, 1, MPI_INT, recv_count, 1, MPI_INT, comm );
    int ntotal_send = send_count[0];
    ntotal_recv = recv_count[0];
    send_disp[0] = 0;
    recv_disp[0] = 0;
    // scan counts
    for(int l = 1; l < size; l++) {
      ntotal_send += send_count[l];
      ntotal_recv += recv_count[l];
      send_disp[l] = send_disp[l-1] + send_count[l-1];
      recv_disp[l] = recv_disp[l-1] + recv_count[l-1];
    }
    
    assert(send_request_point_info.size() == ntotal_send);
    
    if (_DEBUG_LET_) {
      for(int r  = 0; r < size; r++) {
      
        if(r == rank) {
        
          cout << "\nRank " << rank << " SEND sizes: \n";
          for (int j = 0; j < size; j++)
          {
            cout << send_count[j] << ", ";
          }
          cout << "\n\n";
        
          cout << "Rank " << rank << " RECV sizes: \n";
          for (int j = 0; j < size; j++)
          {
            cout << recv_count[j] << ", ";
          }
          cout << "\n\n";
      
        }
      
        cout.flush();
        MPI_Barrier(comm);
    
      }
    }    
    
    // exchange point gids
    // setting this to default to something non-useful to see if I'm really
    // receiving zeros here
    vector<double> recv_request_point_info(ntotal_recv, -1.11111);

    // cout << "Rank " << rank << " Exchanging data.\n";
    
    //cout << "rank " << rank << " exchanging coordinates\n";
    // Trying to take this out for debugging purposes
    MPI_Alltoallv( &(send_request_point_info[0]), send_count, send_disp, MPI_DOUBLE,
   &(recv_request_point_info[0]), recv_count, recv_disp, MPI_DOUBLE, comm);
   
    if (check_sizes)
    {
      cout << "rank " << rank << ": before put into let tree of exchange_let check.\n";

      for (int i = 0; i < letNodeList.size(); i++)
      {
    
        fks_ompNode* node = letNodeList[i];
    
        if (node && node->skeletons)
        {
      
          if (skeleton_sizes[i] != node->skeletons->numof_points)
          {
            cout << "Rank " << rank << ": failing on node " << i << ". size: " << skeleton_sizes[i] << ", numof_points: " << node->skeletons->numof_points << "\n";
            assert(skeleton_sizes[i] == node->skeletons->numof_points);
          }
        } // does it have a skeleton?
    
      } // loop over nodes
  
    } // if debug skeleton scan


    // cout << "Rank " << rank << " filling in LET nodes locally.\n";

    //cout  << "rank " << rank << " putting in LET tree\n";
    // .2.2 put into let tree
    int offset = 0;
    int old_offset;
    for(int i = 0; i < request_node_gids.size(); i++) {
      fks_ompNode *curr_let = let_node(request_node_gids[i].third);
      long np = (long)recv_request_point_info[offset];
      offset++;
      
      //cout << "\nadding skeleton for node " << request_node_gids[i].third << "\n";
      if(curr_let->skeletons == NULL)
        curr_let->skeletons = new fksData();
      curr_let->skeletons->numof_points = np;
      
      if (_DEBUG_LET_ && np <= 0)
      {
        cout << "Rank " << rank << " receiving node global_node_id " << curr_let->global_node_id << ", i: " << i << ", request_node_gids[i].third " << request_node_gids[i].third << ", lid " << curr_let->lnid << ", morton " << curr_let->node_morton;
        assert(0 == 1);
      }
      
      old_offset = offset; // store the offset at the start of the last iteration for debugging
      //curr_let->skeletons->charges.resize(np);
      for(int j = 0; j < np; j++)
      {
        curr_let->skeletons->charges[j] = recv_request_point_info[offset+j];
        // if (rank == 0)
//  cout << "charge: " << curr_let->skeletons->charges[j] << "\n";
      }
      offset += np;
      
    } // loop over all requested gids

  }   // embrace vector to release memory


  if (check_sizes)
  {
    cout << "rank " << rank << ": after put into let tree of exchange_let check.\n";

    for (int i = 0; i < letNodeList.size(); i++)
    {
    
      fks_ompNode* node = letNodeList[i];
    
      if (node && node->skeletons)
      {
      
        if (skeleton_sizes[i] != node->skeletons->numof_points)
        {
          cout << "\n\n\nRank " << rank << ": failing on node " << i << ". size: " << skeleton_sizes[i] << ", numof_points: " << node->skeletons->numof_points << "\n";
          print_tree_single(node);

          cout << "\nSkeleton: charge_vec: " << node->skeletons->charges.size() << ", gids: " << node->skeletons->gids.size() << "\n";
          cout << "Charges: ";
          for (int j = 0; j < node->skeletons->charges.size(); j++)
          {
            cout << node->skeletons->charges[j] << ", ";
          }
          cout << "\n";
          cout << "gids: ";
          for (int j = 0; j < node->skeletons->gids.size(); j++)
          {
            cout << node->skeletons->gids[j] << ", ";
          }
          cout << "\n\n\n";          

          assert(skeleton_sizes[i] == node->skeletons->numof_points);

        }

      } // does it have a skeleton?
    
    } // loop over nodes
  
  } // if debug skeleton scan


  delete [] send_count;
  delete [] send_disp;
  delete [] recv_count;
  delete [] recv_disp;

}



int fksTree::morton_to_rank(long node_morton, int node_level)
{
  int check_level = node_level > (depth_mpi-1) ? (depth_mpi-1) : node_level;
  int myrank = 0;
  for(int i = 0; i <= check_level; i++) {
    myrank += (int)pow(2.0, depth_mpi-1-i)*(check_bit(node_morton, i));
  }
  return myrank;
}

long fksTree::morton_to_lid(long node_morton, int node_level)
  // local id from left to right (start from 0) at level 'node_level'
{
  long lid = 0;
  for(int i = 0; i <= node_level; i++) {
    lid += (long)pow(2.0, node_level-i)*(check_bit(node_morton, i));
  }
  return lid;
}

long fksTree::morton_to_gid(long node_morton, int node_level)
{
  long offset = (long)pow(2.0, node_level) - 1;
  return (offset + morton_to_lid(node_morton, node_level));
}


int fksTree::check_bit(long number, int bit)
{
  int result = number & (1 << bit);
  if(result == 0) {
    return 0;
  } else {
    return 1;
  }
}

// direct_ints is the list of direct interactions for this target point -- we
// store the global node ids, then change them over later
void fksTree::pruneByKNN(long node_morton, int node_level,
  long *knn_morton_ids, int k, int min_skeleton_global_level,
  set< triple<long, long, int> > &set_leaves,
  set< triple<long, long, int> > &set_skeletons, 
  vector<long>& direct_ints,
  vector<long>& approx_ints,
    vector<pair<long, int> >& skeleton_frontier)
{
  if( !prunable(node_morton, node_level, knn_morton_ids, k, min_skeleton_global_level) ) {
    if(node_level == (depth_let-1)) {
      long node_gid = morton_to_gid(node_morton, node_level);
      set_leaves.insert(triple<long, long, int>(node_gid, node_morton, node_level));
      // In order to simplify the collection of test points later, we ensure 
      // that if we have a leaf node, we also have its skeleton
      // WARNING: this might be an issue if we can't skeletonize a leaf in the 
      // adaptive level restriction algorithm
      set_skeletons.insert(triple<long, long, int>(node_gid, node_morton, node_level));
      
      // Add to the interaction list as well
      direct_ints.push_back(node_gid);
      
      return;
    }
    long morton_left = ( node_morton & (~(1 << (node_level+1))) );
    pruneByKNN(morton_left, node_level+1, knn_morton_ids, k,
                min_skeleton_global_level, set_leaves, set_skeletons, 
                direct_ints, approx_ints, skeleton_frontier);
    long morton_right = (node_morton | (1 <<(node_level+1)));
    pruneByKNN(morton_right, node_level+1, knn_morton_ids, k,
                min_skeleton_global_level, set_leaves, set_skeletons,
                direct_ints, approx_ints, skeleton_frontier);
  }
  else {

    // Need to check that the node is prunable
    // skeleton_frontier is sorted by morton ids, so we need to find where the 
    // ID of this node fits in it and make sure that its not above the frontier

    pair<long, int> node_pair = make_pair(node_morton, node_level);
    vector<pair<long, int> >::iterator it;
    it = lower_bound(skeleton_frontier.begin(), skeleton_frontier.end(), node_pair, askit::LessThanTreeOrder);

    // if the node or its ancestor appears in the skeleton_frontier, then it
    // should be the lower bound, so we're ok to prune
    // otherwise, we'll keep splitting
    if (it != skeleton_frontier.end() && (it->first == node_pair.first || isAncestor(*it, node_pair)))
    {
    
      long node_gid = morton_to_gid(node_morton, node_level);
      set_skeletons.insert(triple<long, long, int>(node_gid, node_morton, node_level));

      // As above, we are now collecting the leaf whenever we want it's skeleton
      // This avoids a corner case in the construction of the point interaction
      // lists -- the skeleton points basic charges would appear much earlier
      // in the charge table than the non-skeleton points, and I'm not sure of a
      // clean way to handle that right now -- Bill
      if (node_level == (depth_let - 1))
        set_leaves.insert(triple<long, long, int>(node_gid, node_morton, node_level));

      // add to the interaction list
      approx_ints.push_back(node_gid);

      return;
    }
    else
    {
      // we're above the skeleton frontier, so we have to recurse
      long morton_left = ( node_morton & (~(1 << (node_level+1))) );
      pruneByKNN(morton_left, node_level+1, knn_morton_ids, k,
                  min_skeleton_global_level, set_leaves, set_skeletons,
                  direct_ints, approx_ints, skeleton_frontier);
      long morton_right = (node_morton | (1 <<(node_level+1)));
      pruneByKNN(morton_right, node_level+1, knn_morton_ids, k,
                  min_skeleton_global_level, set_leaves, set_skeletons,
                  direct_ints, approx_ints, skeleton_frontier);
    }
  }
}


bool fksTree::prunable(long node_morton, int node_level,
long *knn_morton_ids, int k, int min_skeleton_global_level)
{
  if(node_level < min_skeleton_global_level)
    return false;

  bool isprunable = true;
  for(int i = 0; i < k; i++) {
    isprunable = isprunable && (!belong2Node(node_morton, node_level, knn_morton_ids[i]));
    if( !isprunable ) break;
  }
  return isprunable;
}

bool fksTree::belong2Node(long node_morton, int node_level, long point_morton)
{
  if(node_level == 0)
    return true;

  int nbits = node_level + 1;
  long mask = ~( (~0) << nbits );
  long node_bits = node_morton & mask;
  long point_bits = point_morton & mask;
  return (node_bits == point_bits);
}

fks_ompNode* fksTree::insert_node_by_morton(fks_ompNode *inNode, long morton, int level)
{
  fks_ompNode *curr = inNode;

  if(inNode == NULL || inNode->level == depth_let-1) {
    return curr;
  }

  while(curr->level < level) {
    if(check_bit(morton, curr->level+1) == 1) {     // go right
      if(curr->rightNode == NULL) {
        curr->rightNode = new fks_ompNode();
        curr->rightNode->level = curr->level+1;
        curr->rightNode->parent = curr;
        curr->rightNode->global_node_id = 2*curr->global_node_id+2;
        long mymorton = curr->node_morton;
        curr->rightNode->node_morton = (mymorton | (1 << (curr->level+1)) );
        curr->rightNode->lnid = 2*curr->lnid+1;
      }
      curr = curr->rightNode;
    }
    else {  // go left
      if(curr->leftNode == NULL) {
        curr->leftNode = new fks_ompNode();
        curr->leftNode->level = curr->level+1;
        curr->leftNode->parent = curr;
        curr->leftNode->global_node_id = 2*curr->global_node_id+1;
        long mymorton = curr->node_morton;
        curr->leftNode->node_morton = ( mymorton & (~(1 << (curr->level+1))) );
        curr->leftNode->lnid = 2*curr->lnid+0;
      }
      curr = curr->leftNode;
    }
  }
  return curr;
}

void fksTree::level_order_traversal(fks_ompNode *root)
{
  int idx = 0;
  fks_ompNode *curr = root;
  queue<fks_ompNode *> myqueue;
  myqueue.push(curr);
  while( !myqueue.empty() ) {
    // dequeue the front node
    curr = myqueue.front();
    myqueue.pop();
    letNodeList.push_back(curr);
    letNodeMap.insert(make_pair(curr->global_node_id, idx));
    // I'll use this to reference the list later -- Bill
    // I don't think this is used anywhere else
    curr->lnid = idx;
    idx++;

    // enqueue left child
    if(curr->leftNode != NULL)
      myqueue.push(curr->leftNode);

    // enqueue right child
    if(curr->rightNode != NULL)
      myqueue.push(curr->rightNode);
  }
}


void fksTree::update_let_node_list(fks_ompNode *root)
{
  int idx = letNodeList.size();
  fks_ompNode *curr = root;
  queue<fks_ompNode *> myqueue;
  myqueue.push(curr);
  while( !myqueue.empty() ) {
    // dequeue the front node
    curr = myqueue.front();
    myqueue.pop();

    map<long, int>::iterator it = letNodeMap.find(curr->global_node_id);
    if(it == letNodeMap.end()) {    // do not exist, append it
        letNodeList.push_back(curr);
        letNodeMap.insert(make_pair(curr->global_node_id, idx));
        // I'll use this to reference the list later -- Bill
        // I don't think this is used anywhere else
        curr->lnid = idx;
        idx++;
    }

    // enqueue left child
    if(curr->leftNode != NULL)
      myqueue.push(curr->leftNode);

    // enqueue right child
    if(curr->rightNode != NULL)
      myqueue.push(curr->rightNode);
  }
}


void fksTree::printLET_preorder(fks_ompNode *inNode)
{
  if(NULL == inNode->leftNode && NULL == inNode->rightNode) {
    cout<<"@leafnode,\tlevel = "<<inNode->level
      <<",\tglb id = "<<inNode->global_node_id
        <<",\tlnid = "<<inNode->lnid
          <<",\tmorton = "<<inNode->node_morton
            <<",\tmyrank = "<<inNode->myrank
              <<",\tneed skeleton = "<<inNode->skeleton_request
                <<",\tskeleton_ptr = "<<inNode->skeletons
                  <<",\tlocalroot = "<<inNode->isLocalRoot
                    <<",\tneed leaf = "<<inNode->leaf_request
                      <<",\tleaf size = "<<inNode->leaf_point_gids.size()
                        <<endl;
    /*
    if(inNode->skeletons != NULL) {
    int dim = inNode->skeletons->dim;
    cout<<"\tskeletons: "<<endl;
    for(int i = 0; i < inNode->skeletons->numof_points; i++) {
    cout<<"\t("<<inNode->skeletons->charges[i]<<"): ";
    cout<<"\t("<<inNode->skeletons->gids[i]<<"). ";
    //for(int j = 0; j < inNode->skeletons->dim; j++)
    //    cout<<inNode->skeletons->X[i*dim+j]<<" ";
    cout<<endl;
    }
    }
    */
    return;
  }
  else {
    cout<<"@internal,\tlevel = "<<inNode->level
      <<",\tglb id = "<<inNode->global_node_id
        <<",\tlnid = "<<inNode->lnid
          <<",\tmorton = "<<inNode->node_morton
            <<",\tmyrank = "<<inNode->myrank
              <<",\tneed skeleton = "<<inNode->skeleton_request
                <<",\tskeleton_ptr = "<<inNode->skeletons
                  <<",\tlocalroot = "<<inNode->isLocalRoot
                    <<endl;
    /*
    if(inNode->skeletons != NULL) {
    cout<<"\tskeletons: "<<endl;
    int dim = inNode->skeletons->dim;
    for(int i = 0; i < inNode->skeletons->numof_points; i++) {
    cout<<"\t("<<inNode->skeletons->charges[i]<<"): ";
    cout<<"\t("<<inNode->skeletons->gids[i]<<"). ";
    //for(int j = 0; j < inNode->skeletons->dim; j++)
    //    cout<<inNode->skeletons->X[i*dim+j]<<" ";
    cout<<endl;
    }
    }
    */

    printLET_preorder(inNode->leftNode);
    printLET_preorder(inNode->rightNode);
  }
}


// ======================== Testing point functions ================
void fksTree::DistributeTestingPoints(fksData *inData, vector< pair<double, long> > *inKNN)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int k = inKNN->size() / inData->numof_points;
    int dim = inData->dim;
    long numof_testing_points = inData->numof_points;

    if(_DEBUG_TEST_POINT_ && 0) {
        if(rank == 0) cout<<"inProcData and inProcKNN: "<<endl;
        for(int r = 0; r < size; r++) {
            if(r == rank) {
                for(int i = 0; i < numof_points_of_dist_leaf; i++) {
                    cout<<"(rank "<<rank<<") "<<inProcData->gids[i]<<": ";
                    for(int j = 0; j < dim; j++) {
                        cout<<inProcData->X[i*dim+j]<<" ";
                    }
                    cout<<endl;
                    cout<<"(rank "<<rank<<") "<<inProcData->gids[i]<<": ";
                    for(int j = 0; j < k; j++) {
                        cout<<(*inProcKNN)[i*k+j].second<<" ";
                    }
                    cout<<endl;
                 }
            }
            cout.flush();
            MPI_Barrier(comm);
        }


        if(rank == 0) cout<<"input test data: "<<endl;
        for(int r = 0; r < size; r++) {
            if(r == rank) {
                for(int i = 0; i < inData->numof_points; i++) {
                    cout<<"(rank "<<rank<<") "<<inData->gids[i]<<": ";
                    for(int j = 0; j < dim; j++)
                        cout<<inData->X[i*dim+j]<<" ";
                    cout<<endl;
                    cout<<"(rank "<<rank<<") "<<inData->gids[i]<<": ";
                    for(int j = 0; j < k; j++) {
                        cout<<(*inKNN)[i*k+j].second<<" ";
                    }
                    cout<<endl;
                }
            }
            cout.flush();
            MPI_Barrier(comm);
        }
        if(rank == 0) cout<<endl;
    }

    // if inProcTestKNN and inProcTestData is not set to be NULL, release memory
    if(inProcTestData != NULL) {
        delete inProcTestData;
        inProcTestData = NULL;
    }
    if(inProcTestKNN != NULL) {
        delete inProcTestKNN;
        inProcTestKNN = NULL;
    }

    if (size == 1)
    {
      //inProcTestData = inData;
      inProcTestData = new fksData();
      inProcTestData->Copy(inData);

      inProcTestKNN = new vector<pair<long, long> >();
      inProcTestKNN->resize(inKNN->size());

      for (int i = 0; i < inProcTestData->numof_points; i++)
      {
        inProcTestMap.insert(make_pair(inProcTestData->gids[i], (int)i));
        for (int j = 0; j < k; j++)
        {
          int nn_lid = pos((*inKNN)[i*k + j].second);
          (*inProcTestKNN)[i*k + j] = make_pair((*inKNN)[i*k + j].second, inProcData->mortons[nn_lid]);
        }
      }

      return;

    }

    // 0. assgin internal_origin_ranks and internal_origin_mortons
    int divd = glbN / size;
    int rem = glbN % size;
    long numof_original_points = rank < rem ? (divd+1) : divd;
    long original_gid_offset;
    MPI_Scan(&numof_original_points, &original_gid_offset, 1, MPI_LONG, MPI_SUM, comm);
    original_gid_offset -= numof_original_points;

    internal_origin_ranks.resize(numof_original_points);
    vector<double> current_ranks(numof_points_of_dist_leaf);
    vector<double> origin_ranks(numof_original_points);
    for(int i = 0; i < numof_points_of_dist_leaf; i++)
        current_ranks[i] = (double)rank;
    shuffle_back( numof_points_of_dist_leaf, &(current_ranks[0]), &(inProcData->gids[0]),
                  numof_original_points, &(origin_ranks[0]), comm );
    for(int i = 0; i < numof_original_points; i++)
        internal_origin_ranks[i] = (int)origin_ranks[i];

    internal_origin_mortons.resize(numof_original_points);
    vector<double> current_mids(numof_points_of_dist_leaf);
    vector<double> origin_mids(numof_original_points);
    for(int i = 0; i < numof_points_of_dist_leaf; i++)
        current_mids[i] = (double)inProcData->mortons[i];
    shuffle_back( numof_points_of_dist_leaf, &(current_mids[0]), &(inProcData->gids[0]),
                  numof_original_points, &(origin_mids[0]), comm );
    for(int i = 0; i < numof_original_points; i++)
        internal_origin_mortons[i] = (int)origin_mids[i];

    if(_DEBUG_TEST_POINT_) {
        for(int r = 0; r < size; r++) {
            if(r == rank) {
                cout<<"tree leaf data on rank "<<rank<<": ";
                for(int i = 0; i < numof_points_of_dist_leaf; i++) {
                    cout<<inProcData->gids[i]<<"-"<<inProcData->mortons[i]<<"  ";
                }
                cout<<endl;
            }
            cout.flush();
            MPI_Barrier(comm);
        }

        if(rank == 0) cout<<endl;
        for(int r = 0; r < size; r++) {
            if(r == rank) {
                cout<<"original data destination and morton on "<<rank<<": ";
                for(int i = 0; i < internal_origin_ranks.size(); i++) {
                    cout<<original_gid_offset+i<<"-"<<internal_origin_ranks[i]<<"-"<<internal_origin_mortons[i]<<"  ";
                }
                cout<<endl;
            }
            cout.flush();
            MPI_Barrier(comm);
        }
        if(rank == 0) cout<<endl;
    }


    // 1. determine the home rank of each testing point by 1st nn
    int *send_count = new int [size];
    int *recv_count = new int [size];
    int *send_disp = new int [size];
    int *recv_disp = new int [size];
    // 1.1. send point to original rank to know where is the 1st nn now
    long glb_numof_testing_points = 0;
    MPI_Allreduce(&numof_testing_points, &glb_numof_testing_points, 1, MPI_LONG, MPI_SUM, comm);
    // triple<long, int, int> (global_id, local_id, target_rank)
    vector< triple<long, int, int> > send_1st_nn(numof_testing_points);
    //triple<long, int, int> *send_1st_nn = new triple<long, int, int> [numof_testing_points];
    for(long i = 0; i < numof_testing_points; i++) {
        send_1st_nn[i].first = (*inKNN)[i*k+0].second;
        send_1st_nn[i].second = i;
        send_1st_nn[i].third = -1;
    }

    if(_DEBUG_TEST_POINT_) {
        if(rank == 0) cout<<endl;
        for(int r = 0; r < size; r++) {
            if(r == rank) {
                cout<<"send_1st_nn on rank "<<rank<<" (test_global_id, 1st_nn_gid, test_local_id): "<<endl;
                for(int i = 0; i < send_1st_nn.size(); i++) {
                    cout<<i+original_gid_offset<<"-"<<send_1st_nn[i].first
                        <<"-"<<send_1st_nn[i].second<<"  ";
                }
                cout<<endl;
            }
            cout.flush();
            MPI_Barrier(comm);
        }
    }

    omp_par::merge_sort(send_1st_nn.begin(), send_1st_nn.end(), triple<long, int, int>::firstLess);


    if(_DEBUG_TEST_POINT_) {
        if(rank == 0) cout<<endl;
        for(int r = 0; r < size; r++) {
            if(r == rank) {
                cout<<"send_1st_nn on rank "<<rank<<" (test_global_id, 1st_nn_gid, test_local_id): "<<endl;
                for(int i = 0; i < send_1st_nn.size(); i++) {
                    cout<<send_1st_nn[i].first<<"-"<<send_1st_nn[i].second<<"  ";
                }
                cout<<endl;
            }
            cout.flush();
            MPI_Barrier(comm);
        }
    }

    memset(send_count, 0, sizeof(int)*size);
    for(long i = 0; i < numof_testing_points; i++) {
        int target_rank = knn::home_rank(glbN, size, send_1st_nn[i].first);
        send_count[target_rank]++;
    }
    MPI_Alltoall( send_count, 1, MPI_INT, recv_count, 1, MPI_INT, comm );
    long n_recv = recv_count[0];
    send_disp[0] = 0;
    recv_disp[0] = 0;
    for(int i = 1; i < size; i++) {
        n_recv += recv_count[i];
        send_disp[i] = send_disp[i-1] + send_count[i-1];
        recv_disp[i] = recv_disp[i-1] + recv_count[i-1];
    }
    vector< triple<long, int, int> > recv_1st_nn(n_recv);
    MPI_Datatype mpi_1st_nn_msgtype;
    MPI_Type_contiguous(sizeof(triple<long, int, int>), MPI_BYTE, &mpi_1st_nn_msgtype);
    MPI_Type_commit(&mpi_1st_nn_msgtype);
    MPI_Alltoallv( &(send_1st_nn[0]), send_count, send_disp, mpi_1st_nn_msgtype,
                   &(recv_1st_nn[0]), recv_count, recv_disp, mpi_1st_nn_msgtype, comm );


    if(_DEBUG_TEST_POINT_) {
        if(rank == 0) cout<<endl;
        for(int r = 0; r < size; r++) {
            if(r == rank) {
                cout<<"recv_1st_nn on rank "<<rank<<"(1st_nn_gid, test_local_id): "<<endl;
                for(int i = 0; i < recv_1st_nn.size(); i++) {
                    cout<<recv_1st_nn[i].first<<"-"<<recv_1st_nn[i].second<<"  ";
                }
                cout<<endl;
            }
            cout.flush();
            MPI_Barrier(comm);
        }
    }


    // 1.2 target_rank
    vector<int> send_1st_nn_rank(n_recv);
    vector<int> recv_1st_nn_rank(numof_testing_points);
    for(int i = 0; i < send_1st_nn_rank.size(); i++) {
        send_1st_nn_rank[i] = internal_origin_ranks[ recv_1st_nn[i].first - original_gid_offset ];
    }
    MPI_Alltoallv( &(send_1st_nn_rank[0]), recv_count, recv_disp, MPI_INT,
                   &(recv_1st_nn_rank[0]), send_count, send_disp, MPI_INT, comm );


    if(_DEBUG_TEST_POINT_) {
        if(rank == 0) cout<<endl;
        for(int r = 0; r < size; r++) {
            if(r == rank) {
                cout<<"testing points destination on rank (1st_nn_gid, rank, test_local_id)"<<rank<<": ";
                for(int i = 0; i < recv_1st_nn_rank.size(); i++) {
                    cout<<send_1st_nn[i].first<<"-"<<recv_1st_nn_rank[i]<<"-"<<send_1st_nn[i].second<<"  ";
                }
                cout<<endl;
            }
            cout.flush();
            MPI_Barrier(comm);
        }
    }


    // 2. redistrubte testing point and its nn to the target rank
    for(int i = 0; i < send_1st_nn.size(); i++) {
        send_1st_nn[i].third = recv_1st_nn_rank[i];
    }
    omp_par::merge_sort(send_1st_nn.begin(), send_1st_nn.end(), triple<long, int, int>::thirdLess);


    if(_DEBUG_TEST_POINT_) {
        if(rank == 0) cout<<endl;
        for(int r = 0; r < size; r++) {
            if(r == rank) {
                cout<<"send_1st_nn on rank "<<rank<<" (1st_nn_gid, test_local_id, target_rank): "<<endl;
                for(int i = 0; i < send_1st_nn.size(); i++) {
                    cout<<send_1st_nn[i].first<<"-"<<send_1st_nn[i].second<<"-"<<send_1st_nn[i].third<<"  ";
                }
                cout<<endl;
            }
            cout.flush();
            MPI_Barrier(comm);
        }
    }


    // testing_point_gid, coordinates, knn_gids
    long msg_dim = 1 + dim + k;
    vector<double> send_testing_points(numof_testing_points*msg_dim);
    memset(send_count, 0, sizeof(int)*size);
    for(long i = 0; i < send_1st_nn.size(); i++) {
        int lid = send_1st_nn[i].second;
        send_count[ send_1st_nn[i].third ]++;
        send_testing_points[msg_dim*i+0] = (double)inData->gids[lid];
        // copy coord
        for(long j = 0; j < inData->dim; j++) {
            send_testing_points[msg_dim*i+1+j] = inData->X[lid*dim+j];
        }
        // copy nn gid
        for(long t = 0; t < k; t++) {
            send_testing_points[msg_dim*i+1+dim+t] = (double)(*inKNN)[lid*k+t].second;
        }
    }
    MPI_Alltoall( send_count, 1, MPI_INT, recv_count, 1, MPI_INT, comm );
    n_recv = recv_count[0];
    send_disp[0] = 0;
    recv_disp[0] = 0;
    for(int i = 1; i < size; i++) {
        n_recv += recv_count[i];
        send_disp[i] = send_disp[i-1] + send_count[i-1];
        recv_disp[i] = recv_disp[i-1] + recv_count[i-1];
    }
    vector<double> recv_testing_points(n_recv*msg_dim);
    for(int i = 0; i < size; i++) {
        send_count[i] *= msg_dim;
        recv_count[i] *= msg_dim;
        send_disp[i] *= msg_dim;
        recv_disp[i] *= msg_dim;
    }
    MPI_Alltoallv( &(send_testing_points[0]), send_count, send_disp, MPI_DOUBLE,
                   &(recv_testing_points[0]), recv_count, recv_disp, MPI_DOUBLE, comm );
    inProcTestData = new fksData();
    inProcTestData->dim = dim;
    inProcTestData->numof_points = n_recv;
    inProcTestData->gids.resize(n_recv);
    inProcTestData->mortons.resize(n_recv);
    inProcTestData->X.resize(n_recv*dim);
    inProcTestKNN = new vector< pair<long, long> >();
    inProcTestKNN->resize(n_recv*k);

    // cout << "Rank " << rank << " receiving " << n_recv << " test points.\n";

    // assgin morton later when i receive the nn morton id
    for(long i = 0; i < n_recv; i++) {
        inProcTestData->gids[i] = recv_testing_points[i*msg_dim+0];

        inProcTestMap.insert(make_pair(inProcTestData->gids[i], (int)i));

        for(long j = 0; j < dim; j++) {
            inProcTestData->X[i*dim+j] = recv_testing_points[i*msg_dim+1+j];
        }

        for(long j = 0; j < k; j++) {
            (*inProcTestKNN)[i*k+j].first = recv_testing_points[i*msg_dim+1+dim+j];
        }
    }


    if(_DEBUG_TEST_POINT_) {
        if(rank == 0) cout<<endl;
        for(int r = 0; r < size; r++) {
            if(r == rank) {
                cout<<"recieved data on "<<rank<<": "<<endl;
                for(int i = 0; i < inProcTestData->numof_points; i++) {
                    cout<<inProcTestData->gids[i]<<": ";
                    for(int j = 0; j < dim; j++) {
                        cout<<inProcTestData->X[i*dim+j]<<" ";
                    }
                    cout<<endl;
                }
                cout<<endl;
            }
            cout.flush();
            MPI_Barrier(comm);
        }
    }


    // 3. collect the morton id of all nn
    // 3.1 remove duplicate of all nn
    vector<long> send_nn_gids(inProcTestKNN->size());
    #pragma omp parallel for
    for(long i = 0; i < inProcTestKNN->size(); i++) {
        send_nn_gids[i] = (*inProcTestKNN)[i].first;
    }
    omp_par::merge_sort(send_nn_gids.begin(), send_nn_gids.end());
    vector<long>::iterator it = unique(send_nn_gids.begin(), send_nn_gids.end());
    send_nn_gids.resize(it-send_nn_gids.begin());
    // 3.2 send to origin to get morton id
    memset(send_count, 0, sizeof(int)*size);
    for(int i = 0; i < send_nn_gids.size(); i++) {
        int target_rank = knn::home_rank(glbN, size, send_nn_gids[i]);
        send_count[ target_rank ]++;
    }
    MPI_Alltoall( send_count, 1, MPI_INT, recv_count, 1, MPI_INT, comm );
    n_recv = recv_count[0];
    send_disp[0] = 0;
    recv_disp[0] = 0;
    for(int i = 1; i < size; i++) {
        n_recv += recv_count[i];
        send_disp[i] = send_disp[i-1] + send_count[i-1];
        recv_disp[i] = recv_disp[i-1] + recv_count[i-1];
    }
    vector<long> recv_nn_gids(n_recv);
    MPI_Alltoallv( &(send_nn_gids[0]), send_count, send_disp, MPI_LONG,
                   &(recv_nn_gids[0]), recv_count, recv_disp, MPI_LONG, comm );
    // 3.3 send morton id of all nn
    vector<long> send_morton_table(n_recv);
    for(int i = 0; i < n_recv; i++) {
        send_morton_table[i] = internal_origin_mortons[ recv_nn_gids[i] - original_gid_offset ];
    }
    vector<long> recv_morton_table(send_nn_gids.size());
    MPI_Alltoallv( &(send_morton_table[0]), recv_count, recv_disp, MPI_LONG,
                   &(recv_morton_table[0]), send_count, send_disp, MPI_LONG, comm );
    // 3.4 put morton back into inProcTestKNN and inProcTestData
    #pragma omp parallel for
    for(long i = 0; i < inProcTestKNN->size(); i++) {
        int idx = binary_search(send_nn_gids, (*inProcTestKNN)[i].first);
        (*inProcTestKNN)[i].second = recv_morton_table[idx];
    }
    // the first nn morton
    for(int i = 0; i < inProcTestData->numof_points; i++) {
        inProcTestData->mortons[i] = (*inProcTestKNN)[i*k+0].second;
    }

    if(_DEBUG_TEST_POINT_) {
        if(rank == 0) cout<<endl;
        for(int r = 0; r < size; r++) {
            if(r == rank) {
                cout<<"recieved data on "<<rank<<": "<<endl;
                for(int i = 0; i < inProcTestData->numof_points; i++) {
                    cout<<inProcTestData->gids[i]<<"("<<inProcTestData->mortons[i]<<"): ";
                    for(int j = 0; j < k; j++) {
                        cout<<(*inProcTestKNN)[i*k+j].first<<"("<<(*inProcTestKNN)[i*k+j].second<<") ";
                    }
                    cout<<endl;
                }
                cout<<endl;
            }
            cout.flush();
            MPI_Barrier(comm);
        }
    }

    delete [] send_count;
    delete [] recv_count;
    delete [] send_disp;
    delete [] recv_disp;

}


void fksTree::UpdateTestLET(vector<vector<long> >& direct_ints, vector<vector<long> >& approx_ints, long k)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    long total_k = inProcTestKNN->size() / inProcTestData->numof_points;

    if (k <= 0)
      k = total_k;

    if (rank == 0)
      cout << "rank " << rank << " Test k " << k << "\n";

    // set< triple<long, long, int> > testing_set_leaves;
    // set< triple<long, long, int> > testing_set_skeletons;

    long root_morton = 0;
    int root_level = 0;

    int nthreads = 1;
    #pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
    vector< set< triple<long, long, int> > > testing_set_leaves_multithreads(nthreads);
    vector< set< triple<long, long, int> > > testing_set_skeletons_multithreads(nthreads);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long *mortons_knn = new long [k];
        #pragma omp for
        for(long i = 0; i < inProcTestData->numof_points; i++) {
          for(long j = 0; j < k; j++) {
              mortons_knn[j] = (*inProcTestKNN)[i*total_k+j].second;
          }
          pruneByKNN(root_morton, root_level, mortons_knn, k, min_training_skeleton_level,
                     testing_set_leaves_multithreads[tid], testing_set_skeletons_multithreads[tid], direct_ints[i], approx_ints[i], skeleton_frontier);
        }
        delete [] mortons_knn;
    } // end of parallel block
    
    // merge results for each thread
    for(int i = 0; i < nthreads; i++) {
        for(set< triple<long, long, int> >::iterator it = testing_set_leaves_multithreads[i].begin(); it != testing_set_leaves_multithreads[i].end(); it++) {
            testing_set_leaves.insert(*it);
        }
    }
    for(int i = 0; i < nthreads; i++) {
        for(set< triple<long, long, int> >::iterator it = testing_set_skeletons_multithreads[i].begin(); it != testing_set_skeletons_multithreads[i].end(); it++) {
            testing_set_skeletons.insert(*it);
        }
    }
    
    

    if(_DEBUG_TEST_POINT_) {
        if(rank == 0) cout<<endl;
        for(int r = 0; r < size; r++) {
            if(r == rank) {
                cout<<"testing_set_leaves (node_gid, node_morton, node_level) on "<<rank<<": "<<endl;
                for(set< triple<long, long, int> >::iterator it = testing_set_leaves.begin();
                        it != testing_set_leaves.end(); it++) {
                    cout<<"("<<it->first<<","<<it->second<<","<<it->third<<")  ";
                }
                cout<<endl<<endl;

                cout<<"my_set_leaves (node_gid, node_morton, node_level) on "<<rank<<": "<<endl;
                for(set< triple<long, long, int> >::iterator it = my_set_leaves.begin();
                        it != my_set_leaves.end(); it++) {
                    cout<<"("<<it->first<<","<<it->second<<","<<it->third<<")  ";
                }
                cout<<endl<<endl;


                cout<<"testing_set_skeletons (node_gid, node_morton, node_level) on "<<rank<<": "<<endl;
                for(set< triple<long, long, int> >::iterator it = testing_set_skeletons.begin();
                        it != testing_set_skeletons.end(); it++) {
                    cout<<"("<<it->first<<","<<it->second<<","<<it->third<<")  ";
                }
                cout<<endl<<endl;

                cout<<"my_set_skeletons (node_gid, node_morton, node_level) on "<<rank<<": "<<endl;
                for(set< triple<long, long, int> >::iterator it = my_set_skeletons.begin();
                        it != my_set_skeletons.end(); it++) {
                    cout<<"("<<it->first<<","<<it->second<<","<<it->third<<")  ";
                }
                cout<<endl<<endl;

            }
            cout.flush();
            MPI_Barrier(comm);
        }
        if(rank == 0) cout<<endl<<endl;
        MPI_Barrier(comm);
    }

    // 2. remove all existing let node
    //for(set< triple<long, long, int> >::iterator it = testing_set_leaves.begin();
    //        it != testing_set_leaves.end(); it++) {
    //    set< triple<long, long, int> >::iterator it_found = my_set_leaves.find(*it);
    //    if(it_found != my_set_leaves.end()) {   // exist in my_set_leaves, do not recollect
    //        testing_set_leaves.erase(it);
    //    } else {
    //        my_set_leaves.insert(*it);
    //    }
    //}

    for(set< triple<long, long, int> >::iterator it = my_set_leaves.begin();
            it != my_set_leaves.end(); it++) {
        set< triple<long, long, int> >::iterator it_found = testing_set_leaves.find(*it);
        if(it_found != testing_set_leaves.end())
            testing_set_leaves.erase(it_found);
    }
    for(set< triple<long, long, int> >::iterator it = testing_set_leaves.begin();
            it != testing_set_leaves.end(); it++) {
            my_set_leaves.insert(*it);
    }


    //for(set< triple<long, long, int> >::iterator it = testing_set_skeletons.begin();
    //        it != testing_set_skeletons.end(); it++) {
    //    set< triple<long, long, int> >::iterator it_found = my_set_skeletons.find(*it);
    //    if(it_found != my_set_skeletons.end()) {
    //        testing_set_skeletons.erase(it);
    //    } else {
    //        my_set_skeletons.insert(*it);
    //    }
    //}

    for(set< triple<long, long, int> >::iterator it = my_set_skeletons.begin();
            it != my_set_skeletons.end(); it++) {
        set< triple<long, long, int> >::iterator it_found = testing_set_skeletons.find(*it);
        if(it_found != testing_set_skeletons.end())
            testing_set_skeletons.erase(it_found);
    }
    for(set< triple<long, long, int> >::iterator it = testing_set_skeletons.begin();
            it != testing_set_skeletons.end(); it++) {
            my_set_skeletons.insert(*it);
    }


    // -- since I insert testing nodes into the same set, if retrain the model, updateCharges would
    // collect those charges for testing nodes as well (it is correct, just collect more data)


    if(_DEBUG_TEST_POINT_) {
        if(rank == 0) cout<<endl;
        for(int r = 0; r < size; r++) {
            if(r == rank) {
                cout<<"testing_set_leaves (node_gid, node_morton, node_level) on "<<rank<<": "<<endl;
                for(set< triple<long, long, int> >::iterator it = testing_set_leaves.begin();
                        it != testing_set_leaves.end(); it++) {
                    cout<<"("<<it->first<<","<<it->second<<","<<it->third<<")  ";
                }
                cout<<endl;

                cout<<"testing_set_skeletons (node_gid, node_morton, node_level) on "<<rank<<": "<<endl;
                for(set< triple<long, long, int> >::iterator it = testing_set_skeletons.begin();
                        it != testing_set_skeletons.end(); it++) {
                    cout<<"("<<it->first<<","<<it->second<<","<<it->third<<")  ";
                }
                cout<<endl;
            }
            cout.flush();
            MPI_Barrier(comm);
        }
    }

    // 3. insert extra node in let node

    if(_DEBUG_TEST_POINT_) {
        if(rank == 0) cout<<endl<<"before insert let: "<<endl;
        MPI_Barrier(comm);
        print_tree(root_let, comm);
        if(rank == 0) cout<<endl<<endl;
    }

    if (size > 1) {

      for(set< triple<long, long, int> >::iterator it = testing_set_leaves.begin();
              it != testing_set_leaves.end(); it++) {
          insert_node_by_morton(root_let, it->second, it->third);
      }
      for(set< triple<long, long, int> >::iterator it = testing_set_skeletons.begin();
              it != testing_set_skeletons.end(); it++) {
          insert_node_by_morton(root_let, it->second, it->third);
      }

      if(_DEBUG_TEST_POINT_) {
          if(rank == 0) cout<<endl<<"after insert let: "<<endl;
          MPI_Barrier(comm);
          print_tree(root_let, comm);
          if(rank == 0) cout<<endl<<endl;
      }

      // 4. append extra nodes into the letNodeList and letNodeMap
      update_let_node_list(root_let);

      if(_DEBUG_TEST_POINT_ && 0) {
          MPI_Barrier(comm);
          if(rank == 0) cout<<endl;
          for(int r = 0; r < size; r++) {
              if(r == rank) {
                  cout<<"letNodeList after update on (gid, lid, lid by map) "<<rank<<": "<<endl;
                  for(int i = 0; i < letNodeList.size(); i++) {
                      cout<<letNodeList[i]->global_node_id<<", "
                          <<letNodeList[i]->lnid<<", "
                          <<letNodeMap[letNodeList[i]->global_node_id]<<endl;
                  }
                  cout<<endl;
              }
              cout.flush();
              MPI_Barrier(comm);
          }
      }



    // MPI_Barrier(comm);
  //   if(rank == 0) cout<<"let tree before exchange_let: "<<endl;
  //   print_tree(root_let, comm);
  //
  //   MPI_Barrier(comm);
  //   if(rank == 0) cout<<"inProcData: "<<endl;
  //   print_data(inProcData, comm);
  //
  //   MPI_Barrier(comm);
  //   if(rank == 0) cout<<"inProcTestData: "<<endl;
  //   print_data(inProcTestData, comm);



      // 5. collect points
      exchange_let(testing_set_leaves, testing_set_skeletons);

      if(_DEBUG_TEST_POINT_) {
          if(rank == 0) cout<<endl<<"after exchange_let: "<<endl;
          MPI_Barrier(comm);
          print_tree(root_let, comm);
          if(rank == 0) cout<<endl<<endl;

          if (rank == 0) cout << "\nLET Node List\n";
          MPI_Barrier(comm);
          print_let_node_list(letNodeList);
          if(rank == 0) cout<<endl<<endl;
      }

  } // more than one MPI rank
  
}


int fksTree::test_pos(long test_gid)
{
    map<long, int>::iterator it = inProcTestMap.find(test_gid);
    if(it != inProcTestMap.end()) {
        return it->second;
    } else {
        return -1;
    }
}














