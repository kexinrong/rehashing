#include "binTree.h"

#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <stdio.h>
#include <ompUtils.h>

#include "repartition.h"
#include "CmdLine.h"
#include "generator.h"
#include "clustering.h"
#include "direct_knn.h"
#include "mpitree.h"
#include "rotation.h"
#include "verbose.h"
#include "parallelIO.h"

#define _DEBUG_FKS_ false

using namespace std;
using namespace knn;
using namespace knn::repartition;

binNode::~binNode()
{
	if(NULL == this->kid && NULL != this->data) {
		delete this->data;
	}
	else {
		delete this->kid;
	}
	if(this->parent != NULL) {
		MPI_Barrier(comm);
		MPI_Comm_free(&comm);
	}
}


// hyper tree
void binNode::Insert(pbinNode in_parent, int maxp, int maxLevel, int minCommSize, MPI_Comm inComm, pbinData inData)
{
	double start_t, end_t;
	double stage_t = omp_get_wtime();

	int worldsize, worldrank;
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);

	// input checks
	int numof_kids = 2;
	assert( maxp > 1 );
	assert( maxLevel >= 0 && maxLevel <= options.max_max_treelevel);

	comm = inComm;
	int size; iM( MPI_Comm_size(comm, &size));
	int rank; iM( MPI_Comm_rank(comm, &rank)); // needed to print information
	int dim = inData->dim;
	MPI_CALL(MPI_Allreduce(&(inData->numof_points), &Nglobal, 1, MPI_INT, MPI_SUM, comm));

	vector<double> &X = inData->X;
	vector<long> &gids= inData->gids;

	// Initializations
	int its_child_id = 0;
	int n_over_p = Nglobal / size;

	#if STAGE_DTREE_OUTPUT_VERBOSE
		if(worldrank == 0) cout<<"   > Insert(hyper): level "<<level<<": before load balance -> "<<omp_get_wtime() - stage_t<<endl;
		stage_t = omp_get_wtime();
	#endif

	knn::repartition::loadBalance_arbitraryN(inData->X, inData->gids, inData->numof_points,
                                                inData->dim, inData->numof_points, comm);

    if(in_parent!=NULL)  {
		level = in_parent->level + 1;
		parent = in_parent;
		its_child_id = chid;
	}

	#if STAGE_DTREE_OUTPUT_VERBOSE
		if(worldrank == 0) cout<<"   > Insert(hyper): level "<<level<<": load balance done! -> "<<omp_get_wtime() - stage_t<<endl;
		stage_t = omp_get_wtime();
	#endif

    if(_DEBUG_FKS_) {
        MPI_Barrier(MPI_COMM_WORLD);
        if(worldrank == 0)
            cout<<"\t@rkdt_insert: level = "<<level<<" initilization done"<<endl;
    }


	// BASE CASE TO TERMINATE RECURSION
	if (size <= minCommSize ||                // don't want smaller processor partitions
			level == maxLevel ||              // reached max level
			Nglobal <= maxp					  // not enough points to further partition
		) {

		data = new binData;
		data->Copy(inData);
		data->X.resize(data->numof_points*dim);
		data->gids.resize(data->numof_points);
		//data = inData;

		MPI_Barrier(MPI_COMM_WORLD);

		#if STAGE_DTREE_OUTPUT_VERBOSE
            if(worldrank == 0) {
				cout<<"   > Insert(hyper): level "<<level<<": copy leaf data done! -> "<<omp_get_wtime() - stage_t
                    <<", size = "<<size<<", level = "<<level<<", Nglobal = "<<Nglobal
                    <<", minComm = "<<minCommSize<<", maxLevel = "<<maxLevel<<", mppn = "<<maxp
                    <<endl;
			}
		#endif

		return;

	}// end of base case

	int numof_clusters = 2;
	vector<int> point_to_cluster_membership(inData->numof_points);
	vector<int> local_numof_points_per_cluster(numof_clusters);
	vector<int> global_numof_points_per_cluster(numof_clusters);
	{ //Scope tmpX

		// 1. rotate points
		vector<double> tmpX;
		if(this->level == 0 && options.flag_r == 1
                && strcmp(options.splitter.c_str(), "rkdt") == 0 ) {		// root level, only rotate on root level
			tmpX.resize(inData->numof_points*dim);
			generateRotation(dim, rw, comm);
			memcpy(&(tmpX[0]), &(X[0]), sizeof(double)*inData->numof_points*dim);
			rotatePoints( &(tmpX[0]), inData->numof_points, dim, rw, &(X[0]) );		// distribute rotated points ("rkdt", flag_r=1)
			tmpX.clear();
		}
		else if(options.flag_r == 2										// rotate on each level
                && strcmp(options.splitter.c_str(), "rkdt") == 0 ) {
			tmpX.resize(inData->numof_points*dim);
			generateRotation(dim, rw, comm);
			rotatePoints( &(X[0]), inData->numof_points, dim, rw, &(tmpX[0]) );		// distributed original points, rotated points only used to partition points ("rkdt", flag_r=2)
		}

		stage_t = omp_get_wtime();

		// 2. split points into two partitions
		if(0 == strcmp(options.splitter.c_str(), "rsmt"))
		{
			coord_mv = -1;
			proj.resize(dim);
			mtreeSplitter(	&X[0], inData->numof_points, dim, &(proj[0]), median,
							&(point_to_cluster_membership[0]),
							&(local_numof_points_per_cluster[0]),
							&(global_numof_points_per_cluster[0]),
							comm);
		}


		if (0 == strcmp(options.splitter.c_str(), "rkdt"))
		{
			if(options.flag_r == 2) {
				proj.resize(dim);
				maxVarSplitter( &(tmpX[0]), inData->numof_points, dim, options.flag_c,
							coord_mv, median,
							&(point_to_cluster_membership[0]),
							&(local_numof_points_per_cluster[0]),
							&(global_numof_points_per_cluster[0]),
							comm);
				vector<double> e; e.resize(dim);
				e[coord_mv] = 1.0;
				RROT_INV( &(e[0]), &(proj[0]), &(rw[0]) );
				tmpX.clear();
			}
			else {
				proj.resize(dim);
				maxVarSplitter( &(X[0]), inData->numof_points, dim, options.flag_c,
							coord_mv, median,
							&(point_to_cluster_membership[0]),
							&(local_numof_points_per_cluster[0]),
							&(global_numof_points_per_cluster[0]),
							comm);
				proj[coord_mv] = 1.0;
			}
		}	// end if ("rkdt")
		tmpX.clear();
	}	// end scope


	#if TREE_DEBUG_VERBOSE
		cout<<worldrank<<" : membership: ";
		for(int i = 0; i < inData->numof_points; i++)
			cout<<point_to_cluster_membership[i]<<" ";
		cout<<endl;
		cout<<worldrank<<" : proj "<<options.splitter<<" : ";
		for(int i = 0; i < dim; i++)
			cout<<proj[i]<<" ";
		cout<<" median: "<<median;
		cout<<endl;
	#endif

	#if STAGE_OUTPUT_VERBOSE
		if(worldrank == 0) {
			cout<<"   > Insert(hyper): level "<<level<<": splitter done! "
				<<"  cluster size: "<<global_numof_points_per_cluster[0]
				<<" "<<global_numof_points_per_cluster[1]
				<<" -> "<<omp_get_wtime() - stage_t<<endl;
		}
		stage_t = omp_get_wtime();
		if(abs(global_numof_points_per_cluster[0] - global_numof_points_per_cluster[1]) > 2) {
			cout<<"!!!!!!!!! imbalance at level "<<level<<" - worldrank: "<<worldrank
				<<" - cluster size: "<<global_numof_points_per_cluster[0]
				<<" "<<global_numof_points_per_cluster[1]<<endl;
		}
		stage_t = omp_get_wtime();
	#endif


    if(_DEBUG_FKS_) {
        MPI_Barrier(MPI_COMM_WORLD);
        if(worldrank == 0)
            cout<<"\t@rkdt_insert: level = "<<level<<" hyper plane splitting done"<<endl;
    }


	int my_rank_color;
    if(size % 2 == 0)   // Even split
		my_rank_color = (rank < size/2) ? 0 : 1; // used to assign this rank to a processor
    else
		my_rank_color = (rank <= size/2) ? 0 : 1; // used to assign this rank to a processor
	rank_colors.resize(size);
	if(size % 2 == 0) {  // Even split
		for(int i = 0; i < size; i++)
			rank_colors[i] = (i < size/2) ? 0 : 1;
	} else {
        for(int i = 0; i < size; i++)
			rank_colors[i] = (i <= size/2) ? 0 : 1;

	}
	//MPI_CALL(MPI_Allgather(&my_rank_color, 1, MPI_INT, &(rank_colors[0]), 1, MPI_INT, comm));


	pre_all2all(&(gids[0]), &(point_to_cluster_membership[0]), &(X[0]), (long)(inData->numof_points), dim);

    stage_t = omp_get_wtime();

	#if COMM_TIMING_VERBOSE
		MPI_Barrier(comm);
		start_t = omp_get_wtime();
	#endif


	int newN = tree_repartition_arbitraryN(inData->gids, inData->X, inData->numof_points, 
                            &(point_to_cluster_membership[0]), &(rank_colors[0]), dim, comm);
	inData->numof_points = newN;

	#if PCL_DEBUG_VERBOSE
		MPI_Barrier(comm);
		cout<<"("<<worldrank<<")"<<" after repartition: ";
		for(int i = 0; i < inData->numof_points; i++)
			cout<<inData->gids[i]<<" ";
		cout<<endl;
	#endif

	#if COMM_TIMING_VERBOSE
		Repartition_Tree_Build_T_ += omp_get_wtime() - start_t;
	#endif


    if(_DEBUG_FKS_) {
        MPI_Barrier(MPI_COMM_WORLD);
        if(worldrank == 0)
            cout<<"\t@rkdt_insert: level = "<<level<<" redistribution done"<<endl;
    }


	stage_t = omp_get_wtime();

	//6. create new communicator
	#if COMM_TIMING_VERBOSE
		MPI_Barrier(comm);
		start_t = omp_get_wtime();
	#endif

	MPI_Comm new_comm = MPI_COMM_NULL;
	if( MPI_SUCCESS != MPI_Comm_split( comm, my_rank_color, rank, &new_comm) )
		assert(NULL);
	assert(new_comm != MPI_COMM_NULL);

	#if COMM_TIMING_VERBOSE
		Comm_Split_T_ += omp_get_wtime() - start_t;
	#endif

	//7. Create new node and insert new data
	kid = new binNode(its_child_id);
	kid->options.hypertree = options.hypertree;
	kid->options.flag_r = options.flag_r;
	kid->options.flag_c = options.flag_c;
	kid->options.pruning_verbose = options.pruning_verbose;
	kid->options.timing_verbose = options.timing_verbose;
	kid->options.splitter = options.splitter;
	kid->options.debug_verbose = options.debug_verbose;
	kid->Insert( this, maxp, maxLevel, minCommSize, new_comm, inData);

};


void binNode::InsertInMemory(pbinNode in_parent, int maxp, int maxLevel, int minCommSize, MPI_Comm inComm, pbinData inData, binData *datapool, vector<int> &gid2lid)
{
	comm = inComm;
	int size, rank;
    MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);
    int worldrank, worldsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldsize);

    int numof_kids = 2;
    int dim = inData->dim;
    long numof_points = inData->numof_points;
	long glb_numof_points;
    MPI_Allreduce(&numof_points, &glb_numof_points, 1, MPI_LONG, MPI_SUM, comm);
	Nglobal = glb_numof_points;

	vector<double> &X = inData->X;
	vector<long> &gids= inData->gids;

	int n_over_p = glb_numof_points / size;
	knn::repartition::loadBalance_arbitraryN(inData->X, inData->gids, inData->numof_points,
                                             inData->dim, inData->numof_points, comm);

	if (in_parent!=NULL)  {
		level = in_parent->level + 1;
		parent = in_parent;
	}


    // BASE CASE TO TERMINATE RECURSION
    if (size <= minCommSize || level == maxLevel || glb_numof_points <= maxp) {

        //cout<<"size = "<<size<<", mincommsize = "<<minCommSize
        //    <<" level = "<<level<<", maxlevel = "<<maxLevel
        //    <<" glb_n = "<<glb_numof_points<<", maxp = "<<maxp
        //    <<endl;

        data = new binData();
        data->dim = inData->dim;
        data->numof_points = inData->numof_points;
        data->gids.resize(inData->gids.size());
        memcpy( &(data->gids[0]), &(inData->gids[0]), sizeof(long)*inData->numof_points );

        datapool->dim = dim;
        int np = 0;
        for(int i = 0; i < inData->numof_points; i++) {
            if( gid2lid[inData->gids[i]] == -1 ) np++;
        }

        datapool->X.resize((datapool->numof_points+np)*dim);
        datapool->gids.resize(datapool->numof_points+np);
        int pp = datapool->numof_points;
        for(int i = 0; i < inData->numof_points; i++) {
            if( gid2lid[inData->gids[i]] == -1 ) {
                memcpy( &(datapool->X[pp*dim]), &(inData->X[i*dim]), dim*sizeof(double) );
                datapool->gids[pp] = inData->gids[i];
                gid2lid[inData->gids[i]] = pp;
                pp++;
            }
        }
        datapool->numof_points += np;
        return;
	}// end of base case

    // 0. generate rotation matrix
    if(this->level == 0) {
		generateRotation(dim, rw, comm);
    }
    else {
        rw.resize(in_parent->rw.size());
        memcpy(&(rw[0]), &(in_parent->rw[0]), sizeof(double)*rw.size());
    }

    // 1. generate projection direction
    int coid = rand() % dim;
    proj.resize(dim);
    memset( &(proj[0]), 0, sizeof(double)*dim);
    proj[coid] = 1.0;
    newRotatePoints( &(proj[0]), 1, dim, rw);

    // 2. project points
    vector<double> px;
    px.resize(inData->numof_points);
    int ONE = 1;
    #pragma omp parallel for
    for(int i = 0; i < inData->numof_points; i++) {
		px[i] = ddot(&dim, &(proj[0]), &ONE, &(inData->X[i*dim]), &ONE);
    }

    // 2. split points into two partitions
	int numof_clusters = 2;
	vector<int> point_to_cluster_membership(inData->numof_points);
	vector<int> local_numof_points_per_cluster(numof_clusters);
	vector<int> global_numof_points_per_cluster(numof_clusters);
	medianSplitter(px, median, &(point_to_cluster_membership[0]),
		&(local_numof_points_per_cluster[0]), &(global_numof_points_per_cluster[0]), comm);

    // 3. assign comm color
	int my_rank_color;
    if( size % 2 == 0)   // Even split
		my_rank_color = (rank < size/2) ? 0 : 1; // used to assign this rank to a processor
    else
		my_rank_color = (rank <= size/2) ? 0 : 1; // used to assign this rank to a processor
	rank_colors.resize(size);
    if( size % 2 == 0) {  // Even split
		for(int i = 0; i < size; i++)
			rank_colors[i] = (i < size/2) ? 0 : 1;
	}
    else {
        for(int i = 0; i < size; i++)
			rank_colors[i] = (i <= size/2) ? 0 : 1;
	}

    // 4. preprocess
	pre_all2all(&(gids[0]), &(point_to_cluster_membership[0]), &(X[0]), (long)(inData->numof_points), dim);

    // 5. repartition data
	int newN = tree_repartition_arbitraryN(inData->gids, inData->X, inData->numof_points,
                            &(point_to_cluster_membership[0]), &(rank_colors[0]), dim, comm);
	inData->numof_points = newN;

    // 6. split comm
	MPI_Comm new_comm = MPI_COMM_NULL;
	if( MPI_SUCCESS != MPI_Comm_split( comm, my_rank_color, rank, &new_comm) )
		assert(NULL);
	assert(new_comm != MPI_COMM_NULL);
	
	// 7. Create new node and insert new data
	kid = new binNode();
	kid->options.hypertree = options.hypertree;
	kid->options.flag_r = options.flag_r;
	kid->options.flag_c = options.flag_c;
	kid->options.pruning_verbose = options.pruning_verbose;
	kid->options.timing_verbose = options.timing_verbose;
	kid->options.splitter = options.splitter;
	kid->options.debug_verbose = options.debug_verbose;
	kid->InsertInMemory(this, maxp, maxLevel, minCommSize, new_comm, inData, datapool, gid2lid);
};



/*
// old tree 273-533
void binNode::Insert_oldtree(pbinNode in_parent, int maxp, int maxLevel, int minCommSize, MPI_Comm inComm, pbinData inData)
{

	double start_t, end_t;
	double stage_t;

	int worldsize, worldrank;
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);

	int numof_kids = 2;
	assert( maxp > 1 );
	assert( maxLevel >= 0 && maxLevel <= options.max_max_treelevel);

	comm = inComm;
	int size; iM( MPI_Comm_size(comm, &size));
	int rank; iM( MPI_Comm_rank(comm, &rank)); // needed to print information
	int dim = inData->dim;
	int N = inData->numof_points;
	MPI_CALL(MPI_Allreduce(&N, &Nglobal, 1, MPI_INT, MPI_SUM, comm));

	vector<double> &X = inData->X;
	vector<long> &gids= inData->gids;
	int its_child_id = 0;

	if (in_parent!=NULL)  {
		level = in_parent->level + 1;
		parent = in_parent;
		its_child_id = chid;
		N=inData->numof_points;
	}

	if(!worldrank && level == 0) cout<<"  old tree insert ..."<<endl;
	#if STAGE_OUTPUT_VERBOSE
		MPI_Barrier(comm);
		stage_t = omp_get_wtime();
	#endif

	// BASE CASE TO TERMINATE RECURSION
	if (size <= minCommSize ||                // don't want smaller processor partitions
			level == maxLevel ||              // reached max level
			Nglobal <= maxp					  // not enough points to further partition
		) {

		data = new binData;
		data->Copy(inData);
		data->X.resize(data->numof_points*dim);
		data->gids.resize(data->numof_points);

		//data = inData;

		MPI_Barrier(MPI_COMM_WORLD);

		#if STAGE_OUTPUT_VERBOSE
			if(worldrank == 0) {
				cout<<"   > Insert(old): level "<<level<<": copy leaf data done! -> "<<omp_get_wtime() - stage_t<<endl;
				cout<<"   == Insert Old Tree Done!"<<endl;
			}
		#endif
		
		return;
	
	}// end of base case 

	int numof_clusters = 2;
	vector<int> point_to_cluster_membership(N);
	vector<int> local_numof_points_per_cluster(numof_clusters);
	vector<int> global_numof_points_per_cluster(numof_clusters);
	
	{ //Scope tmpX
		// 1. rotation points
		vector<double> tmpX;
		if(this->level == 0 && options.flag_r == 1 
				&& strcmp(options.splitter.c_str(), "rkdt") == 0 ) {		// root level, only rotate on root level
			tmpX.resize(N*dim);
			generateRotation(dim, rw, comm);
			memcpy(&(tmpX[0]), &(X[0]), sizeof(double)*N*dim);
			rotatePoints( &(tmpX[0]), N, dim, rw, &(X[0]) );		// distribute rotated points ("rkdt", flag_r=1)
			tmpX.clear();
		}
		else if (options.flag_r == 2										// rotate on each level
				&& strcmp(options.splitter.c_str(), "rkdt") == 0 ) {
			tmpX.resize(N*dim);
			generateRotation(dim, rw, comm);
			rotatePoints( &(X[0]), N, dim, rw, &(tmpX[0]) );		// distributed original points, rotated points only used to partition points ("rkdt", flag_r=2)
		}
	
		#if STAGE_OUTPUT_VERBOSE
			MPI_Barrier(comm);
			if(worldrank == 0 && level < 3) cout<<"   > Insert(old): level "<<level<<": rotate points done! -> "<<omp_get_wtime() - stage_t<<endl;
			stage_t = omp_get_wtime();
		#endif
	
		// 2. split points into two partitions
		if(0 == strcmp(options.splitter.c_str(), "rsmt")) 
		{
			coord_mv = -1;
			proj.resize(dim);
			mtreeSplitter(	&X[0], N, dim, &(proj[0]), median,
							&(point_to_cluster_membership[0]),
							&(local_numof_points_per_cluster[0]), 
							&(global_numof_points_per_cluster[0]),
							comm);
	
		}
		if (0 == strcmp(options.splitter.c_str(), "rkdt")) 
		{
			if(options.flag_r == 2) {
				proj.resize(dim);
				maxVarSplitter( &(tmpX[0]), N, dim, options.flag_c,
							coord_mv, median,
							&(point_to_cluster_membership[0]),
							&(local_numof_points_per_cluster[0]), 
							&(global_numof_points_per_cluster[0]),
							comm);
				vector<double> e; e.resize(dim);
				e[coord_mv] = 1.0;
				RROT_INV( &(e[0]), &(proj[0]), &(rw[0]) );
				tmpX.clear();
			}
			else {
				proj.resize(dim);
				maxVarSplitter( &(X[0]), N, dim, options.flag_c,
							coord_mv, median,
							&(point_to_cluster_membership[0]),
							&(local_numof_points_per_cluster[0]), 
							&(global_numof_points_per_cluster[0]),
							comm);
				proj[coord_mv] = 1.0;
			}
		}	// end if ("rkdt")
		tmpX.clear();
	}


	#if STAGE_OUTPUT_VERBOSE
		MPI_Barrier(comm);
		stage_t = omp_get_wtime();
	#endif

	
	#if STAGE_OUTPUT_VERBOSE
		if(worldrank == 0) {
			cout<<"  = Build: level "<<level<<": redistribute to kid done! -> "<<omp_get_wtime() - stage_t
				<<"  cluster size: "<<global_numof_points_per_cluster[0]
				<<" "<<global_numof_points_per_cluster[1]
				<<" -> "<<omp_get_wtime() - stage_t<<endl;
		}
		if(abs(global_numof_points_per_cluster[0] - global_numof_points_per_cluster[1]) > 2) {
			cout<<"!!!!!!!!! imbalance at level "<<level<<" - worldrank: "<<worldrank
				<<" - cluster size: "<<global_numof_points_per_cluster[0]
				<<" "<<global_numof_points_per_cluster[1]
				<<endl;
		}
		stage_t = omp_get_wtime();
	#endif


	//3. given cluster and kid membership, fingure out processors to redistribute points
	int my_rank_color;
	int *point_to_rank_membership = new int[N];
	groupDistribute( &point_to_cluster_membership[0], 
					 N, numof_kids, 
					 comm, 
					 my_rank_color, 
					 point_to_rank_membership);
	rank_colors.resize(size);
	MPI_CALL(MPI_Allgather(&my_rank_color, 1, MPI_INT, &(rank_colors[0]), 1, MPI_INT, comm));
	its_child_id = numof_kids * its_child_id + my_rank_color;
	
	#if STAGE_OUTPUT_VERBOSE
		MPI_Barrier(comm);
		if(worldrank == 0 && level < 3) cout<<"   > Insert(old): level "<<level<<": groupDistribute done! -> "<<omp_get_wtime() - stage_t<<endl;
		stage_t = omp_get_wtime();
	#endif
	

	//4. repartition points given the new processor ids
	// it should be faster
	pre_all2all(&(gids[0]), point_to_rank_membership, &(X[0]), (long)N, dim);

	#if STAGE_OUTPUT_VERBOSE
		MPI_Barrier(comm);
		if(worldrank == 0 && level < 3) cout<<"   > Insert(old): level "<<level<<": pre_alltoall done! -> "<<omp_get_wtime() - stage_t<<endl;
		stage_t = omp_get_wtime();
	#endif
	
	double *new_X;
	long *new_gids;
	long new_N;
	vector<int> send_count(size);
	for(int i=0;i<N;i++) send_count[ point_to_rank_membership[i] ] ++;
	
	delete [] point_to_rank_membership;
	
	#if COMM_TIMING_VERBOSE
		MPI_Barrier(comm);
		start_t = omp_get_wtime();
	#endif

	knn::repartition::repartition( &(gids[0]), &(X[0]), long(N), &send_count[0], 
									dim, &new_gids, &new_X, &new_N, comm);
	
	#if COMM_TIMING_VERBOSE
		Repartition_Tree_Build_T_ += omp_get_wtime() - start_t;
	#endif

	#if STAGE_OUTPUT_VERBOSE
		MPI_Barrier(comm);
		if(worldrank == 0 && level < 3) cout<<"   > Insert(old): level "<<level<<": repartition done! -> "<<omp_get_wtime() - stage_t<<endl;
		stage_t = omp_get_wtime();
	#endif

	//5. Assign new data to the inData information
	inData->X.resize(new_N*dim);
	inData->gids.resize(new_N);
	inData->numof_points = new_N;
	int tnsize = new_N*dim;
	#pragma omp parallel for
	for (int i=0; i<tnsize; i++) 
		inData->X[i] = new_X[i];
	#pragma omp parallel for
	for (int i=0; i<new_N; i++) 
		inData->gids[i] = new_gids[i]; 
	delete [] new_X;
	delete [] new_gids;
	
	#if STAGE_OUTPUT_VERBOSE
		MPI_Barrier(comm);
		if(worldrank == 0 && level < 3) cout<<"   > Insert(old): level "<<level<<": copy repartitioned data done! -> "<<omp_get_wtime() - stage_t<<endl;
		stage_t = omp_get_wtime();
	#endif

	
	//6. create new communicator
	#if COMM_TIMING_VERBOSE
		MPI_Barrier(comm);
		start_t = omp_get_wtime();
	#endif
	MPI_Comm new_comm = MPI_COMM_NULL;
	if( MPI_SUCCESS != MPI_Comm_split( comm, my_rank_color, rank, &new_comm) )
		assert(NULL);
	assert(new_comm != MPI_COMM_NULL);
	#if COMM_TIMING_VERBOSE
		Comm_Split_T_ += omp_get_wtime() - start_t;
	#endif
	
	#if STAGE_OUTPUT_VERBOSE
		MPI_Barrier(comm);
		if(worldrank == 0 && level < 3) cout<<"   > Insert(old): level "<<level<<": creat new comm done! -> "<<omp_get_wtime() - stage_t<<endl;
		stage_t = omp_get_wtime();
	#endif
	
	//7. Create new node and insert new data
	kid = new binNode(its_child_id);
	kid->options.hypertree = options.hypertree;
	kid->options.flag_r = options.flag_r;
	kid->options.flag_c = options.flag_c;
	kid->options.pruning_verbose = options.pruning_verbose;
	kid->options.timing_verbose = options.timing_verbose;
	kid->options.splitter = options.splitter;
	kid->options.debug_verbose = options.debug_verbose;
	kid->Insert_oldtree( this, maxp, maxLevel, minCommSize, new_comm, inData);

};
*/



void binNode::parvar(double *points, int numof_points, int dim, double *mean, double *var) 
{
	int worldrank;
	MPI_Comm_rank( MPI_COMM_WORLD, &worldrank );
/*
        MPI_Barrier(MPI_COMM_WORLD);
        if(!worldrank) cout << worldrank << ": Beginning parvar" << endl;
*/

	int stride = dim + 128 / sizeof(double); //Make sure two threads don't write to same cache line.
	int maxt = omp_get_max_threads();

	vector<double> threadvar( stride * maxt );
	for(int i = 0; i < dim; i++) var[i] = 0.0;
	#pragma omp parallel if(numof_points > 1500)
	{
		int t = omp_get_thread_num();
		double *localvar = &(threadvar[t*stride]);

		for(int i = 0; i < dim; i++) localvar[i] = 0.0;
        int npdim = numof_points*dim;
       	register int j;
        #pragma omp for schedule(dynamic,25)
        for(int i = 0; i < numof_points; i++) {
           	register int idim = i*dim;
           	#pragma vector
           	for(j = 0; j < dim; j++) {
				double diff = points[idim+j] - mean[j];
				localvar[j] += diff*diff;
           	}
		}
	}
	for(int t = 0; t < maxt; t++) {
		double *localvar = &(threadvar[t*stride]);
		for(int i = 0; i < dim; i++)
			var[i] += localvar[i];
	}
/*
        MPI_Barrier(MPI_COMM_WORLD);
        if(!worldrank) cout << worldrank << ": Finished parvar" << endl;
*/
}


void binNode::medianSplitter(// input
			const vector<double> &px,
			// output
			double &medV,
			int* point_to_hyperplane_membership,
			int* local_numof_points_per_hyperplane,
			int* global_numof_points_per_hyperplane,
			MPI_Comm comm )
{
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	int worldrank;
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);

    int numof_points = px.size();
	int glb_numof_points = 0;
	MPI_CALL(MPI_Allreduce(&numof_points, &glb_numof_points, 1, MPI_INT, MPI_SUM, comm));

	// 1. find median
    vector<double> tmppx(numof_points);
    memcpy(&(tmppx[0]), &(px[0]), sizeof(double)*numof_points);
	medV = distributeSelect(tmppx, glb_numof_points/2, comm);
	assert(medV == medV); //make sure medV is not NaN;

	// 2. assign membership of each point
	local_numof_points_per_hyperplane[0] = 0;
	local_numof_points_per_hyperplane[1] = 0;
    int pointsperhp0 = 0, pointsperhp1 = 0;
	int nsame = 0;

	#pragma omp parallel for reduction(+:pointsperhp0,pointsperhp1,nsame)
	for(int i = 0; i < numof_points; i++) {
		double diff = fabs((px[i] - medV)/medV);
		if(diff < 1.0e-6 || isnan(diff) || isinf(diff)) {	// treat it as the same as median
			nsame++;
			point_to_hyperplane_membership[i] = -1;
		}
		else {
			if(px[i] < medV) {
				point_to_hyperplane_membership[i] = 0;
				pointsperhp0++;
			}
			else {
				point_to_hyperplane_membership[i] = 1;
				pointsperhp1++;
			}
		}
	}


	local_numof_points_per_hyperplane[0] = pointsperhp0;
	local_numof_points_per_hyperplane[1] = pointsperhp1;
	MPI_CALL(MPI_Allreduce(local_numof_points_per_hyperplane, global_numof_points_per_hyperplane, 2, MPI_INT, MPI_SUM, comm));

	// make splitting perfect load balanced

	int glb_nsame = 0;
	MPI_CALL(MPI_Allreduce(&nsame, &glb_nsame, 1, MPI_INT, MPI_SUM, comm));

	if(glb_nsame > 0) {

		int scan_nsame;
		MPI_Scan(&nsame, &scan_nsame, 1, MPI_INT, MPI_SUM, comm);

		int glb_n_move_left = glb_numof_points/2 - global_numof_points_per_hyperplane[0];
		int glb_n_move_right = glb_nsame - glb_n_move_left;
		double n_tmp_right = (double)glb_n_move_right*( (double)nsame/(double)glb_nsame );
		n_tmp_right = floor(n_tmp_right+0.5);
		int local_n_move_right = (int)(n_tmp_right);
		local_n_move_right = min(local_n_move_right, nsame);
		int local_n_move_left = nsame - local_n_move_right;
		int pmove = 0;

		if(scan_nsame < glb_n_move_left) {
			local_n_move_left = nsame;
			local_n_move_right = 0;
		}
		else {
			int excl_scan = scan_nsame - nsame;
			local_n_move_left = glb_n_move_left - excl_scan;
			if( local_n_move_left < 0 ) local_n_move_left = 0;
			local_n_move_right = nsame - local_n_move_left;
		}

		for(int i = 0; i < numof_points; i++) {
			if(point_to_hyperplane_membership[i] == -1) {
				if(pmove < local_n_move_left) {
					point_to_hyperplane_membership[i] = 0;
					local_numof_points_per_hyperplane[0]++;
				}
				else {
					point_to_hyperplane_membership[i] = 1;
					local_numof_points_per_hyperplane[1]++;
				}
				pmove++;
			}	// end if (-1)
		} // end for
		MPI_CALL(MPI_Allreduce(local_numof_points_per_hyperplane, global_numof_points_per_hyperplane, 2, MPI_INT, MPI_SUM, comm));
	}

}



// max variance coordinates splitter
void binNode::maxVarSplitter(// input
					double *points, int numof_points, int dim,
					int flag_c,
					// output
					int &coord_mv,
					double &medV,
					int* point_to_hyperplane_membership,
					int* local_numof_points_per_hyperplane,
					int* global_numof_points_per_hyperplane,
					MPI_Comm comm )
{
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	int worldrank;
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
	double stage_t = omp_get_wtime();

	//#if STAGE_OUTPUT_VERBOSE
	//	MPI_Barrier(comm); if(!worldrank)cout<<"worldrank: "<<worldrank<<" in splitter: "<<endl;
	//	stage_t = omp_get_wtime();
	//#endif

	int glb_numof_points = 0;
	MPI_CALL(MPI_Allreduce(&numof_points, &glb_numof_points, 1, MPI_INT, MPI_SUM, comm));
	int mvidx;

    if(flag_c == 0) {
		double ri = (double)rand() / (double)RAND_MAX;
		mvidx = (int)(ri*(double)dim);
		MPI_CALL(MPI_Bcast(&mvidx, 1, MPI_INT, 0, comm));
		coord_mv = mvidx;
	}
    else {
		// 1. find the coord has the maximum variance
		vector<double> local_var, glb_var;
		local_var.resize(dim);
		glb_var.resize(dim);
		double *glb_mean = centroids( points, numof_points, glb_numof_points, dim, comm);

		parvar(points, numof_points, dim, glb_mean, &(local_var[0]));
		MPI_CALL(MPI_Allreduce(&(local_var[0]), &(glb_var[0]), dim, MPI_DOUBLE, MPI_SUM, comm));
		vector<double>::iterator it = max_element(glb_var.begin(), glb_var.end());
		mvidx = it - glb_var.begin();
		coord_mv = mvidx;

		delete [] glb_mean;
	}


	// 2. find median
	vector<double> coords(numof_points);
	#pragma omp parallel for
	for(int i = 0; i < numof_points; i++)
		coords[i] = points[i*dim+coord_mv];

	medV = distributeSelect(coords, glb_numof_points/2, comm);
	assert(medV == medV);	// make sure medV is not NaN;

	// 3. assign membership of each point
	local_numof_points_per_hyperplane[0] = 0;
	local_numof_points_per_hyperplane[1] = 0;
    int pointsperhp0 = 0, pointsperhp1 = 0;
	int nsame = 0;

	#pragma omp parallel for reduction(+:pointsperhp0,pointsperhp1,nsame)
    for(int i = 0; i < numof_points; i++) {
		double diff = fabs((points[i*dim+mvidx] - medV)/medV);

		//if(diff < 2*DBL_EPSILON) {	// treat it as the same as median
		if(diff < 1.0e-6 || isnan(diff) ) {	// treat it as the same as median or medV is 0
			nsame++;
			point_to_hyperplane_membership[i] = -1;
		}
		else {
			if(points[i*dim+mvidx] < medV) {
				point_to_hyperplane_membership[i] = 0;
				pointsperhp0++;
			}
			else {
				point_to_hyperplane_membership[i] = 1;
				pointsperhp1++;
			}
		}
	}

	local_numof_points_per_hyperplane[0] = pointsperhp0;
	local_numof_points_per_hyperplane[1] = pointsperhp1;
	MPI_CALL(MPI_Allreduce(local_numof_points_per_hyperplane, global_numof_points_per_hyperplane, 2, MPI_INT, MPI_SUM, comm));

	// make splitting perfect load balanced

	int glb_nsame = 0;
	MPI_CALL(MPI_Allreduce(&nsame, &glb_nsame, 1, MPI_INT, MPI_SUM, comm));

	if(glb_nsame > 0) {

		int scan_nsame;
		MPI_Scan(&nsame, &scan_nsame, 1, MPI_INT, MPI_SUM, comm);

		int glb_n_move_left = glb_numof_points/2 - global_numof_points_per_hyperplane[0];
		int glb_n_move_right = glb_nsame - glb_n_move_left;
		double n_tmp_right = (double)glb_n_move_right*( (double)nsame/(double)glb_nsame );
		n_tmp_right = floor(n_tmp_right+0.5);
		int local_n_move_right = (int)(n_tmp_right);
		local_n_move_right = min(local_n_move_right, nsame);
		int local_n_move_left = nsame - local_n_move_right;
		int pmove = 0;

		if(scan_nsame < glb_n_move_left) {
			local_n_move_left = nsame;
			local_n_move_right = 0;
		}
		else {
			int excl_scan = scan_nsame - nsame;
			local_n_move_left = glb_n_move_left - excl_scan;
			if( local_n_move_left < 0 ) local_n_move_left = 0;
			local_n_move_right = nsame - local_n_move_left;
		}

		for(int i = 0; i < numof_points; i++) {
			if(point_to_hyperplane_membership[i] == -1) {
				if(pmove < local_n_move_left) {
					point_to_hyperplane_membership[i] = 0;
					local_numof_points_per_hyperplane[0]++;
				}
				else {
					point_to_hyperplane_membership[i] = 1;
					local_numof_points_per_hyperplane[1]++;
				}
				pmove++;
			}	// end if (-1)
		} // end for
		MPI_CALL(MPI_Allreduce(local_numof_points_per_hyperplane, global_numof_points_per_hyperplane, 2, MPI_INT, MPI_SUM, comm));
	}

	/*#if STAGE_OUTPUT_VERBOSE
	MPI_Barrier(comm);
		if(worldrank == 0 && level < 3) {
			cout<<"    >> : assign membership done! -> "<<omp_get_wtime() - stage_t<<endl;
		}
		stage_t = omp_get_wtime();
	#endif*/

}




// MT splitter
void binNode::mtreeSplitter( // input
					double * points, int numof_points, int dim,
					// output
					double *proj,
					double &medianValue,
					int* point_to_hyperplane_membership,
					int* local_numof_points_per_hyperplane,
					int* global_numof_points_per_hyperplane,
					MPI_Comm comm)
{
	int rank, nproc;
	MPI_Comm_rank(comm, &rank);
	int worldrank;
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);

	// .1 get projection direction
	getProjection(points, numof_points, dim, proj, comm);

	// .2 project points along the projection line
	vector<double> projValue;
	projValue.resize(numof_points);
	#pragma omp parallel for
    for(int i = 0; i < numof_points; i++) {
		for(int j = 0; j < dim; j++)
			projValue[i] += proj[j] * points[i*dim+j];
	}

	// .3 find the median of the projected values
	int glb_numof_points;
	MPI_CALL(MPI_Allreduce(&numof_points, &glb_numof_points, 1, MPI_INT, MPI_SUM, comm));
	medianValue = distributeSelect(projValue, glb_numof_points/2, comm);

	#if PCL_DEBUG_VERBOSE
		cout<<"("<<worldrank<<") projV - median: ";
		for(int i = 0; i < numof_points; i++)
			cout<<projValue[i] - medianValue<<" ";
		cout<<endl;
	#endif

	// .4 assign membership to each point
	local_numof_points_per_hyperplane[0] = 0;
	local_numof_points_per_hyperplane[1] = 0;
    int pointsperhp0 = 0, pointsperhp1 = 0, nsame=0;
	#pragma omp parallel for reduction(+:pointsperhp0,pointsperhp1,nsame)
    for(int i = 0; i < numof_points; i++) {
        double diff = fabs( (projValue[i]-medianValue) / medianValue );
        if( diff < 1.0e-6 || isnan(diff) ) {
            nsame++;
            point_to_hyperplane_membership[i] = -1;
        }
        else {
            if(projValue[i] < medianValue) {
			    point_to_hyperplane_membership[i] = 0;
			    pointsperhp0++;
		}
            else {
			    point_to_hyperplane_membership[i] = 1;
			    pointsperhp1++;
		    }
        }
	}

	local_numof_points_per_hyperplane[0] = pointsperhp0;
	local_numof_points_per_hyperplane[1] = pointsperhp1;
	MPI_CALL(MPI_Allreduce(local_numof_points_per_hyperplane, global_numof_points_per_hyperplane, 2, MPI_INT, MPI_SUM, comm));

    // .5 dealing with 'same' points
	int glb_nsame = 0;
	MPI_CALL(MPI_Allreduce(&nsame, &glb_nsame, 1, MPI_INT, MPI_SUM, comm));

	if(glb_nsame > 0) {

		int scan_nsame;
		MPI_Scan(&nsame, &scan_nsame, 1, MPI_INT, MPI_SUM, comm);

		int glb_n_move_left = glb_numof_points/2 - global_numof_points_per_hyperplane[0];
		int glb_n_move_right = glb_nsame - glb_n_move_left;
		double n_tmp_right = (double)glb_n_move_right*( (double)nsame/(double)glb_nsame );
		n_tmp_right = floor(n_tmp_right+0.5);
		int local_n_move_right = (int)(n_tmp_right);
		local_n_move_right = min(local_n_move_right, nsame);
		int local_n_move_left = nsame - local_n_move_right;
		int pmove = 0;

		if(scan_nsame < glb_n_move_left) {
			local_n_move_left = nsame;
			local_n_move_right = 0;
		}
		else {
			int excl_scan = scan_nsame - nsame;
			local_n_move_left = glb_n_move_left - excl_scan;
			if( local_n_move_left < 0 ) local_n_move_left = 0;
			local_n_move_right = nsame - local_n_move_left;
		}

		for(int i = 0; i < numof_points; i++) {
			if(point_to_hyperplane_membership[i] == -1) {
				if(pmove < local_n_move_left) {
					point_to_hyperplane_membership[i] = 0;
					local_numof_points_per_hyperplane[0]++;
				}
				else {
					point_to_hyperplane_membership[i] = 1;
					local_numof_points_per_hyperplane[1]++;
				}
				pmove++;
			}	// end if (-1)
		} // end for
		MPI_CALL(MPI_Allreduce(local_numof_points_per_hyperplane, global_numof_points_per_hyperplane, 2, MPI_INT, MPI_SUM, comm));
	}


	#if PCL_DEBUG_VERBOSE
		cout<<"("<<worldrank<<") membership: ";
		for(int i = 0; i < numof_points; i++)
			cout<<point_to_hyperplane_membership[i]<<" ";
		cout<<endl;
	#endif

}


/*
void binNode::splitByMedian(double *Pvalues, int numof_points, double median
                            int* point_to_hyperplane_membership,
                            int* local_numof_points_per_hyperplane,
                            int* global_numof_points_per_hyperplane,
                            MPI_Comm comm)
{
	// 1. assign membership of each point
	local_numof_points_per_hyperplane[0] = 0;
	local_numof_points_per_hyperplane[1] = 0;
    int pointsperhp0 = 0, pointsperhp1 = 0;
	int nsame = 0;

    // 2. split data accd. to < median, > median, and = median
	#pragma omp parallel for reduction(+:pointsperhp0,pointsperhp1,nsame)
	for(int i = 0; i < numof_points; i++) {
		double diff = fabs( (Pvalues[i]-median)/median );

        //if(diff < 2*DBL_EPSILON) {	// treat it as the same as median
        if(diff < 1.0e-6 || isnan(diff) ) {	// treat it as the same as median or median is 0
			nsame++;
			point_to_hyperplane_membership[i] = -1;
		}
        else {
            if( Pvalues[i] < median) {
				point_to_hyperplane_membership[i] = 0;
				pointsperhp0++;
			}
            else {
				point_to_hyperplane_membership[i] = 1;
				pointsperhp1++;
			}
		}
	}   // end for

	local_numof_points_per_hyperplane[0] = pointsperhp0;
	local_numof_points_per_hyperplane[1] = pointsperhp1;
	MPI_CALL(MPI_Allreduce(local_numof_points_per_hyperplane, global_numof_points_per_hyperplane, 2, MPI_INT, MPI_SUM, comm));

	// 3. make splitting perfect load balanced
	int glb_nsame = 0;
	MPI_CALL(MPI_Allreduce(&nsame, &glb_nsame, 1, MPI_INT, MPI_SUM, comm));
    if(glb_nsame > 0) {

        int scan_nsame;
		MPI_Scan(&nsame, &scan_nsame, 1, MPI_INT, MPI_SUM, comm);

		int glb_n_move_left = glb_numof_points/2 - global_numof_points_per_hyperplane[0];
		int glb_n_move_right = glb_nsame - glb_n_move_left;
		double n_tmp_right = (double)glb_n_move_right*( (double)nsame/(double)glb_nsame );
		n_tmp_right = floor(n_tmp_right+0.5);
		int local_n_move_right = (int)(n_tmp_right);
		local_n_move_right = min(local_n_move_right, nsame);
		int local_n_move_left = nsame - local_n_move_right;
		int pmove = 0;

		if(scan_nsame < glb_n_move_left) {
			local_n_move_left = nsame;
			local_n_move_right = 0;
		}
		else {
			int excl_scan = scan_nsame - nsame;
			local_n_move_left = glb_n_move_left - excl_scan;
			if( local_n_move_left < 0 ) local_n_move_left = 0;
			local_n_move_right = nsame - local_n_move_left;
		}

		for(int i = 0; i < numof_points; i++) {
			if(point_to_hyperplane_membership[i] == -1) {
				if(pmove < local_n_move_left) {
					point_to_hyperplane_membership[i] = 0;
					local_numof_points_per_hyperplane[0]++;
				}
				else {
					point_to_hyperplane_membership[i] = 1;
					local_numof_points_per_hyperplane[1]++;
				}
				pmove++;
			}	// end if (-1)
		} // end for
		MPI_CALL(MPI_Allreduce(local_numof_points_per_hyperplane, global_numof_points_per_hyperplane, 2, MPI_INT, MPI_SUM, comm));
	}   // end if(glb_nsame > 0)

}
*/



void binNode::furthestPoint(// input
					double *points, int numof_points, int dim, double *query,
					// output
					double *furP,
					MPI_Comm comm)
{
	int rank, nproc;
	MPI_Comm_size(comm, &nproc);
	MPI_Comm_rank(comm, &rank);

	double * dist = new double [numof_points];
	knn::compute_distances(points, query, numof_points, 1, dim, dist);

	double * pdmax = max_element(dist, dist+numof_points);
	int idmax = pdmax - dist;
	for(int i = 0; i < dim; i++)
		furP[i] = points[idmax*dim+i];

	double * dmaxg = new double [nproc];
	MPI_CALL(MPI_Allgather(pdmax, 1, MPI_DOUBLE, dmaxg, 1, MPI_DOUBLE, comm));
	double *pm = max_element(dmaxg, dmaxg+nproc);

	int rankmax = pm - dmaxg;
	MPI_Bcast(furP, dim, MPI_DOUBLE, rankmax, comm);

	delete [] dist;
	delete [] dmaxg;
}


void binNode::getProjection(// input
		   double * points, int numof_points, int dim,
		   // output
		   double * proj,
		   MPI_Comm comm)
{
	int rank;
	MPI_Comm_rank(comm, &rank);

        int worldrank;
        MPI_Comm_rank( MPI_COMM_WORLD, &worldrank );
/*
        MPI_Barrier(MPI_COMM_WORLD);
        if(!worldrank) cout << worldrank << ": Beginning getProjection" << endl;
*/

	int global_numof_points;
	double *p1 = new double [dim];
	double *p2 = new double [dim];

	MPI_CALL(MPI_Allreduce(&numof_points, &global_numof_points, 1, MPI_INT, MPI_SUM, comm));
	double *global_mean = centroids(points, numof_points,
									global_numof_points, dim, comm);
	MPI_Barrier(comm);
	furthestPoint(points, numof_points, dim, global_mean, p1, comm);
	furthestPoint(points, numof_points, dim, p1, p2, comm);

	for(int i = 0; i < dim; i++)
		proj[i] = p2[i] - p1[i];
	double norm = 0.0;
	for(int i = 0; i < dim; i++)
		norm += proj[i] * proj[i];
	norm = sqrt(norm);
	for(int i = 0; i < dim; i++)
		proj[i] /= norm;

    /*
    cout<<"\tenter get projection"<<endl;
    cout<<"p1: ";
    for(int i = 0; i < dim; i++)
        cout<<p1[i]<<" ";
    cout<<endl;
    cout<<"p2: ";
    for(int i = 0; i < dim; i++)
        cout<<p2[i]<<" ";
    cout<<endl;
    cout<<"proj: ";
    for(int i = 0; i < dim; i++)
        cout<<proj[i]<<" ";
    cout<<endl;
    */

	delete [] p1;
	delete [] p2;
	delete [] global_mean;
/*
        MPI_Barrier(MPI_COMM_WORLD);
        if(!worldrank) cout << worldrank << ": Finished getProjection" << endl;
*/
}



// select the kth smallest element in arr
// for median, ks = glb_N / 2
double binNode::distributeSelect(vector<double> &arr, int ks, MPI_Comm comm)
{

	int worldrank;
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);

	vector<double> S_less;
	//vector<double> S_equal;
	vector<double> S_great;
	S_less.reserve(arr.size());
	S_great.reserve(arr.size());
	
	int N = arr.size();
	int glb_N;
	MPI_CALL(MPI_Allreduce(&N, &glb_N, 1, MPI_INT, MPI_SUM, comm));

	//cout<<worldrank<<" : N="<<N<<" glb_N="<<glb_N<<endl;


	assert(glb_N > 0);

	double *pmean = centroids(&(arr[0]), N, glb_N, 1, comm);
	double mean = *pmean;
	delete pmean;
	
	//cout<<worldrank<<" : mean="<<mean<<endl;
	
	for(int i = 0; i < arr.size(); i++) {
		if(arr[i] > mean) S_great.push_back(arr[i]);
		else S_less.push_back(arr[i]);
	}

	int N_less, N_great, glb_N_less, glb_N_great;
	N_less = S_less.size();
	N_great = S_great.size();
	MPI_CALL(MPI_Allreduce(&N_less, &glb_N_less, 1, MPI_INT, MPI_SUM, comm));
	MPI_CALL(MPI_Allreduce(&N_great, &glb_N_great, 1, MPI_INT, MPI_SUM, comm));
	
	//cout<<worldrank<<" : N_less="<<N_less<<" glb_N_less="<<glb_N_less<<endl;
	//cout<<worldrank<<" : N_great="<<N_great<<" glb_N_great="<<glb_N_great<<endl;
	
	//assert(glb_N > glb_N_great);

	if( glb_N_less == ks || glb_N == 1 || glb_N == glb_N_less || glb_N == glb_N_great ) {
		//cout<<worldrank<<" : median="<<mean<<endl;
		return mean;
	}
	else if(glb_N_less > ks) {
		return distributeSelect(S_less, ks, comm);
	}
	else {
		return distributeSelect(S_great, ks-glb_N_less, comm);
	}

}



/*

// max variance coordinates splitter
void binNode::maxVarSplitter(// input
					double *points, int numof_points, int dim,
					bool flag_r,     // rotate points or not
					// output
					double *proj,
					double &medV,
					int* point_to_hyperplane_membership,
					int* local_numof_points_per_hyperplane,
					int* global_numof_points_per_hyperplane,
					MPI_Comm comm )
{
	double start_t = omp_get_wtime();
	int debug_verbose = 0;

	int worldrank, worldsize;
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
	MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	vector<double> newPoints(numof_points*dim);
	vector<double> MatRotation(dim*dim);
	
	if(flag_r) {	// rotate points
		start_t = omp_get_wtime();
		// 1. create a random rotation matrix using gram schimdt orthogonalization 
			vector<double> MatRand;
			MatRand.resize(dim*dim);
			for(int i = 0; i < dim*dim; i++)
				MatRand[i] = (double)rand() / (double)RAND_MAX;
			GramSchmidt( &(MatRand[0]), &(MatRotation[0]), dim, dim );
	
		MPI_Bcast(&(MatRotation[0]), dim*dim, MPI_DOUBLE, 0, comm);

		if(debug_verbose == 1 && worldrank == 0) {
			cout<<"    - gram schimdt done! - "<<omp_get_wtime() - start_t<<endl;
		}

		start_t = omp_get_wtime();
		// 2. rotate points

		//#pragma omp parallel for
		//for(int i = 0; i < numof_points; i++) {
		//	for(int j = 0; j < dim; j++) {
		//		newPoints[i*dim+j] = prod( &(MatRotation[j*dim]), &(points[i*dim]), dim );
		//	}
		//}


		int blocksize;
                if( numof_points > KNN_MAX_BLOCK_SIZE ) {
                        blocksize = std::min(KNN_MAX_BLOCK_SIZE, numof_points);
                } else {
                        blocksize = numof_points;
                }

                assert(blocksize > 0);
                int nblocks = (int) numof_points / blocksize;
                int iters = (int) ceil((double)numof_points/(double)blocksize);

                double alpha = 1.0;
                double beta = 0.0;
                for(int i = 0; i < iters; i++) {
                        double *currpts = points + i*blocksize*dim;
                        double *currnewpts = &(newPoints[i*blocksize*dim]);
                        if( (i == iters-1) && (numof_points % blocksize) ) {
                                blocksize = numof_points%blocksize;
                        }
                        bool omptest = blocksize > 4 * omp_get_max_threads();
                         #pragma omp parallel if( omptest )
                         {
                                int omp_num_points, last_omp_num_points;
                                int t = omp_get_thread_num();
                                int numt = omp_get_num_threads();
                                omp_num_points = blocksize / numt;
                                last_omp_num_points = blocksize - (omp_num_points * (numt-1));

                                //This thread's number of points
                                int npoints = (t == numt-1) ? last_omp_num_points : omp_num_points;
                                dgemm( "T", "N", &dim, &npoints, &dim, &alpha, &(MatRotation[0]), &dim,
                                        currpts  + (dim*t*omp_num_points),
                                        &dim, &beta, currnewpts + (dim*t*omp_num_points), &dim );
                        }

                }



		if(debug_verbose == 1 && worldrank == 0)  {
			cout<<"    - rotation done! - "<<omp_get_wtime() - start_t<<endl;
		}

	}  // end if (flag_r)
	else {
		#pragma omp parallel for
		for(int i = 0; i < numof_points*dim; i++)
			newPoints[i] = points[i];
	}


	start_t = omp_get_wtime();
	// 3. find the coord has the maximum variance
	vector<double> local_var, glb_var;
	local_var.resize(dim);
	glb_var.resize(dim);
	int glb_numof_points = 0;
	MPI_CALL(MPI_Allreduce(&numof_points, &glb_numof_points, 1, MPI_INT, MPI_SUM, comm));
	double *glb_mean = centroids( &(newPoints[0]), numof_points, glb_numof_points, dim, comm);

	parvar(&(newPoints[0]), numof_points, dim, glb_mean, &(local_var[0]));
	
	MPI_CALL(MPI_Allreduce(&(local_var[0]), &(glb_var[0]), dim, MPI_DOUBLE, MPI_SUM, comm));
	vector<double>::iterator it = max_element(glb_var.begin(), glb_var.end());
	int mvidx = it - glb_var.begin();
	if(flag_r) {
		for(int i = 0; i < dim; i++)
			proj[i] = MatRotation[mvidx*dim+i];
	}
	else {
		for(int i = 0; i < dim; i++)
			proj[i] = 0.0;
		proj[mvidx] = 1.0;
	}
	
	if(debug_verbose == 1 && worldrank == 0) {
		cout<<"    - find max variance coord done! - "<<omp_get_wtime() - start_t<<endl;
	}

	start_t = omp_get_wtime();
	// 4. find median
	vector<double> coords(numof_points);
	#pragma omp parallel for
	for(int i = 0; i < numof_points; i++)
		coords[i] = newPoints[i*dim+mvidx];
	medV = distributeSelect(coords, glb_numof_points/2, comm);
	
	if(debug_verbose == 1 && worldrank == 0) {
		cout<<"    - find median done! - "<<omp_get_wtime() - start_t<<endl;
	}

	start_t = omp_get_wtime();
	// 5. assign membership of each point
	local_numof_points_per_hyperplane[0] = 0;
	local_numof_points_per_hyperplane[1] = 0;
        int pointsperhp0 = 0, pointsperhp1 = 0;
	#pragma omp parallel for reduction(+:pointsperhp0,pointsperhp1)
	for(int i = 0; i < numof_points; i++) {
		if(newPoints[i*dim+mvidx] < medV) {
			point_to_hyperplane_membership[i] = 0;
			pointsperhp0++;
		}
		else {
			point_to_hyperplane_membership[i] = 1;
			pointsperhp0++;
		}
		
	}
	local_numof_points_per_hyperplane[0] = pointsperhp0;
	local_numof_points_per_hyperplane[1] = pointsperhp1;
	MPI_CALL(MPI_Allreduce(local_numof_points_per_hyperplane, global_numof_points_per_hyperplane, 2, MPI_INT, MPI_SUM, comm));
	
	if(debug_verbose == 1 && worldrank == 0) {
		cout<<"    - assign membership done! - "<<omp_get_wtime() - start_t<<endl;
	}


	delete [] glb_mean;
}
*/

/*


void binNode::generateRotation(int dim, double *R, MPI_Comm comm)
{
	vector<double> MatRand(dim*dim);
	for(int i = 0; i < dim*dim; i++)
		MatRand[i] = (double)rand() / (double)RAND_MAX;
	
	GramSchmidt( &(MatRand[0]), R, dim, dim );

	MPI_Bcast(R, dim*dim, MPI_DOUBLE, 0, comm);

}


void binNode::inverseRotatePoints(double *points, int numof_points, int dim, 
				  double *MatRotation,
				  // output
				  double *newPoints)
{
	int blocksize;
	if( numof_points > KNN_MAX_BLOCK_SIZE ) {
		blocksize = std::min(KNN_MAX_BLOCK_SIZE, numof_points);
    } else {
        blocksize = numof_points;
	}

	assert(blocksize > 0);
    int nblocks = (int) numof_points / blocksize;
	int iters = (int) ceil((double)numof_points/(double)blocksize);

    double alpha = 1.0;
	double beta = 0.0;
	for(int i = 0; i < iters; i++) {
		double *currpts = points + i*blocksize*dim;
		double *currnewpts = newPoints + i*blocksize*dim;
        if( (i == iters-1) && (numof_points % blocksize) ) {
            blocksize = numof_points%blocksize;
        }
        bool omptest = blocksize > 4 * omp_get_max_threads();
        #pragma omp parallel if( omptest )
        {
            int omp_num_points, last_omp_num_points;
            int t = omp_get_thread_num();
            int numt = omp_get_num_threads();
            omp_num_points = blocksize / numt;
            last_omp_num_points = blocksize - (omp_num_points * (numt-1));

            //This thread's number of points
            int npoints = (t == numt-1) ? last_omp_num_points : omp_num_points;
            dgemm( "N", "N", &dim, &npoints, &dim, &alpha, MatRotation, &dim,
                    currpts  + (dim*t*omp_num_points),
                    &dim, &beta, currnewpts + (dim*t*omp_num_points), &dim );
        }

    }	// end for (i < iters)

}


void binNode::rotatePoints(double *points, int numof_points, int dim, 
				  double *MatRotation,
				  // output
				  double *newPoints)
{
	int blocksize;
	if( numof_points > KNN_MAX_BLOCK_SIZE ) {
		blocksize = std::min(KNN_MAX_BLOCK_SIZE, numof_points);
    } else {
        blocksize = numof_points;
	}

	assert(blocksize > 0);
    int nblocks = (int) numof_points / blocksize;
	int iters = (int) ceil((double)numof_points/(double)blocksize);

    double alpha = 1.0;
	double beta = 0.0;
	for(int i = 0; i < iters; i++) {
		double *currpts = points + i*blocksize*dim;
		double *currnewpts = newPoints + i*blocksize*dim;
        if( (i == iters-1) && (numof_points % blocksize) ) {
            blocksize = numof_points%blocksize;
        }
        bool omptest = blocksize > 4 * omp_get_max_threads();
        #pragma omp parallel if( omptest )
        {
            int omp_num_points, last_omp_num_points;
            int t = omp_get_thread_num();
            int numt = omp_get_num_threads();
            omp_num_points = blocksize / numt;
            last_omp_num_points = blocksize - (omp_num_points * (numt-1));

            //This thread's number of points
            int npoints = (t == numt-1) ? last_omp_num_points : omp_num_points;
            dgemm( "T", "N", &dim, &npoints, &dim, &alpha, MatRotation, &dim,
                    currpts  + (dim*t*omp_num_points),
                    &dim, &beta, currnewpts + (dim*t*omp_num_points), &dim );
        }

    }	// end for (i < iters)

}
*/





