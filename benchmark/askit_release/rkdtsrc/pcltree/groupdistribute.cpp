#include "mpitree.h"
#include <algorithm>

void work_partition( vector<int> work_per_group, int size, vector<int> &group_size)
{
	int total_work;
	int numof_groups = work_per_group.size();
	assert( numof_groups == 2 );
	
	total_work=work_per_group[0] + work_per_group[1];

	group_size.resize(2);
	group_size[0] = ((double)work_per_group[0]/(double)total_work) * (double)size;
	group_size[1] = ((double)work_per_group[1]/(double)total_work) * (double)size;

	//Because of rounding error, total process assignment might not equal size, so adjust as necessary.
	while( group_size[0] + group_size[1] < size ) {
		if( group_size[0] < group_size[1] ) group_size[0]++;
		else group_size[1]++;
	}
	while( group_size[0] + group_size[1] > size ) {  //Probably won't happen, but just to be safe.
		if( group_size[0] > group_size[1] ) group_size[0]--;
		else group_size[1]--;
	}

	//if(group_size[0]%2 != 0 && group_size[0]+group_size[1] > 2) {
	//	if(group_size[0] < group_size[1]) { group_size[1]--; group_size[0]++; }
	//	else { group_size[1]++; group_size[0]--; }
	//}

	assert(group_size[0]);
	assert(group_size[1]);
	assert(group_size[0] + group_size[1] == size);
}



/*
 Algorithm
	 for each group
	    find how many points I sent to it
			do a prefix sum on this points
			last processor broadcasts total count	
			find points per processors in the new group using my count and the prefix sum.

example:
4 ranks in comm, two groups, just show numbers for one group, say group 1 (ranks 2 and 3)

group     :  -- 0  --     --  1 --
rank      :  0      1      2     3
points    :  1     11      4     4 
scan      :  1     12     16    20  -- size per processor: 10  (wo groups

rank 0: sends 1 points to rank 2  ( proc 0 of the second group )
rank 1: sends 9 points to rank 2
        sends 3 points to rank 3

rank 2,3: send all of their points to rank 3.

*/	 
int groupDistribute(	// input
			int *point_group, //store the group id for each point
			int numof_points,    // number of points
			int numof_groups,   // number of groups 
			MPI_Comm comm,   // mpi communicator
			// output
			int &my_group_id,  // the group that I belong to
			int *proc_id // the processor id for each data point
			){
	int size;  MPI_Comm_size(comm, &size);
	int rank;  MPI_Comm_rank(comm, &rank);
	int ierr;

	// compute group information
	vector<int> loc_numof_points_per_group;    // local number of points per group
	loc_numof_points_per_group.resize(numof_groups); 
	for( int i=0; i< numof_groups; i++) loc_numof_points_per_group[i]=0;
	for( int i=0; i< numof_points; i++) loc_numof_points_per_group[   point_group[i] ] ++;

	// prefix sum and global counts for each group
	vector<int> glb_numof_points_per_group;  // global number of points per group
	glb_numof_points_per_group.resize(numof_groups);
	vector<int> scan_loc_numof_points_per_group;   
	scan_loc_numof_points_per_group.resize(numof_groups);
	iM( MPI_Scan( (void *) &loc_numof_points_per_group[0],  
				(void *) &scan_loc_numof_points_per_group[0], numof_groups, 
				MPI_INT, MPI_SUM, comm));
	int last_rank = size-1;
	for(int i = 0; i < numof_groups; i++)
		glb_numof_points_per_group[i] = scan_loc_numof_points_per_group[i];
	iM( MPI_Bcast( (void *) &glb_numof_points_per_group[0], numof_groups, 
								 MPI_INT, last_rank, comm)); 	// rank 
	// convert to exclusive scan
	for( int i=0; i<numof_groups; i++) scan_loc_numof_points_per_group[i] -= loc_numof_points_per_group[i];

	// decide how many processes you want to use per rank
	vector<int> group_size(numof_groups);
	work_partition( glb_numof_points_per_group, size, group_size);


	vector<int> points_per_group_rank(numof_groups);
	for(int i=0; i<numof_groups; i++) {
		points_per_group_rank[i] = (group_size[i]==0) ? 0 : glb_numof_points_per_group[i]/group_size[i];
	}

	// compute proc_id for each point.
	for (int i=0; i<numof_points; i++) proc_id[i] = 0;
	int proc_shift=0;
	int local_cumulative_size = 0;
	for (int j=0; j<numof_groups; j++){
		proc_shift += (j==0)?0: group_size[j-1];
		int cnt = 0;
		if(group_size[j] == 0) continue;
		local_cumulative_size += group_size[j];
		for(int i=0; i<numof_points; i++){	 
			proc_id[i] = ( point_group[i] != j ) ?
				proc_id[i] :
				proc_shift + ( scan_loc_numof_points_per_group[j]+ cnt++)/points_per_group_rank[j];  // assign ID

			proc_id[i] = min(local_cumulative_size-1, proc_id[i]);

			// to deal with the case in which we have more points than points_per_group_rank
			cnt = cnt % points_per_group_rank[j];	
		}
	}

	// compute my_group_id
	int inclusive_scan_group_size=group_size[0];
	for( int j=0; j<numof_groups; j++){
		if( rank < inclusive_scan_group_size ) { my_group_id = j;  break;  }
		inclusive_scan_group_size += group_size[j+1];
	}

	return 0;   // return value reserved for error code
}

		
		
		
// used after local_rearrange, item_group has already been sorted
// new function, do not output proc_id
void group2rankDistribute(//input
			int numof_terms, 
			int * rank_colors, int numof_ranks,
			int * point_to_kid_membership, 
			//output
			int *send_count)
{
	int numof_kids = 2;

	//find the first position
	int * first_pos_per_group = new int [numof_kids];
	for(int i = 0; i < numof_kids; i++) {
		int * tmp_pos = find(rank_colors, rank_colors+numof_ranks, i);
		first_pos_per_group[i] = tmp_pos-rank_colors;
		
	}

	for(int i = 0; i < numof_terms; i++) {
		send_count[first_pos_per_group[point_to_kid_membership[i]]]++;
	}
	int istart = 0, iend = 0, iproc = 0;
	int divd, rem;
	for(int i = 0; i < numof_kids; i++) {
		int tmp_count = (int) count(rank_colors, rank_colors+numof_ranks, i);
		divd = send_count[first_pos_per_group[i]] / tmp_count;
		rem = send_count[first_pos_per_group[i]] % tmp_count;
		for(int j = 0; j < tmp_count; j++) {
			send_count[iproc] = (j < rem) ? (divd+1) : divd;
			iproc++;
		}
	} 
	
	delete[] first_pos_per_group;	
	return;
}	
		
