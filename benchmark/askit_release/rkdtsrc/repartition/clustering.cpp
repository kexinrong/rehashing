#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <stTree.h>
#include "ompUtils.h"
#include "direct_knn.h"
#include "clustering.h"
//#include "random123wrapper.h"

using namespace std;


__inline static
pair<double, int> find_nearest_cluster(double *dist, int n)
{
	pair<double, int> tmp_min; 
	tmp_min.first = dist[0];
	tmp_min.second = 0;
	for(int i = 1; i < n; i++) {
		if(dist[i] < tmp_min.first) {
			tmp_min.first = dist[i];
			tmp_min.second = i;
		}
	}
	return tmp_min;
}


// find the smallest value in the range [first, last)
__inline static
double find_min(double *arr, int first, int last)
{
	double minv = 0.0;
	minv = *min_element(arr+first, arr+last);
	return minv;
}


bool less_second(const pair<int, double>& m1, const pair<int, double>& m2)
{
	return m1.second > m2.second;
}


// give m different random int in a range [0, N)
void randperm(int m, int N, vector<int>& arr)
{

	if(m > N) { 
		cerr<<" m must <= N"<<endl;
		return;
	}

	arr.resize(m);
	for(int i = 0; i <arr.size(); i++) {
		double tmp = floor( (double)N*(double)rand()/(double)RAND_MAX );
		arr[i] = (int)tmp;
	}
	sort(arr.begin(), arr.end());
	vector<int>::iterator it = unique(arr.begin(), arr.end());
	arr.resize(it - arr.begin());

	int pp = m;
	while(arr.size() < m) {
		pp++;
		double tmp = floor( (double)N*(double)rand()/(double)RAND_MAX );
		arr.push_back((int)tmp);
		sort(arr.begin(), arr.end());
		vector<int>::iterator it = unique(arr.begin(), arr.end());
		arr.resize(it - arr.begin());
	}
}



double sqdiff(double *src1, double *src2, int n)
{
	double dist = 0.0;
	for(int i = 0; i < n; i++) {
		dist += (src1[i] - src2[i]) * (src1[i] - src2[i]);
	}
	return dist;
}



// for the case that one input matrix has much smaller number of points 
// than the other, it should be faster and stable than using BLAS (e. g. n >> m)
void SqEuDist(double *ref, double *query, long n, long m, int dim, double *dist)
{
	#pragma omp parallel for
	for(int j = 0; j < n; j++) {
		for(int i = 0; i < m; i++) {
			dist[i*n+j] = sqdiff(query+i*dim, ref+j*dim, dim);
		}
	}
}


int k_clusters_balanced(double    *points,
           int        dim,  
           long        numPoints, 
           int        numClusters, 	
	       int 	  maxIter,
	       int 	 seedType,
	       int 	nfold,
           int       *membership, 
           double    *centers,   
	       int 	 *clusterSize,
	       int       *p_clusterSize,
               MPI_Comm   comm) 
{
	int current_num_seeds = 0;
	
	vector<int> tmp_membership; tmp_membership.resize(numPoints);
	vector<double> tmp_centers; tmp_centers.resize(numClusters*dim);
	vector<int> tmp_global_clusterSize; tmp_global_clusterSize.resize(numClusters);
	vector<int> tmp_local_clusterSize;  tmp_local_clusterSize.resize(numClusters);

	double ratio_max = 0.0, ratio_cur = 0.0;
	for(int i = 0; i < nfold; i++) {
		if(0 == seedType) {	// random seeds
			randomSeeds(points, numPoints, dim,
				    numClusters, &(tmp_centers[0]), comm);
		}
		else if (1 == seedType) {
			//addSeeds(points, numPoints, dim, centers, 
			//	   numClusters, &current_num_seeds, comm);
			ostrovskySeeds(points, numPoints, dim, numClusters, &(tmp_centers[0]), comm);
		}

		mpi_kmeans(points, dim, numPoints, numClusters, maxIter,
		   	   &(tmp_membership[0]), &(tmp_centers[0]), 
			   &(tmp_global_clusterSize[0]), &(tmp_local_clusterSize[0]), comm);
		
		ratio_cur = ( (double)*min_element(tmp_global_clusterSize.begin(), tmp_global_clusterSize.end()) ) 
		      /( (double)*max_element(tmp_global_clusterSize.begin(), tmp_global_clusterSize.end()) );
		
		//cout<<"ratio_cur: "<<ratio_cur<<endl;
		if(ratio_cur > ratio_max) {
			ratio_max = ratio_cur;
			copy(tmp_membership.begin(), tmp_membership.end(), membership);
			copy(tmp_centers.begin(), tmp_centers.end(), centers);
			copy(tmp_global_clusterSize.begin(), tmp_global_clusterSize.end(), clusterSize);
			copy(tmp_local_clusterSize.begin(), tmp_local_clusterSize.end(), p_clusterSize);
		}
		//cout<<"ratio_max: "<<ratio_max<<endl;
	}

	return 1;
}


int k_clusters(double    *points,
               int        dim,  
               long        numPoints, 
               int        numClusters, 	
	       int 	  maxIter,
	       int 	 seedType,
               int       *membership, 
               double    *centers,   
	       int 	 *clusterSize,
	       int       *p_clusterSize,
               MPI_Comm   comm) 
{
	int current_num_seeds = 0;
	if(0 == seedType) {	// random seeds
		randomSeeds(points, numPoints, dim,
				numClusters, centers, comm);
	}
	else if (1 == seedType) {
		//addSeeds(points, numPoints, dim, centers, 
		//		numClusters, &current_num_seeds, comm);
		ostrovskySeeds(points, numPoints, dim, numClusters, centers, comm);
	}

	mpi_kmeans(points, dim, numPoints, numClusters, maxIter, 
		   membership, centers, clusterSize, p_clusterSize, comm);

	return 1;
}


pair<double, int> mpi_kmeans(double *points,
               		     int    dim,   
               		     long   numPoints,  
			     int    numClusters,
	       		     int    maxIter,
               		     int    *membership, 
               		     double *clusters,  
	       		     int    *clusterSize,
	      	 	     int    *p_clusterSize,
               		     MPI_Comm   comm)
{
    	int loop = 0, idx = 1;
    	double *p_centers = new double [numClusters*dim];	// local center
	double *temp_dist = new double [numClusters*numPoints]; 

	int rank, nproc;
	MPI_Comm_size(comm, &nproc);
	MPI_Comm_rank(comm, &rank);

    	double obj_prev = (double)RAND_MAX;
	obj_prev = fabs(obj_prev);
	double obj_curr = 0.0;
	double ratio = 1.0;
	double local_ssd = 0.0, glb_ssd = 0.0;

	long glb_numof_points = 0;
	MPI_Allreduce(&numPoints, &glb_numof_points, 1, MPI_LONG_LONG, MPI_SUM, comm);

	while(loop++ < maxIter && ratio > 0.0001) {
		local_ssd = 0.0;
		memset(p_clusterSize, 0, numClusters*sizeof(int));
		memset(p_centers, 0, numClusters*dim*sizeof(double));
		SqEuDist(clusters, points, numClusters, numPoints, dim, temp_dist);
		#pragma omp parallel for
		for(int i = 0; i < numPoints; i++) {
			pair<double, int> NearestNeighbor 
				= find_nearest_cluster(temp_dist+i*numClusters, numClusters);
			int index = NearestNeighbor.second;
			membership[i] = index;

			#pragma omp atomic
			p_clusterSize[index]++;
			#pragma omp atomic
			local_ssd += NearestNeighbor.first;

			for(int j = 0; j < dim; j++) {
				#pragma omp atomic
				p_centers[index*dim+j] += points[i*dim+j];
			}
		}
	
		MPI_Allreduce(p_centers, clusters, 
				numClusters*dim, MPI_DOUBLE, MPI_SUM, comm);
        	MPI_Allreduce(p_clusterSize, clusterSize, 
				numClusters, MPI_INT, MPI_SUM, comm);
		MPI_Allreduce(&local_ssd, &glb_ssd, 1, MPI_DOUBLE, MPI_SUM, comm);
		obj_curr = glb_ssd / (double)glb_numof_points;
		ratio = ( obj_prev - obj_curr ) / obj_curr;
		obj_prev = obj_curr;

		#pragma omp parallel for
        	for (int i=0; i<numClusters; i++) {
                	if (clusterSize[i] > 1) {
				for(int j = 0; j < dim; j++) 
					clusters[i*dim+j] = clusters[i*dim+j]/(double)clusterSize[i];
			}
        	}
        
    	}


       	memset(p_clusterSize, 0, numClusters*sizeof(int));
 	SqEuDist(clusters, points, numClusters, numPoints, dim, temp_dist);
	
	local_ssd = 0.0;
	#pragma omp parallel for
	for(int i = 0; i < numPoints; i++) {
		pair<double, int> NearestNeighbor 
			= find_nearest_cluster(temp_dist+i*numClusters, numClusters);
		membership[i] = NearestNeighbor.second;
		
		#pragma omp atomic
		p_clusterSize[NearestNeighbor.second]++;
		#pragma omp atomic
		local_ssd += NearestNeighbor.first;
	}
	MPI_Allreduce(p_clusterSize, clusterSize, numClusters, MPI_INT, MPI_SUM, comm);
	MPI_Allreduce(&local_ssd, &glb_ssd, 1, MPI_DOUBLE, MPI_SUM, comm);

    	obj_curr =  glb_ssd / (double)glb_numof_points;

	delete [] p_centers;
	delete [] temp_dist;

    	return make_pair<double, int>(obj_curr, loop);
}


double VarianceRatio(double *points, int numof_points, int dim,
		     double *cls_centroids, int numof_centers, int * glb_cluster_size,
		     MPI_Comm comm)
{
	double BV = 0.0, WV = 0.0, local_v = 0.0, glb_v = 0.0;
	int glb_numof_points = 0; 
	MPI_Allreduce(&numof_points, &glb_numof_points, 1, MPI_INT, MPI_SUM, comm);
	
	double *glb_center = centroids(points, numof_points, glb_numof_points, dim, comm);
	
	for (int i = 0; i < numof_centers; i++) {
		double pi = (double)glb_cluster_size[i] / (double)glb_numof_points;
		BV += pi * sqdiff(cls_centroids+i*dim, glb_center, dim);
	}
	double *dist = new double [numof_points];
	SqEuDist(glb_center, points, 1, numof_points, dim, dist);
	for(int i = 0; i < numof_points; i++)
		local_v += dist[i];
	MPI_Allreduce(&local_v, &glb_v, 1, MPI_DOUBLE, MPI_SUM, comm);
	WV = glb_v / (double)glb_numof_points;
	WV = WV - BV;

	delete [] dist;
	delete [] glb_center;
	
	return BV/ WV;

}



void randomSeeds(double *points, 
		  int numof_points,
		  int dim,
		  int numof_clusters,
		  // output
		  double *seeds,
		  MPI_Comm comm)
{
	int rank, nproc;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nproc);

	int glb_numof_points;
	MPI_Allreduce(&numof_points, &glb_numof_points, 1, MPI_INT, MPI_SUM, comm);
	vector<int> pos_seeds;
	pos_seeds.resize(numof_clusters);
	if(rank == 0) {
		randperm(numof_clusters, glb_numof_points, pos_seeds);
	}
	MPI_Bcast(&(pos_seeds[0]), numof_clusters, MPI_INT, 0, comm);

	int right_b = 0;
	MPI_Scan(&numof_points, &right_b, 1, MPI_INT, MPI_SUM, comm);
	int left_b = right_b - numof_points;
	right_b -= 1;

	vector<double> chosen_seed;
	for(int i = 0; i < pos_seeds.size(); i++) {
		if(pos_seeds[i] <= right_b && pos_seeds[i] >= left_b) {
			int index = pos_seeds[i] - left_b;
			for(int j = 0; j < dim; j++) {
				chosen_seed.push_back(points[index*dim+j]);
			}
		}
	}
	
	vector<int> recv_count;
	vector<int> send_count;
	recv_count.resize(nproc);
	send_count.resize(nproc);
	int sendbuf = chosen_seed.size();
	MPI_Allgather(&sendbuf, 1, MPI_INT, &(recv_count[0]), 1, MPI_INT, comm);
	for(int i = 0; i < send_count.size(); i++) {
		send_count[i] = sendbuf;
	}
	vector<int> send_disp;
	vector<int> recv_disp;
	send_disp.resize(nproc); 
	recv_disp.resize(nproc);
	send_disp[0] = 0;
	recv_disp[0] = 0;
	chosen_seed.resize(sendbuf*nproc);
	for(int i = 1; i < nproc; i++) {
		send_disp[i] = send_disp[i-1] + send_count[i-1];
		recv_disp[i] = recv_disp[i-1] + recv_count[i-1];
		if(chosen_seed.size() > 0) {
			for(int j = 0; j < sendbuf; j++)
				chosen_seed[i*sendbuf+j] = chosen_seed[j];
		}
	}

	MPI_Alltoallv( &(chosen_seed[0]), &(send_count[0]), &(send_disp[0]), MPI_DOUBLE,
			seeds, &(recv_count[0]), &(recv_disp[0]), MPI_DOUBLE, comm);

}


double *centroids(double *X, 
		int loc_n, 		// local no. of points
		int glb_n,		// global no. of points
		int dim, 
		MPI_Comm comm)
{
	double *localCenter = new double [dim];

	int stride = dim + 128 / sizeof(double); //Make sure two threads don't write to same cache line.
        int maxt = omp_get_max_threads();

        vector<double> threadmu( stride * maxt );
        for(int i = 0; i < dim; i++) localCenter[i] = 0.0;
        #pragma omp parallel if(loc_n > 2000)
        {
                int t = omp_get_thread_num();
                double *localmu = &(threadmu[t*stride]);
                for(int i = 0; i < dim; i++) localmu[i] = 0.0;
                int npdim = loc_n * dim;

                register int idim;
                register int j;
                #pragma omp for schedule(dynamic,50)
                for(int i = 0; i < loc_n; i++) {
                        idim = i*dim;
                        #pragma vector
                        for(j = 0; j < dim; j++)
                                localmu[j] += X[idim+j];
                }

        }

        for(int t = 0; t < maxt; t++) {
                double *localmu = &(threadmu[t*stride]);
                for(int i = 0; i < dim; i++)
                        localCenter[i] += localmu[i];
        }



	double *globalCenter = new double [dim];
        MPI_Allreduce(localCenter, globalCenter, dim, MPI_DOUBLE, MPI_SUM, comm);
	for(int i = 0; i < dim; i++)
        	globalCenter[i] = globalCenter[i] / glb_n;

	delete [] localCenter;

	return globalCenter;
}


double *initial_two_seeds(double *X,
			   int n,
			   int dim,
			   MPI_Comm comm)
{
	double tmp = 1.0;
	int N = 0;

	MPI_Allreduce(&n, &N, 1, MPI_INT, MPI_SUM, comm);
	double *center = centroids(X, n, N, dim, comm);

	double *dx = new double [n];
	double local_d1 = 0.0, D1 = 0.0;
	
	// ------------ sample the 1st seed ----------
	SqEuDist(center, X, 1, n, dim, dx);
	
	#pragma omp parallel for reduction(+:local_d1)
	for(int i = 0; i < n; i++)
		local_d1 += dx[i];
	MPI_Allreduce(&local_d1, &D1, 1, MPI_DOUBLE, MPI_SUM, comm);
	double *prob = new double [n];
	tmp = 2*N*D1;
	#pragma omp parallel for
	for(int i = 0; i < n; i++)
		prob[i] = (D1 + N*dx[i]) / tmp;
	double *c1; // [dim]
	c1 = sample(X, prob, n, dim, 1, 1, comm);
	// ------------ sample the 1st seed Done! --------------

	// ------------ sample the 2nd seed ----------
	SqEuDist(c1, X, 1, n, dim, dx);
	local_d1 = 0.0;
	#pragma omp parallel for reduction(+:local_d1)
	for(int i = 0; i < n; i++)
		local_d1 += dx[i];
	MPI_Allreduce(&local_d1, &D1, 1, MPI_DOUBLE, MPI_SUM, comm);
	#pragma omp parallel for
	for(int i = 0; i < n; i++)
		prob[i] = dx[i] / D1;
	double *c2; // [dim]
	c2 = sample(X, prob, n, dim, 1, 2, comm);

	// ------------ sample the 2nd seed DONE ----------

	double *seeds = new double [2*dim];
	memcpy(seeds, c1, dim*sizeof(double));
	memcpy(seeds+dim, c2, dim*sizeof(double));
	
	delete [] c1;
	delete [] c2;
	delete [] center;
	delete [] dx;
	delete [] prob;

	return seeds;

}


double *sample(double *X, double *pw, int n, int dim, int k, int idx_base, MPI_Comm comm)
{
	int nproc, rank;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nproc);

	double *prefix_sum_pw = new double [n+1];
	omp_par::scan(pw, prefix_sum_pw, n+1);
	
	double tmp_pw = prefix_sum_pw[n];
	double cumm_pw = 0.0;
	MPI_Scan(&tmp_pw, &cumm_pw, 1, MPI_DOUBLE, MPI_SUM, comm);

	#pragma omp parallel for
	for(int i = 1; i < n+1; i++) 
		prefix_sum_pw[i] += cumm_pw - tmp_pw;

	double *sampled_points = new double [k*dim];
	double *sp = new double [dim];
	int *idx_per_proc = new int [nproc];
	
	for(int i = 0; i < k; i++) {
		double rs = 0.001;
		//if(rank==0) { 	// if root proc, generate a rand number between [0, 1]
			//std::srand( (unsigned)time(NULL)*dim*idx_base );
			rs = (double)rand() / (double)RAND_MAX;
		//}
		MPI_Bcast(&rs, 1, MPI_DOUBLE, 0, comm);
		
		int idx = -1;
		for(int j = 1; j < n+1; j++) {
			if(rs < prefix_sum_pw[j]) {
				idx = j - 1;
				break;
			}
		}
		int hit = 0;
		MPI_Allgather(&idx, 1, MPI_INT, idx_per_proc, 1, MPI_INT, comm);
		for(int pp = 0; pp < nproc; pp++ ) {
			if(idx_per_proc[pp] != -1) {
				hit = pp;
				break;
			}
		}

		if(rank == hit) 
			memcpy(sp, X+idx*dim, dim*sizeof(double));
		MPI_Bcast(sp, dim, MPI_DOUBLE, hit, comm);
		memcpy(sampled_points+i*dim, sp, dim*sizeof(double));
	} // for i = 1 : k, sample k points


	delete [] prefix_sum_pw;
	delete [] sp;
	delete [] idx_per_proc;
		
	return sampled_points;

}


void addSeeds(double *X, int numPoints, int dim,
		     double *seeds, int numSeeds, 
		     int *current_num_seeds, 
		     MPI_Comm comm)
{
	if (0 == (*current_num_seeds)) {
		double * _2seeds = new double [2*dim];
		_2seeds = initial_two_seeds(X, numPoints, dim, comm);
		memcpy(seeds, _2seeds, 2*dim*sizeof(double));
		*current_num_seeds = 2;
		delete [] _2seeds;
	}

	if ((*current_num_seeds) >= numSeeds) {
		return;
	}
	int tmp_num = (*current_num_seeds);

	double *Dx = new double [numPoints];
	double *tmp_dist = new double [numPoints*tmp_num];
	knn::compute_distances(seeds, X, tmp_num, numPoints, dim, tmp_dist);
	#pragma omp parallel for 
	for(int i = 0; i < numPoints; i++)
		Dx[i] = find_min(tmp_dist, i*tmp_num, (i+1)*tmp_num);
	delete [] tmp_dist;
	
	double local_sum = 0;
	#pragma omp parallel for reduction(+:local_sum)
	for(int i = 0; i < numPoints; i++)
		local_sum += Dx[i];
	double global_sum = 0;
	MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
	
	double *prob = new double [numPoints];
	#pragma omp parallel for 
	for(int i = 0; i < numPoints; i++) 
		prob[i] = Dx[i] / global_sum;

	double *sn = sample(X, prob, numPoints, dim, 1, tmp_num, comm);
	memcpy(seeds+(*current_num_seeds)*dim, sn, dim*sizeof(double));
	(*current_num_seeds) +=1;

	delete [] sn;
	delete [] prob;
	delete [] Dx;

	addSeeds(X, numPoints, dim, seeds, numSeeds, current_num_seeds, comm);

}

// returen subset = points \ {idx}th point
void exclusive_subset(double *points, int numof_points, int dim, 
		     int idx, double *subset)
{
	if(idx == 0) {
		memcpy(subset, points+dim, (numof_points-1)*dim*sizeof(double));
	}
	if(idx == numof_points - 1) {
		memcpy(subset, points, (numof_points-1)*dim*sizeof(double));
	}
	if(0 < idx < numof_points - 1) {
		memcpy(subset, points, idx*dim*sizeof(double));
		memcpy(subset+dim*idx, points+(idx+1)*dim, (numof_points-1-idx)*dim*sizeof(double));
	}
}


void eliminateSeeds(double *points, int numof_points, int dim,
		      vector<double>& oversampled_seeds,
		      int numof_seeds, double *seeds,
		      MPI_Comm comm)
{
	int numof_oversampled_seeds = oversampled_seeds.size() / dim;
	vector<int> membership(numof_points);
	vector<int> local_cluster_size(numof_oversampled_seeds);
	vector<int> glb_cluster_size(numof_oversampled_seeds);
	vector<double> subset(numof_oversampled_seeds*dim);
	// greedy elimination
	vector<double> tmp_seeds;
	for(int i = numof_oversampled_seeds; i > numof_seeds; i--) {
		tmp_seeds.resize((i-1)*dim);
		double loss_min = (double)RAND_MAX;
		mpi_kmeans(points, dim, numof_points, 
			   i, 1, &(membership[0]), &(oversampled_seeds[0]),
			   &(glb_cluster_size[0]), &(local_cluster_size[0]), comm);
		for(int j = 0; j < i; j++) {
			subset.resize((i-1)*dim);
			exclusive_subset( &(oversampled_seeds[0]), i, dim, j, &(subset[0]) );
			pair<double, int> kmeans_info = mpi_kmeans(points, dim, numof_points, 
						i-1, 0, &(membership[0]), &(subset[0]),
						&(glb_cluster_size[0]), &(local_cluster_size[0]), comm);
			if(loss_min > (double)glb_cluster_size[j]*kmeans_info.first) {
				loss_min = (double)glb_cluster_size[j]*kmeans_info.first;
				copy(subset.begin(), subset.end(), tmp_seeds.begin());
			}
		}
		oversampled_seeds.resize((i-1)*dim);
		copy(tmp_seeds.begin(), tmp_seeds.end(), oversampled_seeds.begin());
	}

	copy(oversampled_seeds.begin(), oversampled_seeds.end(), seeds);

}


void ostrovskySeeds(double *points, int numof_points, int dim,
		     int numof_seeds, double *seeds,
		     MPI_Comm comm)
{
	int numof_oversampled_seeds = 2 * numof_seeds;
	vector<double> oversampled_seeds(numof_oversampled_seeds * dim);

	int curr_numof_seeds = 0;
	addSeeds(points, numof_points, dim, 
		  &(oversampled_seeds[0]), numof_oversampled_seeds, 
		  &curr_numof_seeds, comm);
	eliminateSeeds(points, numof_points, dim, oversampled_seeds,
		       numof_seeds, seeds, comm);
}



void equal2clusters(double * points, 
					int numof_points, 
					int dim,
					// output
					int* point_to_hyperplane_membership,
					double* centroids,
					int* global_numof_points_per_hyperplane,
					int* local_numof_points_per_hyperplane, 
					MPI_Comm comm)
{
	int rank, nproc;
	MPI_Comm_rank(comm, &rank);

	double* projDirection = new double [dim];
	calProjDirection(points, numof_points, dim, projDirection, comm);

	vector<double> projValue(numof_points);
	#pragma omp parallel for
	for(int i = 0; i < numof_points; i++) {
		projValue[i] = 0.0;
		for(int j = 0; j < dim; j++) 
			projValue[i] += projDirection[j] * points[i*dim+j];
	}
	
	int glb_numof_points;
	MPI_Allreduce(&numof_points, &glb_numof_points, 1, MPI_INT, MPI_SUM, comm);
	
	double medianValue = distSelect(projValue, glb_numof_points/2, comm);

	local_numof_points_per_hyperplane[0] = 0;
	local_numof_points_per_hyperplane[1] = 0;

	double* local_centers = new double [dim*2];
	for(int i = 0; i < dim*2; i++) local_centers[i] = 0.0;

	for(int i = 0; i < numof_points; i++) {
		if(projValue[i] < medianValue) {
			point_to_hyperplane_membership[i] = 0;
			local_numof_points_per_hyperplane[0]++;
			for(int j = 0; j < dim; j++)
				local_centers[0*dim+j] += points[i*dim+j];	
		}
		else {
			point_to_hyperplane_membership[i] = 1;
			local_numof_points_per_hyperplane[1]++;
			for(int j = 0; j < dim; j++)
				local_centers[1*dim+j] += points[i*dim+j];
		}
	}
	MPI_Allreduce(local_numof_points_per_hyperplane, global_numof_points_per_hyperplane, 2, MPI_INT, MPI_SUM, comm);
	MPI_Allreduce(local_centers, centroids, 2*dim, MPI_DOUBLE, MPI_SUM, comm);
	for(int i = 0; i < 2; i++) {
		for(int j = 0; j < dim; j++)
			centroids[i*dim+j] /= (double)global_numof_points_per_hyperplane[i];
	}

	delete [] local_centers;
	delete [] projDirection;

}


void findFurthestPoint(// input
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
	MPI_Allgather(pdmax, 1, MPI_DOUBLE, dmaxg, 1, MPI_DOUBLE, comm);
	double *pm = max_element(dmaxg, dmaxg+nproc);
	
	int rankmax = pm - dmaxg;

	MPI_Bcast(furP, dim, MPI_DOUBLE, rankmax, comm);

	delete [] dist;
	delete [] dmaxg;
}


void calProjDirection(// input
		   double * points, int numof_points, int dim,
		   // output
		   double * proj,
		   MPI_Comm comm)
{
	int rank;
	MPI_Comm_rank(comm, &rank);

	int global_numof_points;
	double *p1 = new double [dim];
	double *p2 = new double [dim];
	
	MPI_Allreduce(&numof_points, &global_numof_points, 1, MPI_INT, MPI_SUM, comm);
	double *global_mean = centroids(points, numof_points, 
									global_numof_points, dim, comm);
	MPI_Barrier(comm);
	findFurthestPoint(points, numof_points, dim, global_mean, p1, comm);
	//MPI_Barrier(comm);
	findFurthestPoint(points, numof_points, dim, p1, p2, comm);

	for(int i = 0; i < dim; i++)
		proj[i] = p1[i] - p2[i];
	double norm = 0.0;
	for(int i = 0; i < dim; i++)
		norm += proj[i] * proj[i];
	norm = sqrt(norm);
	for(int i = 0; i < dim; i++)
		proj[i] /= norm;

	delete [] p1;
	delete [] p2;
	delete [] global_mean;
}



// select the kth smallest element in arr
// for median, ks = glb_N / 2
double distSelect(vector<double> &arr, int ks, MPI_Comm comm)
{
	vector<double> S_less;
	//vector<double> S_equal;
	vector<double> S_great;
	S_less.reserve(arr.size());
	S_great.reserve(arr.size());
	
	int N = arr.size();
	int glb_N;
	MPI_Allreduce(&N, &glb_N, 1, MPI_INT, MPI_SUM, comm);
	
	double *pmean = centroids(&(arr[0]), N, glb_N, 1, comm);
	double mean = *pmean;
	delete pmean;
	
	for(int i = 0; i < arr.size(); i++) {
		if(arr[i] > mean) S_great.push_back(arr[i]);
		else S_less.push_back(arr[i]);
	}

	int N_less, N_great, glb_N_less, glb_N_great;
	N_less = S_less.size();
	N_great = S_great.size();
	MPI_Allreduce(&N_less, &glb_N_less, 1, MPI_INT, MPI_SUM, comm);
	MPI_Allreduce(&N_great, &glb_N_great, 1, MPI_INT, MPI_SUM, comm);
	
	if( glb_N_less == ks || glb_N == 1) return mean;
	else if(glb_N_less > ks) {
		return distSelect(S_less, ks, comm);
	}
	else {
		return distSelect(S_great, ks-glb_N_less, comm);
	}


}








