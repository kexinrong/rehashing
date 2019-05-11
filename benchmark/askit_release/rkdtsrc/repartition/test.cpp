#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <assert.h>
#include <iostream>
#include <unistd.h>
#include <mpi.h>
#include "file_io.h"
#include "repartition.h"
#include "CmdLine.h"
#include "generator.h"
#include "clustering.h"
#include "direct_knn.h"

using namespace std;
using namespace Torch;
using namespace knn::repartition;

void printMat(vector<double> mat, int rows, int cols)
{
	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < cols; j++)
			cout<<mat[i*cols+j]<<" ";
		cout<<endl;
	}
}

void printVec(vector<int> arr, int n)
{
	for(int i = 0; i < n; i++)
		cout<<arr[i]<<" ";
	cout<<endl;
}


double check_correctness(int numof_points, int dim,
		       int * labels, int *membership, int numof_classes, 
		       MPI_Comm comm)
{
	vector<int> predicted_labels;
	predicted_labels.resize(numof_points);
	for(int i = 0; i < numof_classes; i++) {
		vector<int> local_count;
		local_count.resize(numof_classes);
		for(int j = 0; j < numof_points; j++) {
			if(membership[j] == i) local_count[labels[j]]++;
		}
		vector<int> glb_count;
		glb_count.resize(numof_classes);
		MPI_Allreduce(&(local_count[0]), &(glb_count[0]), numof_classes, 
				MPI_INT, MPI_SUM, comm);
		vector<int>::iterator it 
			= max_element(glb_count.begin(), glb_count.end());
		int true_label = it - glb_count.begin();
		for(int j = 0; j < numof_points; j++) {
			if(membership[j] == i) predicted_labels[j] = true_label;
		}
	}
	
	double local_ndiff = 0.0, glb_ndiff = 0.0;
	int glb_numof_points = 0;
	for(int i = 0; i < numof_points; i++) {
		if(labels[i] != predicted_labels[i]) local_ndiff++;
	}
		
	MPI_Allreduce(&local_ndiff, &glb_ndiff, 1, MPI_DOUBLE, MPI_SUM, comm);
	MPI_Allreduce(&numof_points, &glb_numof_points, 1, MPI_INT, MPI_SUM, comm);
	double accuracy = 1.0 - glb_ndiff / (double)glb_numof_points;

	return accuracy;
}


double calMean(vector<double> &arr)
{
	int n = arr.size();
	double sum = 0.0;
	for(int i = 0; i < n; i++)
		sum += arr[i];

	return sum / (double)n;
}

double calStd(vector<double> &arr)
{
	int n = arr.size();
	double sum = 0.0;
	double avr = calMean(arr);
	for(int i = 0; i < n; i++) {
		sum += (arr[i]-avr) * (arr[i]-avr);
	}
	sum = sum / (double)n;

	return sqrt(sum);
}



int main(int argc, char **argv) {

	CmdLine cmd;
	const char *pchHelp = "clustering performance testing - general info.";
	cmd.addInfo(pchHelp);
	char *ptrDataType = NULL;
	cmd.addSCmdOption("-dt", &ptrDataType, "line", "data types, must be one of 'line', 'uni', 'norm', 'hyper', 'sphere', 'mog', 'rmog', 'smog' (default = line)");
	int numof_clusters;
	cmd.addICmdOption("-c", &numof_clusters, 4, "number of clusters (default = 4) ");
	int max_iter;
	cmd.addICmdOption("-mi", &max_iter, 5, "maximum iteration of kmeans (default = 5)");
	int numof_points;
	cmd.addICmdOption("-np", &numof_points, 50000, "number of points generated per proc (default = 50000)");
	int dim;
	cmd.addICmdOption("-dim", &dim, 10, "dimensionality of points generated per proc (default = 10)");
	int numof_gaussians;
	cmd.addICmdOption("-ng", &numof_gaussians, 10, "number of gaussians (only works for mixture of gaussian distribution) (default = 10)");
	double dist_among_groups;
	cmd.addRCmdOption("-ds", &dist_among_groups, 5.0, "minumum distancen among different gaussians/spheres when generating mog/sphere data (default = 5.0)");
	double max_eigv;
	cmd.addRCmdOption("-me", &max_eigv, 5.0, "maximum eigenvalues of the covariance for a gaussian (default = 5.0)");
	int max_irun;
	cmd.addICmdOption("-irun", &max_irun, 5, "run experiments using the same data 'irun' times (default = 5)");
	cmd.read(argc, argv);

       	int        rank, nproc;
	double     start_time, end_time, seeding_time;
		
	MPI_Status status;
	MPI_Comm comm = MPI_COMM_WORLD;
    	MPI_Init(&argc, &argv);
    	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    	MPI_Comm_size(MPI_COMM_WORLD, &nproc);

	vector<double> points;
	points.resize(dim*numof_points);
	vector<int> dummy_ids;
	dummy_ids.resize(numof_points);
	vector<int> labels;
	labels.resize(numof_points);

	string strDataType = ptrDataType;
	if( strDataType.compare("hyper") == 0  ) {
		generateUnitHypersphere(numof_points, dim, &(points[0]), comm);
	}

	if( strDataType.compare("line") == 0  ) {
		genPointInRandomLine((long)numof_points, dim, 
				&(points[0]), &(dummy_ids[0]), comm, true, 0);
	}

	if( strDataType.compare("uni") == 0  ) {
		generateUniform(numof_points, dim, &(points[0]), comm);
	}

	if( strDataType.compare("norm") == 0  ) {
		generateNormal(numof_points, dim, &(points[0]), comm);
	}

	if( strDataType.compare("sphere") == 0  ) {
		generateNSphere(numof_points, dim, dist_among_groups, &(points[0]), comm);
	}

	if( strDataType.compare("mog") == 0  ) {
		generateMixOfUnitGaussian(numof_points, dim, 
				numof_gaussians, dist_among_groups, 
				&(points[0]), &(labels[0]), comm);
	}

	if( strDataType.compare("smog") == 0  ) {
		vector<double> eigv;
		eigv.resize(dim);
		eigv[0] = 1; eigv[1] = 0.5; eigv[2] = 0.5; eigv[3] = 0.1;
		generateMixOfSpecifiedGaussian(numof_points, dim, 
				numof_gaussians, dist_among_groups, &(eigv[0]), 
				&(points[0]), &(labels[0]), comm);
	}
	
	if( strDataType.compare("rmog") == 0  ) {
		generateMixOfRandomGaussian(numof_points, dim, 
				numof_gaussians, dist_among_groups, max_eigv, 
				&(points[0]), &(labels[0]), comm);
	}


	/*if(rank == 0) remove("data.check");
	for(int k = 0; k < nproc; k++) {
		if(rank == k) {
			ofstream outdata("data.check", ios::app|ios::out);
			for(int i = 0; i < numof_points; i++) {
				for(int j = 0; j < dim; j++)
					outdata<<points[i*dim+j]<<" ";
				outdata<<endl;
			}
			outdata.close();
		}
		MPI_Barrier(comm);
	}*/


	vector<double> centroids;
	centroids.resize(dim*numof_clusters);
	vector<int> local_cluster_size;
	local_cluster_size.resize(numof_clusters);
	vector<int> glb_cluster_size;
	glb_cluster_size.resize(numof_clusters);
	vector<int> point_to_cluster_membership;
	point_to_cluster_membership.resize(numof_points);

	pair<double, int> kmeans_result;
	int glb_numof_points = 0;

	vector<double> tmp_centroids;
	double VR = 0.0, accuracy = 0.0;

	vector<double> arr_vr_prob(max_irun);
	vector<double> arr_loss_prob(max_irun);
	vector<double> arr_seeding_prob(max_irun);
	vector<double> arr_accuracy_prob(max_irun);
	vector<double> arr_kmeans_prob(max_irun);
	vector<double> arr_iter_prob(max_irun);

	vector<double> arr_vr_prob2(max_irun);
	vector<double> arr_loss_prob2(max_irun);
	vector<double> arr_seeding_prob2(max_irun);
	vector<double> arr_accuracy_prob2(max_irun);
	vector<double> arr_kmeans_prob2(max_irun);
	vector<double> arr_iter_prob2(max_irun);

	vector<double> arr_vr_rand(max_irun);
	vector<double> arr_loss_rand(max_irun);
	vector<double> arr_seeding_rand(max_irun);
	vector<double> arr_accuracy_rand(max_irun);
	vector<double> arr_kmeans_rand(max_irun);
	vector<double> arr_iter_rand(max_irun);

	for(int nc = 4; nc <= 4; nc++) {
		numof_clusters = nc;

		for(int irun = 0; irun < max_irun; irun++) {
			if(rank == 0) cout<<"The "<<irun+1<<"th run:"<<endl;
			
			usleep(500);
			MPI_Barrier(comm);
			if(rank == 0) cout<<"   info of random seeds: "<<endl;
			start_time = MPI_Wtime();
			random_seeds(&(points[0]), numof_points, dim, 
					numof_clusters, &(centroids[0]), comm);
			end_time = MPI_Wtime();
			seeding_time = end_time - start_time;
			//for(int iter = 0; iter <= max_iter; iter++) {
			tmp_centroids.resize(centroids.size());
			copy(centroids.begin(), centroids.end(), tmp_centroids.begin());
			start_time = MPI_Wtime();
			kmeans_result = kmeans(&(points[0]), dim, 
				numof_points, numof_clusters, max_iter, 
				&(point_to_cluster_membership[0]), &(tmp_centroids[0]),
				&(glb_cluster_size[0]), &(local_cluster_size[0]), comm);
			end_time = MPI_Wtime();
			VR = VarianceRatio(&(points[0]), numof_points, dim, 
					&(tmp_centroids[0]), numof_clusters, 
					&(glb_cluster_size[0]), comm);
			accuracy = check_correctness(numof_points, dim, 
					&(labels[0]), &(point_to_cluster_membership[0]),
					numof_gaussians, comm);
			arr_loss_rand[irun] = kmeans_result.first;
			arr_vr_rand[irun] = VR;
			arr_iter_rand[irun] = (double)kmeans_result.second;
			arr_seeding_rand[irun] = seeding_time;
			arr_kmeans_rand[irun] = end_time - start_time;
			arr_accuracy_rand[irun] = accuracy;
			if(rank == 0) {
				cout<<"\tnumof_clusters: "<<numof_clusters
				    <<"\tloss_func: "<<kmeans_result.first
				    <<"\tvariance_ratio: "<<VR
				    <<"\tseeding_time: "<<seeding_time
				    <<"\t#iter: "<<kmeans_result.second
				    <<"\tkmeans_time: "<<end_time-start_time
				    <<"\taccuracy: "<<accuracy
				    <<endl;
				//printMat(tmp_centroids, numof_clusters, dim);
				//printVec(glb_cluster_size, numof_clusters);
			}	
			//} //for(iter<=max_iter)

			
			usleep(500);
			MPI_Barrier(comm);
			if(rank == 0) cout<<"   info of probabilisitc seeds (just sampling): "<<endl;
			int numof_seeds = 0;
			start_time = MPI_Wtime();
			double *aux_prob = new double [numof_points];
			double *aux_Dx = new double [numof_points];
			double *aux_dist = new double [numof_points*numof_clusters];
			prob_seeds(&(points[0]), numof_points, dim,
				&(centroids[0]), numof_clusters, &numof_seeds, 
				aux_prob, aux_Dx, aux_dist, comm);
			delete [] aux_prob;
			delete [] aux_Dx;
			delete [] aux_dist;
			end_time = MPI_Wtime();
			seeding_time = end_time - start_time;
			//for(int iter = 0; iter <= max_iter; iter++) {
			tmp_centroids.resize(centroids.size());
			copy(centroids.begin(), centroids.end(), tmp_centroids.begin());
			start_time = MPI_Wtime();
			kmeans_result = mpi_kmeans(&(points[0]), dim, 
				numof_points, numof_clusters, max_iter, 
				&(point_to_cluster_membership[0]), &(tmp_centroids[0]),
				&(glb_cluster_size[0]), &(local_cluster_size[0]), comm);
			end_time = MPI_Wtime();
			VR = VarianceRatio(&(points[0]), numof_points, dim, &(tmp_centroids[0]), 
					numof_clusters, &(glb_cluster_size[0]), comm);
			accuracy = check_correctness(numof_points, dim, 
					&(labels[0]), &(point_to_cluster_membership[0]),
					numof_gaussians, comm);
			arr_loss_prob[irun] = kmeans_result.first;
			arr_vr_prob[irun] = VR;
			arr_iter_prob[irun] = (double)kmeans_result.second;
			arr_seeding_prob[irun] = seeding_time;
			arr_kmeans_prob[irun] = end_time - start_time;
			arr_accuracy_prob[irun] = accuracy;
			if(rank == 0) {
				cout<<"\tnumof_clusters: "<<numof_clusters
				    <<"\tloss_func: "<<kmeans_result.first
				    <<"\tvariance_ratio: "<<VR
				    <<"\tseeding_time: "<<seeding_time
				    <<"\t#iter: "<<kmeans_result.second
				    <<"\tkmeans_time: "<<end_time-start_time
				    <<"\taccuracy: "<<accuracy
				    <<endl;
				//printMat(tmp_centroids, numof_clusters, dim);
				//printVec(glb_cluster_size, numof_clusters);
			}
			//} // for(iter <= max_iter)

			
			usleep(500);
			MPI_Barrier(comm);
			if(rank == 0) cout<<"   info of probabilisitc seeds (2 stage): "<<endl;
			start_time = MPI_Wtime();
			add_seeds_2stage(&(points[0]), numof_points, dim, numof_clusters, &(centroids[0]), comm);
			end_time = MPI_Wtime();
			seeding_time = end_time - start_time;
			//for(int iter = 0; iter <= max_iter; iter++) {
			tmp_centroids.resize(centroids.size());
			copy(centroids.begin(), centroids.end(), tmp_centroids.begin());
			start_time = MPI_Wtime();
			kmeans_result = mpi_kmeans(&(points[0]), dim, 
				numof_points, numof_clusters, max_iter, 
				&(point_to_cluster_membership[0]), &(tmp_centroids[0]),
				&(glb_cluster_size[0]), &(local_cluster_size[0]), comm);
			end_time = MPI_Wtime();
			VR = VarianceRatio(&(points[0]), numof_points, dim, &(tmp_centroids[0]), 
					numof_clusters, &(glb_cluster_size[0]), comm);
			accuracy = check_correctness(numof_points, dim, 
					&(labels[0]), &(point_to_cluster_membership[0]),
					numof_gaussians, comm);
			arr_loss_prob2[irun] = kmeans_result.first;
			arr_vr_prob2[irun] = VR;
			arr_iter_prob2[irun] = (double)kmeans_result.second;
			arr_seeding_prob2[irun] = seeding_time;
			arr_kmeans_prob2[irun] = end_time - start_time;
			arr_accuracy_prob2[irun] = accuracy;
			if(rank == 0) {
				cout<<"\tnumof_clusters: "<<numof_clusters
				    <<"\tloss_func: "<<kmeans_result.first
				    <<"\tvariance_ratio: "<<VR
				    <<"\tseeding_time: "<<seeding_time
				    <<"\t#iter: "<<kmeans_result.second
				    <<"\tkmeans_time: "<<end_time-start_time
				    <<"\taccuracy: "<<accuracy
				    <<endl;
				//printMat(tmp_centroids, numof_clusters, dim);
				//printVec(glb_cluster_size, numof_clusters);
			}
			//} // for(iter <= max_iter)


		} // for(irun)
	

		if(rank == 0) {
		cout<<endl;
		
		cout<<"summary of random seeds: "<<endl;
		cout<<"\tnumof_clusters: "<<numof_clusters
		    <<"\tloss_func: "<<calMean(arr_loss_rand)<<" +- "<<calStd(arr_loss_rand)
		    <<"\tvariance_ratio: "<<calMean(arr_vr_rand)<<" +- "<<calStd(arr_vr_rand)
		    <<"\tseeding_time: "<<calMean(arr_seeding_rand)<<" +- "<<calStd(arr_seeding_rand)
		    <<"\t#iter: "<<calMean(arr_iter_rand)<<" +- "<<calStd(arr_iter_rand)
		    <<"\tkmeans_time: "<<calMean(arr_kmeans_rand)<<" +- "<<calStd(arr_kmeans_rand)
		    <<"\taccuracy: "<<calMean(arr_accuracy_rand)<<" +- "<<calStd(arr_accuracy_rand)
		    <<endl;

		cout<<"summary of probabilistic seeds (sampling): "<<endl;
		cout<<"\tnumof_clusters: "<<numof_clusters
		    <<"\tloss_func: "<<calMean(arr_loss_prob)<<" +- "<<calStd(arr_loss_prob)
		    <<"\tvariance_ratio: "<<calMean(arr_vr_prob)<<" +- "<<calStd(arr_vr_prob)
		    <<"\tseeding_time: "<<calMean(arr_seeding_prob)<<" +- "<<calStd(arr_seeding_prob)
		    <<"\t#iter: "<<calMean(arr_iter_prob)<<" +- "<<calStd(arr_iter_prob)
		    <<"\tkmeans_time: "<<calMean(arr_kmeans_prob)<<" +- "<<calStd(arr_kmeans_prob)
		    <<"\taccuracy: "<<calMean(arr_accuracy_prob)<<" +- "<<calStd(arr_accuracy_prob)
		    <<endl;
		
		cout<<"summary of probabilistic seeds (2 stage): "<<endl;
		cout<<"\tnumof_clusters: "<<numof_clusters
		    <<"\tloss_func: "<<calMean(arr_loss_prob2)<<" +- "<<calStd(arr_loss_prob2)
		    <<"\tvariance_ratio: "<<calMean(arr_vr_prob2)<<" +- "<<calStd(arr_vr_prob2)
		    <<"\tseeding_time: "<<calMean(arr_seeding_prob2)<<" +- "<<calStd(arr_seeding_prob2)
		    <<"\t#iter: "<<calMean(arr_iter_prob2)<<" +- "<<calStd(arr_iter_prob2)
		    <<"\tkmeans_time: "<<calMean(arr_kmeans_prob2)<<" +- "<<calStd(arr_kmeans_prob2)
		    <<"\taccuracy: "<<calMean(arr_accuracy_prob2)<<" +- "<<calStd(arr_accuracy_prob2)
		    <<endl;
		
		cout<<"========================================================================="<<endl;
		}


	}// for(nc)
	

	MPI_Finalize();

    	return 0;
}

