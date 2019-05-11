#include<direct_knn.h>
#include<knnreduce.h>
#include<generator.h>
#include<vector>
#include<cassert>
#include<cmath>
#include<utility>
#include<omp.h>
#include<mpi.h>
#include <ompUtils.h>
#include <ctime>

#include "mpitree.h"
#include "clustering.h"
#include "repartition.h"
#include "binTree.h"
#include "binQuery.h"
#include "distributeToLeaf.h"
#include "eval.h"

using namespace std;



__inline static
double find_max(double *arr, int first, int last)
{
	double maxv = 0.0;
    maxv = *max_element(arr+first, arr+last);
    return maxv;
}


void printBinTree(pbinNode in_node)
{
	int nproc, rank;
	//MPI_Comm in_comm = in_node->comm;
	//MPI_Comm_size(in_comm, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if(NULL == in_node->kid) { 	// if leaf
		cout<<rank
		    <<" - "<<pow(2.0, (double)in_node->level)-1+in_node->chid
		    <<" - "<<in_node->level
		    <<" - "<<in_node->data->gids.size()
		    <<endl;
		fflush(stdout);
		return;
	}
	else {				// if not leaf
		printBinTree(in_node->kid);
	}

}



void saveBinTree(char * filename, pbinNode in_node)
{
	int worldnproc, worldrank, commsize;
	MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
	MPI_Comm_size(in_node->comm, &commsize);

	ofstream fout(filename, ios::app|ios::out);
        
	if(NULL == in_node->kid) { 	// if leaf

		fout<<worldrank
		    <<" "<<in_node->level
		    <<" "<<pow(2.0, (double)in_node->level)-1+in_node->chid
		    <<" "<<in_node->data->gids.size()
		    <<" "<<commsize
		    <<endl;
		//for(int i = 0; i < in_node->data->gids.size(); i++)
		//	fout<<in_node->data->gids[i]<<" ";
		//fout<<endl;

		fout.flush();
		fout.close();
		return;
	}
	else {				// if not leaf
		saveBinTree(filename, in_node->kid);
	}

}


void get_sample_info(double *ref, double *query, long *refids, long *queryids,
					int numof_ref_points, int numof_query_points,
					int dim, int k,
					//
					vector<long> &sampleIDs,
					vector<double> &globalKdist,
					vector<long> &globalKid)
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	double start_t;

	// ----- random sample logm points for evaluation purpose -----
	long mglobal;
	long tmp_numof_query_points = numof_query_points;
	MPI_Allreduce(&tmp_numof_query_points, &mglobal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
	int logm = std::min( (double)mglobal, 100*std::ceil(log10((double)mglobal)/log10(2.0)) );

	//if(rank == 0) cout<<"\n----- Sample "<<logm<<" points -----"<<endl;
	double *directquery = new double[logm*dim];
	//long *sampleIDs = new long[logm];
	sampleIDs.resize(logm);
	bintree::uniformSample(query, numof_query_points, dim, logm, directquery, &(sampleIDs[0]), MPI_COMM_WORLD);

	//Perform a direct query on local reference points.
	//double *globalKdist = new double[logm*k];
	//long *globalKid = new long [logm*k];
	globalKdist.resize(logm*k);
	globalKid.resize(logm*k);
	start_t = omp_get_wtime();
	pair<double, long> *localResult = new pair<double, long> [logm*k];
	knn::directKQueryLowMem(ref, directquery, numof_ref_points, logm, k, dim, localResult);

	#pragma omp parallel if(logm > 1000)
	{
		#pragma omp for
		for(int i = 0; i < logm; i++) {
			for(int j = 0; j < k; j++)
				localResult[i*k+j].second = refids[localResult[i*k+j].second];
		}
	}
	pair<double, long> *mergedResult;
	knn::query_k(MPI_COMM_WORLD, k, 0, localResult, logm, k, mergedResult);
	double partial_direct_nn_t = omp_get_wtime() - start_t;
	double max_nn_t = 0.0;
	MPI_Reduce(&partial_direct_nn_t, &max_nn_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	//if(rank == 0) cout<<"Estimated direct time: "<<((double)mglobal/(double)logm)*max_nn_t<<endl;

	if(rank == 0) {
		#pragma omp parallel if( logm > 1000 )
		{
			#pragma omp for
			for(int i = 0; i < logm; i++) {
				for(int j = 0; j < k; j++) {
					globalKdist[i*k+j] = mergedResult[i*k+j].first;
					globalKid[i*k+j] = mergedResult[i*k+j].second;
				}
			}     
		}
	}
	MPI_Bcast(&(globalKdist[0]), logm*k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&(globalKid[0]), logm*k, MPI_LONG, 0, MPI_COMM_WORLD);

	delete[] mergedResult;
	delete[] localResult;
	delete[] directquery;

}


void verify(// input: direct search info
				vector<long> &sampleIDs,
				vector<double> &globalKdist,
				vector<long> &globalKid,
				// input: approx search info
				vector<long> &queryIDs,
				vector< pair<double, long> > &kNN,
				// output
				int &numof_missed_neighbors,
				double &hit_rate,
				double &relative_error)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int logm = sampleIDs.size();
	int k = globalKid.size() / logm;
	hit_rate = -1.0;
	relative_error = -1.0;
	numof_missed_neighbors = -1;

	//Check error for random sample of query points
	double localErrorSum = 0.0, globalErrorSum = 0.0;
	double localHitRate = 0.0, globalHitRate = 0.0;
	int local_numof_missed_neighbors = 0, global_numof_missed_neighbors = 0;
    double local_total_true_dist = 0.0, glb_total_true_dist = 0.0;
    double local_total_knn_dist = 0.0, glb_total_knn_dist = 0.0;

    double acc_true_dist = 0.0, acc_knn_dist = 0.0;
    bool pout = true;

	//#pragma omp parallel
    {
		vector<long> tmpVector(k);
		//#pragma omp for reduction(+:localErrorSum,localHitRate)
        for(int j = 0; j < logm; j++) {
			vector<long>::iterator id = std::find( queryIDs.begin(), queryIDs.end(), sampleIDs[j] );
			int i = id - queryIDs.begin();

            acc_true_dist = 0.0;
            acc_knn_dist = 0.0;

            if( id != queryIDs.end() && *id  == sampleIDs[j] ) {
				copy(globalKid.begin()+j*k, globalKid.begin()+j*k+k, tmpVector.begin());

                for(int t = 0; t < k; t++) {
                    long tmpidx = (long)i*(long)k+(long)t;

                    local_total_true_dist += globalKdist[j*k+t];
                    local_total_knn_dist += kNN[tmpidx].first;

                    acc_true_dist += globalKdist[j*k+t];
                    acc_knn_dist += kNN[tmpidx].first;

					vector<long>::iterator it = find(tmpVector.begin(), tmpVector.end(), kNN[tmpidx].second);
					if(it != tmpVector.end()) localHitRate += 1.0;
					if(kNN[tmpidx].second == -1L) local_numof_missed_neighbors++;
                    if(kNN[tmpidx].second != -1L &&
                            globalKdist[j*k+t] > 0 && globalKid[j*k+t] != kNN[tmpidx].second ) {
						double approx_dist = kNN[tmpidx].first < 0.0 ? 0.0 : kNN[tmpidx].first;
						double error = std::abs(std::sqrt(approx_dist)-std::sqrt(globalKdist[j*k+t]));
						localErrorSum += error / std::sqrt(globalKdist[j*k+t]);
					}
				}

                if( false && pout && acc_knn_dist - acc_true_dist < -1.0e-10) {
                    pout = false;
                    if(rank == 0) {
                        cout<<"true knn: ";
                        for(int t = 0; t < k; t++) {
                            cout<<"("<<globalKid[j*k+t]<<" - "<<globalKdist[j*k+t]<<")  ";
                        }
                        cout<<endl;
                        cout<<"estl knn: ";
                        for(int t = 0; t < k; t++) {
                            long tmpidx = i*k+t;
                            cout<<"("<<kNN[tmpidx].second<<" - "<<kNN[tmpidx].first<<")  ";
                        }
                        cout<<endl;
                    }
                }

			} // if (outIDs == sampleIDs)
		}
	}

	localErrorSum /= (double)(k);
	localHitRate /= (double)(k);
	MPI_Allreduce(&local_numof_missed_neighbors, &global_numof_missed_neighbors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&localErrorSum, &globalErrorSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&localHitRate, &globalHitRate, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	hit_rate = globalHitRate/(double)logm * 100.0;
	relative_error = globalErrorSum/(double)logm * 100.0;
	numof_missed_neighbors = global_numof_missed_neighbors;

    MPI_Allreduce(&local_total_true_dist, &glb_total_true_dist, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&local_total_knn_dist, &glb_total_knn_dist, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    //if(rank == 0) {
	//cout<<"\ttotal_true_dist = "<<glb_total_true_dist<<", total_knn_dist = "<<glb_total_knn_dist
	//	<<", local_true_dist = "<<local_total_true_dist<<", local_knn_dist = "<<local_total_knn_dist<<endl<<endl;
    //}

}



void verify(// input: direct search info
				vector<long> &sampleIDs,
				vector<double> &globalKdist,
				vector<long> &globalKid,
				// input: approx search info
				vector<long> &queryIDs,
				vector< pair<double, long> > &kNN,
				// output
				int &numof_missed_neighbors,
				double &hit_rate,
				double &relative_error,
                double &glb_total_true_dist,
                double &glb_total_knn_dist)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int logm = sampleIDs.size();
	int k = globalKid.size() / logm;
	hit_rate = -1.0;
	relative_error = -1.0;
	numof_missed_neighbors = -1;

	//Check error for random sample of query points
	double localErrorSum = 0.0, globalErrorSum = 0.0;
	double localHitRate = 0.0, globalHitRate = 0.0;
	int local_numof_missed_neighbors = 0, global_numof_missed_neighbors = 0;
    double local_total_true_dist = 0.0, local_total_knn_dist = 0.0;

    glb_total_knn_dist = 0.0;
    glb_total_true_dist = 0.0;

    double acc_true_dist = 0.0, acc_knn_dist = 0.0;
    bool pout = true;

	//#pragma omp parallel
    {
		vector<long> tmpVector(k);
		//#pragma omp for reduction(+:localErrorSum,localHitRate)
        for(int j = 0; j < logm; j++) {
			vector<long>::iterator id = std::find( queryIDs.begin(), queryIDs.end(), sampleIDs[j] );
			int i = id - queryIDs.begin();

            acc_true_dist = 0.0;
            acc_knn_dist = 0.0;

            if( id != queryIDs.end() && *id  == sampleIDs[j] ) {
				copy(globalKid.begin()+j*k, globalKid.begin()+j*k+k, tmpVector.begin());

                for(int t = 0; t < k; t++) {
                    long tmpidx = (long)i*(long)k+(long)t;

                    local_total_true_dist += globalKdist[j*k+t];
                    local_total_knn_dist += kNN[tmpidx].first;

                    acc_true_dist += globalKdist[j*k+t];
                    acc_knn_dist += kNN[tmpidx].first;

					vector<long>::iterator it = find(tmpVector.begin(), tmpVector.end(), kNN[tmpidx].second);
					if(it != tmpVector.end()) localHitRate += 1.0;
					if(kNN[tmpidx].second == -1L) local_numof_missed_neighbors++;
                    if(kNN[tmpidx].second != -1L &&
                            globalKdist[j*k+t] > 0 && globalKid[j*k+t] != kNN[tmpidx].second ) {
						double approx_dist = kNN[tmpidx].first < 0.0 ? 0.0 : kNN[tmpidx].first;
						double error = std::abs(std::sqrt(approx_dist)-std::sqrt(globalKdist[j*k+t]));
						localErrorSum += error / std::sqrt(globalKdist[j*k+t]);
					}
				}   // end for t


                if( pout && acc_knn_dist - acc_true_dist < -1.0e-10) {
                    pout = false;
                    if(rank == 0) {
                        cout<<"true knn: ";
                        for(int t = 0; t < k; t++)
                            cout<<"("<<globalKid[j*k+t]<<" - "<<globalKdist[j*k+t]<<")  ";
                        cout<<endl;
                        cout<<"estl knn: ";
                        for(int t = 0; t < k; t++) {
                            long tmpidx = i*k+t;
                            cout<<"("<<kNN[tmpidx].second<<" - "<<kNN[tmpidx].first<<")  ";
                        }
                        cout<<endl;
                    }
                }
			} // if (outIDs == sampleIDs)
		}
	}

	localErrorSum /= (double)(k);
	localHitRate /= (double)(k);
	MPI_Allreduce(&local_numof_missed_neighbors, &global_numof_missed_neighbors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&localErrorSum, &globalErrorSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&localHitRate, &globalHitRate, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	hit_rate = globalHitRate/(double)logm * 100.0;
	relative_error = globalErrorSum/(double)logm * 100.0;
	numof_missed_neighbors = global_numof_missed_neighbors;

    MPI_Allreduce(&local_total_true_dist, &glb_total_true_dist, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&local_total_knn_dist, &glb_total_knn_dist, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

   //if(rank == 0) {
   //    cout<<"\ttotal_true_dist = "<<glb_total_true_dist
   //         <<", total_knn_dist = "<<glb_total_knn_dist
	//        <<", local_true_dist = "<<local_total_true_dist
    //        <<", local_knn_dist = "<<local_total_knn_dist
    //        <<endl<<endl;
   // }

}



void verify(// input: direct search info
				vector<long> &sampleIDs,
				vector<double> &globalKdist,
				vector<long> &globalKid,
				// input: approx search info
				vector<long> &queryIDs,
				vector< pair<double, long> > &kNN,
				MPI_Comm comm,
				// output
				int &numof_missed_neighbors,
				double &hit_rate,
				double &relative_error)
{
	int logm = sampleIDs.size();
	int k = globalKid.size() / logm;
	hit_rate = -1.0;
	relative_error = -1.0;
	numof_missed_neighbors = -1;

	//Check error for random sample of query points
	double localErrorSum = 0.0, globalErrorSum = 0.0; 
	double localHitRate = 0.0, globalHitRate = 0.0;
	int local_numof_missed_neighbors = 0, global_numof_missed_neighbors = 0;
	#pragma omp parallel
	{
		vector<long> tmpVector(k);
		#pragma omp for reduction(+:localErrorSum,localHitRate)
		for(int j = 0; j < logm; j++) {
			vector<long>::iterator id = std::lower_bound( queryIDs.begin(), queryIDs.end(), sampleIDs[j] );
			int i = id - queryIDs.begin();
			if( id != queryIDs.end() && *id  == sampleIDs[j] ) {
				for(int t = 0; t < k; t++) {
					copy(globalKid.begin()+j*k, globalKid.begin()+j*k+k, tmpVector.begin());
					vector<long>::iterator it = find(tmpVector.begin(), tmpVector.end(), kNN[i*k+t].second);
					if(it != tmpVector.end()) localHitRate += 1.0;
					if(kNN[i*k+t].second == -1L) local_numof_missed_neighbors++;
					if(kNN[i*k+t].second != -1L && globalKdist[j*k+t] > 0 && globalKid[j*k+t] != kNN[i*k+t].second ) {
						double approx_dist = kNN[i*k+t].first < 0.0 ? 0.0 : kNN[i*k+t].first;
						double error = std::abs( std::sqrt(approx_dist) - std::sqrt(globalKdist[j*k+t]) );
						localErrorSum += error / std::sqrt(globalKdist[j*k+t]);
					}
				}
			}
		}

	}
     
	localErrorSum /= (double)(k); 
	localHitRate /= (double)(k);
	MPI_Allreduce(&local_numof_missed_neighbors, &global_numof_missed_neighbors, 1, MPI_INT, MPI_SUM, comm);
	MPI_Allreduce(&localErrorSum, &globalErrorSum, 1, MPI_DOUBLE, MPI_SUM, comm);
	MPI_Allreduce(&localHitRate, &globalHitRate, 1, MPI_DOUBLE, MPI_SUM, comm);
	hit_rate = globalHitRate/(double)logm * 100.0;
	relative_error = globalErrorSum/(double)logm * 100.0;
	numof_missed_neighbors = global_numof_missed_neighbors;
	
}



void evaluation_sample(double *ref, double *query,
						long *refids, long *queryids,
						int numof_ref_points, int numof_query_points,
						int dim, int k, 
						vector<long> &queryIDs, 
						vector< pair<double, long> > &kNN)
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	vector<long> sampleIDs;
	vector<double> globalKdist;
	vector<long> globalKid;
	get_sample_info(ref, query, refids, queryids, numof_ref_points, numof_query_points, dim, k,
					sampleIDs, globalKdist, globalKid);

	double hit_rate = 0.0, relative_error = 0.0;
	int nmiss = 0;
	verify(	sampleIDs, globalKdist, globalKid,
				queryIDs, kNN,
				nmiss, hit_rate, relative_error);

	if( rank == 0) cout << "validate results against direct knn using sampled (logm) points: "<<endl;
	if( rank == 0) cout << "- Hit Rate (sampled): " << hit_rate << "%" << endl;
	if( rank == 0) cout << "- Mean error (sampled): " << relative_error << "%" << endl;

}




void evaluation_full( double *ref, double *query, 
					  long *refids, long *queryids,
					  int numof_ref_points, int numof_query_points,
					  int dim, int k,
					  vector<long> &outQueryIDs,
					  vector< pair<double, long> > &approxkNN)
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int global_numof_query_points;
	MPI_Allreduce( &numof_query_points, &global_numof_query_points, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
	
	int nmatch = 0;
	for(int i = 0; i < numof_query_points; i++) {
		if(outQueryIDs[i] == queryids[i]) nmatch++;
	}
	if(nmatch != numof_query_points) {
		if(rank == 0) cout<<"query ids do not match, check does not make sense!"<<endl;
		return;
	}


	// 1. compute direct knn results
	double *ref_clone = new double [numof_ref_points*dim];
	long *refids_clone = new long [numof_ref_points];
	#pragma omp parallel for
	for(int i = 0; i < numof_ref_points*dim; i++) 
		ref_clone[i] = ref[i];
	#pragma omp parallel for
	for(int i = 0; i < numof_ref_points; i++) 
		refids_clone[i] = refids[i];
	std::pair<double, long> *directkNN = knn::dist_directKQuery
					(ref_clone, query, refids_clone,
					 numof_ref_points, numof_query_points, k, dim, MPI_COMM_WORLD);

	delete [] ref_clone;
	delete [] refids_clone;

	// 2. compute error rate and hit rate
	double localError = 0.0, glbError = 0.0;
	double localHit = 0.0, glbHit = 0.0;
	for(int i = 0; i < numof_query_points; i++) {
		vector<long> tmparr(k);
		for(int j = 0; j < k; j++)
			tmparr[j] = directkNN[i*k+j].second;
		for(int j = 0; j < k; j++) {
			if(directkNN[i*k+j].first > 0.00000000001) {
				vector<long>::iterator it = find(tmparr.begin(), tmparr.end(), approxkNN[i*k+j].second);
				if( it != tmparr.end() ) localHit += 1.0;
				localError += abs(sqrt(abs(approxkNN[i*k+j].first)) - sqrt(abs(directkNN[i*k+j].first))) / sqrt(abs(directkNN[i*k+j].first));
			}
		}
	}

	localError /= (double)(k-1);
	localHit /= (double)(k-1);
	MPI_Allreduce(&localError, &glbError, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&localHit, &glbHit, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	
	glbError /= (double)global_numof_query_points;
	glbHit /= (double)global_numof_query_points;
	if( rank == 0) cout << "validate results against direct knn using all points: "<<endl;
	if(rank == 0) cout<<"- Hit Rate: "<<glbHit*100.0<<"%"<<endl;
	if(rank == 0) cout<<"- Mean error: "<<glbError*100.0<<"%"<<endl;

}












