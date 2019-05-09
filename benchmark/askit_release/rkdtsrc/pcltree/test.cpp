#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <assert.h>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <map>
#include <vector>

#include "file_io.h"
#include "repartition.h"
#include "CmdLine.h"
#include "generator.h"
#include "clustering.h"
#include "direct_knn.h"
#include "mpitree.h"
#include "treeprint.h"



using namespace std;
using namespace knn;
using namespace knn::repartition;
using namespace Torch;

struct TestOptions{
	int numof_points;
	int max_tree_level;
	int max_points_per_node;
	int min_comm_size_per_tree_node;
	int dim;
	double range;
	int num_query_per_process;
	int cf;
	int dbg_out;
	int dbg_cmp;
	int seedType;
};



void ParseInput(CmdLine &cmd, TestOptions &o, int argc, char **argv)
{
	const char *pchHelp = "This is hopeless.";
	cmd.addInfo(pchHelp);
	cmd.addICmdOption("-cp", &o.numof_points, 5000, "number of data points generated per proc (5000)");
	cmd.addICmdOption("-qp", &o.num_query_per_process, 1000, "number of query points per proc (1000)");
	cmd.addICmdOption("-gd", &o.dim, 8, "dimensionality of points generated per proc (8)");
	cmd.addICmdOption("-s", &o.seedType, 1, "0: random seeds, 1: ostrovsky seeds (1)");
	cmd.addRCmdOption("-r",  &o.range, 0.001, "search range");
	cmd.addICmdOption("-mtl", &o.max_tree_level, 4, "maximum tree depth");
	cmd.addICmdOption("-mppn", &o.max_points_per_node, 10, "maximum number of points per tree node");
	cmd.addICmdOption("-mcsptn", &o.min_comm_size_per_tree_node, 1, "min comm size per tree node");
	cmd.addICmdOption("-cf", &o.cf, 1, "Clustering factor (1)");
	cmd.addICmdOption("-dbg", &o.dbg_out, 0, "Enable debugging output (0)");
	cmd.addICmdOption("-cmp", &o.dbg_cmp, 0, "Enable debugging - compare the result of direct and tree-based query results (0)");
	cmd.read(argc, argv);
}


ostream& printPt( double* pt, int dim ){
	for( int i = 0; i < dim; i++ )
		cout << pt[i] << " ";

	return cout;
}

ostream& printPt( int* pt, int dim ){
	for( int i = 0; i < dim; i++ )
		cout << pt[i] << " ";

	return cout;
}

void printLeaves(MTNode &node) {
	if( node.kid == NULL ) {  //This is a leaf.
		int rank, size;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &size);
		
		cout << "Rank " << rank << "\n" <<
		"Dim = " << (*(node.data)).dim << "\n" <<
		" cluster center: (" << printPt(&(node.C[0]), (*(node.data)).dim) 
		<< ")\n"  <<
		" cluster radius: " << node.R[0] << "\n" << 
		" point ids: ";

		for(int i = 0; i < (*(node.data)).gids.size(); i++ )
			cout << (*(node.data)).gids[i] << " ";

		cout << "\n--------------------------------------------------------";
		cout << endl;

		return;
	} else {
		printLeaves(*(node.kid));
	}

}


bool read_result(const char * filename, map<int, vector<int> > &results)
{
	ifstream Infile;
	string line, ele;
	int n = 0;
	
	Infile.open(filename);
	if(!Infile.is_open()) {
		std::cout<<"cannot open file "<<filename<<endl;
		return false;
	}
	getline(Infile, line); 		// numof query points
	getline(Infile, line);		// range
	pair<map<int, vector<int> >::iterator, bool> ret;
	while( getline(Infile, line) ) {	// point_id nn1_id nn2_id ...
		istringstream stream_line(line);
		stream_line >> ele;
		int qid = atoi(ele.c_str());
		vector<int> nnlist;
		while (stream_line >> ele)
		{
			int nn_id = atoi(ele.c_str());
			nnlist.push_back(nn_id);
		}
		ret = results.insert(make_pair<int, vector<int> >(qid, nnlist));
		if(false == ret.second)	{  // query_id already exist in the map
			for(int i = 0; i < nnlist.size(); i++)
				(ret.first)->second.push_back(nnlist[i]);
		}
		
		//n++;
	}
	Infile.close();

	return true;	
}


void output_result(map<int, vector<int> > &results)
{
	for(map<int, vector<int> >::iterator iter = results.begin();
		iter != results.end(); iter++) {
		cout<<iter->first<<" ";
		for(int i = 0; i < iter->second.size(); i++)
			cout<<iter->second.at(i)<<" ";
		cout<<endl;
	}
}


void compare_result(// input
					map<int, vector<int> > &direct_results, 
					map<int, vector<int> > &tree_results,
					// output
					int &diff,
					double &precision,
					double &recall
					)
{
	map< int, vector<int> >::iterator ret;
	int numof_intersect_nn = 0;
	int numof_direct_nn = 0;
	int numof_tree_nn = 0;
	int numof_diff_nn = 0;
	for( map<int, vector<int> >::iterator iter = direct_results.begin();
	    iter != direct_results.end(); iter++ ) {
		ret = tree_results.find( iter->first );
		if(tree_results.end() == ret) {
			cout<<"ERROR: the query ids of direct way and tree-based way are not the same, there must be something wrong with the codes itself!"<<endl;
			return;
		}
		vector<int>::iterator iter_intersect;
		vector<int> vec_intersect;
		vec_intersect.resize(iter->second.size());
		vector<int>::iterator iter_diff;
		vector<int> vec_diff;
		vec_diff.resize(iter->second.size());
		iter_intersect = set_intersection(iter->second.begin(), iter->second.end(),
										  ret->second.begin(), ret->second.end(), 
										  vec_intersect.begin());
		numof_intersect_nn += int(iter_intersect - vec_intersect.begin());
		numof_direct_nn += iter->second.size();
		numof_tree_nn += ret->second.size();
		iter_diff = set_difference(iter->second.begin(), iter->second.end(), 
								   ret->second.begin(), ret->second.end(),
								   vec_diff.begin());
		numof_diff_nn += int(iter_diff - vec_diff.begin());
	}
	diff = numof_diff_nn;
	precision = (double)numof_intersect_nn / (double)numof_tree_nn;
	recall = (double)numof_intersect_nn / (double)numof_direct_nn;
}





int main( int argc, char **argv)
{
	CmdLine cmd;
	TestOptions o;
	ParseInput(cmd, o, argc, argv);

	int dim = o.dim;
	int numof_points = o.numof_points;

	int        rank, nproc, mpi_namelen;
	char       mpi_name[MPI_MAX_PROCESSOR_NAME];
	double     start_time, end_time;
	MPI_Status status;
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Get_processor_name(mpi_name,&mpi_namelen);
	MPI_Barrier(MPI_COMM_WORLD);

	MTData data;
	data.X.resize(o.dim*o.numof_points);  // points
	data.gids.resize(o.numof_points);       // global ids
	data.dim = dim;

	MTData query;
	query.X.resize(o.dim*o.num_query_per_process);
	query.gids.resize(o.num_query_per_process);
	query.dim = dim;
	
/*
	int *tmp_ids = new int [numof_points];
	genPointInRandomLine(numof_points, dim, &data.X[0], tmp_ids, comm, true, rank*numof_points);
	PRINTSELF( cout << "Number of points " <<  numof_points << endl );
	for(int i = 0; i < numof_points; i++) {	data.gids[i] = long(tmp_ids[i]); } delete [] tmp_ids;
*/

/*
	//Generate points uniformly distributed on the surface of the unit hypersphere  
        generateUnitHypersphere(o.numof_points, dim, &(data.X[0]), MPI_COMM_WORLD);
        #pragma omp parallel for
        for(int i = 0; i < o.numof_points; i++) data.gids[i] = i + numof_points*rank;
*/

/*
	int *labels = new int[numof_points];
        generateMixOfUnitGaussian( o.numof_points, dim,
                                8, 6,
                                &(data.X[0]), labels, MPI_COMM_WORLD);
        #pragma omp parallel for
        for(int i = 0; i < o.numof_points; i++) data.gids[i] = i + numof_points*rank;
*/

	generateNormal(o.numof_points, dim, &(data.X[0]), MPI_COMM_WORLD);
        #pragma omp parallel for
        for(int i = 0; i < o.numof_points; i++) data.gids[i] = i + numof_points*rank;



        #pragma omp parallel for
	for(int i = 0; i < o.num_query_per_process*dim; i++) 
		query.X[i] = data.X[i];

	#pragma omp parallel for
	for(int i = 0; i < o.num_query_per_process; i++)
		query.gids[i] = data.gids[i];
	
	// --------- save data points into file ---------------
	if(o.dbg_out) {
		if(rank == 0) remove("data.check");
		MPI_Barrier(MPI_COMM_WORLD);
		for(int k = 0; k < nproc; k++) {
			if(rank == k ) {
		        	ofstream outdata("data.check", ios::app|ios::out);
				for(int i = 0; i < o.numof_points; i++) {
					outdata<<data.gids[i]<<" ";
					for(int j = 0; j < o.dim; j++)
						outdata<<data.X[i*o.dim+j]<<" ";
					outdata<<endl;
				}
				outdata.close();
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}
	} // if (o.dgg_out)

	
	if(o.dbg_cmp) {
	// + directRquery
		int *dirt_neighbor_count = NULL;
		std::pair<double, long> *dirt_neighbors = NULL;
		directRQuery(&(data.X[0]), &(query.X[0]), o.numof_points, o.num_query_per_process,
				o.range*o.range, dim, &(data.gids[0]), 
				&dirt_neighbor_count, &dirt_neighbors);
	// + output directRquery results
		if(rank == 0) remove("treeQueryResult_direct.check");
		MPI_Barrier(MPI_COMM_WORLD);
		ofstream output_dirt_result;
		if(rank == 0) {
			output_dirt_result.open("treeQueryResult_direct.check", ios::app|ios::out);
			output_dirt_result<<numof_points*nproc<<endl<<o.range<<endl;
			output_dirt_result.close();
		}
		MPI_Barrier(MPI_COMM_WORLD);
		int numof_query_points = query.X.size() / query.dim; 
		for(int k = 0; k < nproc; k++) {
			if(rank == k) {
				output_dirt_result.open("treeQueryResult_direct.check", ios::app|ios::out);
				int offset = 0;
				for(int i = 0; i < numof_query_points; i++) {
					output_dirt_result<<query.gids[i]<<" ";
					for(int j = 0; j < dirt_neighbor_count[i]; j++)
						output_dirt_result<<dirt_neighbors[offset+j].second<<" ";
					output_dirt_result<<endl;
					offset += dirt_neighbor_count[i];
				}
				output_dirt_result.close();
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}

	
		delete [] dirt_neighbor_count;
		delete [] dirt_neighbors;
	} // if(o.dbg_out)
        


	// ========== build a tree ==========
	MTNode root;
	root.options.pruning_verbose=true;
	int numof_kids = 2;
	double time = MPI_Wtime();
	root.Insert( NULL,
		 o.max_points_per_node,
		 o.max_tree_level,
		 o.min_comm_size_per_tree_node,
		 comm,
		 &data,
		 o.seedType);
	MPI_Barrier(MPI_COMM_WORLD);
	if( rank == 0 ) cout << "construction: " << MPI_Wtime() - time << endl;	

	
	// ------------ print tree info -------------
	if(o.dbg_out) {
		MPI_Barrier(MPI_COMM_WORLD);
		if(rank == 0) remove("treeStruct.check");
		MPI_Barrier(MPI_COMM_WORLD);
		ofstream outfile;
		if(rank == 0)  {
			outfile.open("treeStruct.check", ios::app|ios::out);
			outfile<<"dim "<<dim<<" numof_kids "<<numof_kids
			       <<" numof_clusters "
		      	 	<<numof_kids<<endl;
			outfile.close();
		}

		MPI_Barrier(MPI_COMM_WORLD);

		for(int k = 0; k < nproc; k++) {
                        if(rank == k ) {
				outfile.open("treeStruct.check", ios::app|ios::out);
				treeSaveRadii(outfile, &root);
				outfile.close();
                        }
                        MPI_Barrier(MPI_COMM_WORLD);
                }

	} // if(o.dbg_out)
	

	// ============ query ============
	vector< pair<double, long> > *neighbors = NULL;
	double querytime = MPI_Wtime();
	queryR(&query, numof_points*nproc, 16.0, &root, o.range, neighbors);
	// + output the query results
	if(o.dbg_cmp) {
		if(rank == 0) remove("treeQueryResult_tree.check");
		MPI_Barrier(MPI_COMM_WORLD);
		ofstream output_result;
		if(rank == 0) {
			output_result.open("treeQueryResult_tree.check", ios::app|ios::out);
			output_result<<numof_points*nproc<<endl<<o.range<<endl;
			output_result.close();
		}
		MPI_Barrier(MPI_COMM_WORLD);
		int numof_query_points = query.X.size() / query.dim; 
		for(int k = 0; k < nproc; k++) {
			if(rank == k) {
				output_result.open("treeQueryResult_tree.check", ios::app|ios::out);
				for(int i = 0; i < numof_query_points; i++) {
					output_result<<query.gids[i]<<" ";
					for(int j = 0; j < (neighbors[i]).size(); j++)
						output_result<<(neighbors[i]).at(j).second<<" ";
					output_result<<endl;
				}
				output_result.close();
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}
	} // if(o.dbg_cmp)
        if(rank == 0) cout << "query: " << MPI_Wtime() - querytime << endl;
	delete [] neighbors;

	//comparision between direct way and the tree-based way
	if(o.dbg_cmp) {
		if(rank==0) {

			map<int, vector<int> > direct_results;
			read_result("treeQueryResult_direct.check", direct_results);
			//output_result(direct_results);
			
			map<int, vector<int> > tree_results;
			read_result("treeQueryResult_tree.check", tree_results);
			//output_result(tree_results);
			
			int numof_diff = -1;
			double precision = 0.0, recall = 0.0;
			compare_result(direct_results, tree_results, 
						   numof_diff, precision, recall);
			
			cout<<"correctness info: "<<endl;
			cout<<"    numof_diff: "<<numof_diff
				<<"  - precision: "<<precision
				<<"  - recall: "<<recall
				<<endl;
			
		}
	}

	MPI_Finalize();


	return 0;
}
	


	
	


