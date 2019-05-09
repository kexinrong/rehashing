#ifndef _DISTRIBUTETOLEAF_H_
#define _DISTRIBUTETOLEAF_H_

#include <mpi.h>
#include <cassert>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <utility>
#include <util.h>

#include "binTree.h"

namespace bintree {


	void distributeToLeaves(pbinData inData, long rootNpoints,
						double dupFactor, pbinNode searchNode,
						double range,
						pbinData *outData, pbinNode *leaf);

	void distributeToLeaves(pbinData inData, long rootNpoints,
						double dupFactor, pbinNode searchNode,
						pbinData *outData, pbinNode *leaf);

	void distributeToNearestLeaf( pbinData inData,
								  pbinNode searchNode,
								  pbinData *outData,
								  pbinNode *leaf);

    void GoToNearestLeafbyMedian( pbinData inData, pbinNode searchNode,
								  pbinData *outData, pbinNode *leaf);

	void randperm(int m, int N, vector<long>& arr);

	void uniformSample(	double *points, int numof_points, int dim,
						int numof_sample_points,
						double *sample_points, long *sample_ids,
						MPI_Comm comm);

	void sampleTwoKids(pbinNode inNode, int numof_sample_points,
		             double * sample0, double *sample1);

	void distributeToSampledLeaf( pbinData inData,
								  pbinNode searchNode,
								  pbinData *outData,
								  pbinNode *leaf);

}


#endif




