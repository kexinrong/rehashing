#ifndef _DISTRIBUTETOLEAFOT_H_
#define _DISTRIBUTETOLEAFOT_H_

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
#include "oldTree.h"

namespace oldtree {


	void distributeToLeaves(pbinData inData, long rootNpoints,
						double dupFactor, poldNode searchNode, 
						double range,
						pbinData *outData, poldNode *leaf);

	void distributeToLeaves(pbinData inData, long rootNpoints,
						double dupFactor, poldNode searchNode, 
						pbinData *outData, poldNode *leaf);

	void distributeToNearestLeaf( pbinData inData,
								  poldNode searchNode, 
								  pbinData *outData, 
								  poldNode *leaf);

}


#endif




