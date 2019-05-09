#ifndef _GATHERTREE_H_
#define _GATHERTREE_H_

#include <mpi.h>
#include <vector>
#include "oldTree.h"
#include "stTree.h"

using namespace std;

namespace oldtree {

void gatherTree(poldNode root, vector<pair<int, double> > & treearr);

void plantTree(vector< pair<int, double> >& ArrTree,
			   int depth, pstNode inParent, pstNode inNode);

void destroyTree(pstNode inNode);

void printTree(pstNode inNode);

void visitTree(double *points, int numof_points, int dim,
			   vector<int> & member_ids, pstNode inNode, int depth,
			   int * point_to_visual_leaf_membership,
			   int * count);

void repartitionQueryData(double *points, long * ids, int numof_points, int dim, poldNode inNode,
						  double **new_X, long ** new_gids, long &new_N);


}

#endif
