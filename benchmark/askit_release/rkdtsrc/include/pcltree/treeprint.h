#ifndef _TREEPRINT_H__
#define _TREEPRINT_H__

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>

#include "mpitree.h"

using namespace std;

ostream& printarr(double *arr, int n);
void treePrint(pMTNode in_node);
void treeSave(ofstream &outifle, pMTNode in_node);
void treeSaveRadii(ofstream &outfile, pMTNode in_node);

#endif
