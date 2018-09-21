//------------------------------------------------------------------------------
// The code was written by Vlad I. Morariu
// and is copyrighted under the Lesser GPL:
//
// Copyright (C) 2007 Vlad I. Morariu
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; version 2.1 or later.
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston,
// MA 02111-1307, USA.
//
// The author may be contacted via email at: morariu(at)cs(.)umd(.)edu
//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
// File    : sample.cpp
// Purpose : Show example of how figtree() function can be used to evaluate
//           gauss transforms.
// Author  : Vlad I. Morariu       morariu(at)cs(.)umd(.)edu
// Date    : 2007-06-25
// Modified: 2008-01-23 to add comments and make the sample clearer
// Modified: 2010-05-12 to add the automatic method selection function call
//             example.
//------------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>     /* atof */
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h> // for memset
#include <math.h>   // for abs
#include "figtree.h"
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include <algorithm>    // std::max

// This file only shows examples of how the figtree() function can be used.
//
// See source code of figtree() in figtree.cpp for sample code of how
// the functions figtreeEvaluate*(), figtreeChooseParameters*(), and figtreeKCenterClustering()
// can be used to evaluate gauss transforms.  Calling them directly instead of the
// wrapper function, figtree(), might be useful if the same sources are always used so
// the parameter selection and clustering steps do not have to be performed each time
// the gauss transform needs to be evaluated.

void readFile(std::string filename, bool ignoreHeader, int n, int startCol, int endCol, double *data) {
    std::ifstream infile(filename.c_str());

    int dim = endCol - startCol + 1;
    bool firstRow = true;
    int i = 0;

    std::string line;
    std::string delim = ",";
    while (std::getline(infile, line)) {
        if (ignoreHeader && i == 0) {
            i += 1;
            continue;
        }
        size_t start = 0;
        size_t end = line.find(delim);
        int j = 0;
        while (end != std::string::npos && j <= endCol) {
            if (j >= startCol) {
                if (ignoreHeader) {
                    data[(i-1) * dim + j - startCol] = atof(line.substr(start, end - start).c_str());
                } else {
                    data[i * dim + j - startCol] = atof(line.substr(start, end - start).c_str());
                }
            }
            start = end + delim.length();
            end = line.find(delim, start);
            j += 1;
        }
        i += 1;
        if (ignoreHeader && i == n + 1) {
            break;
        } else if (!ignoreHeader && i == n) {
            break;
        }
    }
    infile.close();
}

void fitCube(double* data, int n, int d) {
    for (int j = 0; j < d; j++) {
        double mmin = data[j];
        double mmax = data[j];
        for (int i = 1; i < n; i ++) {
            mmin = std::min(data[i * d + j], mmin);
            mmax = std::max(data[i * d + j], mmax);
        }
        double range = mmax - mmin;
        for (int i = 0; i < n; i ++) {
            data[i * d + j] = (data[i * d + j] - mmin) / range;
        }
    }
}


int main()
{
    // The dimensionality of each sample vector.
    int d = 10;

    // The number of targets (vectors at which gauss transform is evaluated).
    int M = 50000;

    // The number of sources which will be used for the gauss transform.
    int N = 50000;

    // The bandwidth.  NOTE: this is not the same as standard deviation since
    // the Gauss Transform sums terms exp( -||x_i - y_j||^2 / h^2 ) as opposed
    // to  exp( -||x_i - y_j||^2 / (2*sigma^2) ).  Thus, if sigma is known,
    // bandwidth can be set to h = sqrt(2)*sigma.
    double h = .0154758;

    // Desired maximum absolute error after normalizing output by sum of weights.
    // If the weights, q_i (see below), add up to 1, then this is will be the
    // maximum absolute error.
    // The smaller epsilon is, the more accurate the results will be, at the
    // expense of increased computational complexity.
    double epsilon = 1e-2;

    // The source array.  It is a contiguous array, where
    // ( x[i*d], x[i*d+1], ..., x[i*d+d-1] ) is the ith d-dimensional sample.
    // For example, below N = 20 and d = 7, so there are 20 rows, each
    // a 7-dimensional sample.
    double x[N * d];
    readFile("../../resources/covtype.data", false, N, 0, 9, &x[0]);
    fitCube(&x[0], N, d);

    // The target array.  It is a contiguous array, where
    // ( y[j*d], y[j*d+1], ..., y[j*d+d-1] ) is the jth d-dimensional sample.
    // For example, below M = 10 and d = 7, so there are 10 rows, each
    // a 7-dimensional sample.
    double y[M * d];
    for (size_t i = 0; i < M * d; i ++) { y[i] = x[i]; }

    // The weight array.  The ith weight is associated with the ith source sample.
    // To evaluate the Gauss Transform with the same sources and targets, but
    // different sets of weights, add another row of weights and set W = 2.
    double *q = new double[N];
    for (size_t i = 0; i < N; i ++) { q[i] = 1; }
    //memset(q, 1, sizeof(double) * N );

    // Number of weights.  For each set of weights a different Gauss Transform is computed,
    // but by giving multiple sets of weights at once some overhead can be shared.
    int W = 1;  // in this case W = 1.

    // allocate array into which the result of the Gauss Transform will be stored for each
    // target sample.  The first M elements will correspond to the Gauss Transform computed
    // with the first set of weights, second M elements will correspond to the G.T. computed
    // with the second set of weights, etc.ha
    double * g_auto = new double[W*M];
    double * g_sf = new double[W*M];


    // initialize all output arrays to zero
    memset( g_auto        , 0, sizeof(double)*W*M );
    memset( g_sf          , 0, sizeof(double)*W*M );


    //
    // RECOMMENDED way to call figtree().
    //
    // Evaluates the Gauss transform using automatic method selection (the automatic
    // method selection function analyzes the inputs -- including source/target
    // points, weights, bandwidth, and error tolerance -- to automatically choose
    // between FIGTREE_EVAL_DIRECT, FIGTREE_EVAL_DIRECT_TREE, FIGTREE_EVAL_IFGT,
    // FIGTREE_EVAL_IFGT_TREE.
    // This function call makes use of the default parameters for the eval method
    // and param method, and is equivalent to
    // figtree( d, N, M, W, x, h, q, y, epsilon, g_auto, FIGTREE_EVAL_AUTO, FIGTREE_PARAM_NON_UNIFORM ).
    clock_t t1 = clock();
    figtree( d, N, M, W, x, h, q, y, epsilon, g_auto );
    clock_t t2 = clock();
    std::cout << "FIGTREE auto: " << (float)(t2 - t1)/CLOCKS_PER_SEC << std::endl;

    //
    // MANUAL EVALUATION METHOD and PARAMETER METHOD selection.  If chosen
    // incorrectly, this could cause run times to be several orders of
    // magnitudes longer or to require several orders of magnitude more
    // memory (resulting in crashes).  The recommended way to call figtree
    // is using the automatic method selection, as shown above.
    //
    // evaluate gauss transform using direct (slow) method
    figtree( d, N, M, W, x, h, q, y, epsilon, g_sf, FIGTREE_EVAL_DIRECT );

    // compute absolute error of the Gauss Transform at each target and for all sets of weights.
    double avg_error = 0;
    double avg_relerror = 0;
    for( int i = 0; i < W*M; i++) {
       avg_error += fabs(g_auto[i] - g_sf[i]);
        avg_relerror +=  fabs(g_auto[i] - g_sf[i]) / g_sf[i];
        g_auto[i]         = fabs(g_auto[i]        -g_sf[i]);
    }
    avg_error /= W*M;
    avg_relerror /= W*M;
    printf("Average absolute error: %f\n", avg_error);
    printf("Average relative error: %f\n", avg_relerror);

    // print out results for all six ways to evaluate
//    printf("Results:\n");
//    printf("Direct result | auto error\n");
//    for( int i = 0; i < W*M; i++ )
//        printf("%14.4f %14.4e\n", g_sf[i], g_auto[i]);
////        printf("%13.4f %13.4e %13.4e %13.4e %13.4e %13.4e %13.4e\n",
////               g_sf[i], g_auto[i], g_sf_tree[i], g_ifgt_u[i], g_ifgt_nu[i], g_ifgt_tree_u[i], g_ifgt_tree_nu[i] );
//    printf("\n");

    // deallocate memory
    delete [] q;
    delete [] g_auto;
    delete [] g_sf;
    return 0;
}
