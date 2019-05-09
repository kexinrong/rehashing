#ifndef _KNNREDUCE_HPP_
#define _KNNREDUCE_HPP_
#include <mpi.h>
#include <map>
#include <iostream>
#include <omp.h>
#include <cassert>
#include <limits>
#include <vector>
#include <algorithm>

using namespace std;

typedef std::pair<double,long> dist_t;

namespace knn {

        void query_r(MPI_Comm comm, double R, dist_t * arr, long nq, long len,
                        dist_t *& retarr, long *& retlen);

        void query_k(MPI_Comm comm, int K, int root, dist_t * arr, int nq,
                        long len, dist_t *& retarr);
}

#endif // _KNNREDUCE_HPP_
