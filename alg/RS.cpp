//
// Created by Kexin Rong on 9/4/18.
//

#include "RS.h"
#include "mathUtils.h"
#include <iostream>
#include <unordered_set>

RS::RS(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k) {
    X = data;
    kernel = k;
    numPoints = data->rows();
}

std::vector<double> RS::MoM(VectorXd q, int L, int m) {
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937_64 rng = std::mt19937_64(rd());
    std::uniform_int_distribution<int> distribution(0, numPoints - 1);

    std::vector<double> Z = std::vector<double>(L, 0);
    for (int i = 0; i < L; i ++) {
        std::vector<int> indices(m);
        for (int j = 0; j < m; j ++) {
            indices[j] = distribution(rng);
        }
        std::sort(indices.begin(), indices.end());
        for (int j = 0; j < m; j ++) {
            int idx = indices[j];
//            int idx = distribution(rng);
            Z[i] += kernel->density(q, X->row(idx));
        }
    }
    return Z;
}

std::unordered_set<int> pickSet(int N, int k, std::mt19937& gen) {
    std::unordered_set<int> elems;
    for (int r = N - k; r < N; ++r) {
        int v = std::uniform_int_distribution<>(0, r)(gen);

        // there are two cases.
        // v is not in candidates ==> add it
        // v is in candidates ==> well, r is definitely not, because
        // this is the first iteration in the loop that we could've
        // picked something that big.

        if (!elems.insert(v).second) {
            elems.insert(r);
        }
    }
    return elems;
}


//std::vector<double> RS::MoM(VectorXd q, int L, int m) {
//    // using built-in random generator:
//    std::random_device rd;
//    std::mt19937 gen(rd());
//    std::unordered_set<int> elems = pickSet(numPoints, m, gen);
//
//    std::vector<double> Z = std::vector<double>(L, 0);
//    for (int i = 0; i < L; i ++) {
//        for (int j: elems) {
//            Z[i] += kernel->density(q, X->row(j));
//        }
//    }
//    return Z;
//}
