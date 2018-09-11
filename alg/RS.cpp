//
// Created by Kexin Rong on 9/4/18.
//

#include "RS.h"
#include "mathUtils.h"
#include <iostream>

RS::RS(MatrixXd data, shared_ptr<Kernel> k) {
    X = data;
    kernel = k;
    numPoints = data.rows();
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    rng = std::mt19937_64(rd());
}

std::vector<double> RS::MoM(VectorXd q, int L, int m) {
    std::uniform_int_distribution<int> distribution(0, numPoints - 1);
    std::vector<double> Z = std::vector<double>(L, 0);
    for (int i = 0; i < L; i ++) {
        for (int j = 0; j < m; j ++) {
            int idx = distribution(rng);
            Z[i] += kernel->density(q, X.row(idx));
        }
    }
    return Z;
}
