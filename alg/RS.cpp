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
}

std::vector<double> RS::MoM(VectorXd q, int L, int m) {
    std::vector<double> Z = std::vector<double>(L, 0);
    for (int i = 0; i < L; i ++) {
        for (int j = 0; j < m; j ++) {
            int idx = rand() % numPoints;
            Z[i] += kernel->density(q, X.row(idx));
        }
    }
    return Z;
}
