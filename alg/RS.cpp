//
// Created by Kexin Rong on 9/4/18.
//

#include "RS.h"
#include "mathUtils.h"
#include <iostream>

RS::RS(MatrixXd data, Kernel *k) {
    X = data;
    kernel = k;
    numPoints = data.rows();
    srand (time(NULL));
}

double* RS::MoM(VectorXd q, int L, int m) {
    double* Z = new double[L];
    std::memset(Z, 0, L);
    for (int i = 0; i < L; i ++) {
        for (int j = 0; j < m; j ++) {
            int idx = rand() % numPoints;
            Z[i] += kernel->density(q, X.row(idx));
        }
    }
    return Z;
}
