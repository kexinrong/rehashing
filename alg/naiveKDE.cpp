//
// Created by Kexin Rong on 9/4/18.
//

#include "naiveKDE.h"
#include <iostream>

naiveKDE::naiveKDE(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k) {
    X = data;
    kernel = k;
    numPoints = data->rows();
}

double naiveKDE::query(VectorXd q) {
    double kde = 0;
    for (int i = 0; i < numPoints; i++) {
        kde += kernel->density(q, X->row(i));
    }
    kde /= numPoints;
    return kde;
}



