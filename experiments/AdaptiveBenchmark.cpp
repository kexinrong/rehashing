//
// Created by Kexin Rong on 2018-11-10.
//

#include <stdio.h>
#include <stdlib.h>     /* atof */
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h> // for memset
#include <math.h>   // for abs
#include <algorithm>    // std::max
#include <chrono>
#include "expkernel.h"
#include "gaussiankernel.h"
#include "mathUtils.h"
#include "dataUtils.h"
#include "bandwidth.h"
#include "math.h"
#include "../alg/RS.h"
#include "../alg/AdaptiveRS.h"
#include <boost/math/distributions/normal.hpp>
#include "parseConfig.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Need config file" << std::endl;
        exit(1);
    }

    char *scope = argv[2];
    parseConfig cfg(argv[1], scope);
    const double eps = cfg.getEps();
    const double tau = cfg.getTau();
    const double beta = cfg.getBeta();
    // The dimensionality of each sample vector.
    int dim = cfg.getDim();
    // The number of sources which will be used for the gauss transform.
    int N = cfg.getN();
    int M = cfg.getM();

    double h = cfg.getH();
    if (!cfg.isConst()) {
        if (strcmp(scope, "exp") == 0) {
            h *= pow(N, -1.0 / (dim + 4));
        } else {
            h *= sqrt(2);
        }
    }

    MatrixXd X = dataUtils::readFile(
            cfg.getDataFile(), cfg.ignoreHeader(), N, cfg.getStartCol(), cfg.getEndCol());
    auto band = make_unique<Bandwidth>(N, dim);
    band->useConstant(h);
    shared_ptr<Kernel> kernel;
    shared_ptr<Kernel> simpleKernel;
    if (strcmp(scope, "gaussian") == 0) {
        kernel = make_shared<Gaussiankernel>(dim);
        simpleKernel = make_shared<Gaussiankernel>(dim);
    } else {
        kernel = make_shared<Expkernel>(dim);
        simpleKernel = make_shared<Expkernel>(dim);
    }
    double means = ceil(6 * simpleKernel->RelVar(tau) / eps / eps);

    kernel->initialize(band->bw);
//    dataUtils::checkBandwidthSamples(X, eps, kernel);
    // Normalized by bandwidth
    X = dataUtils::normalizeBandwidth(X, band->bw);
    shared_ptr<MatrixXd> X_ptr = make_shared<MatrixXd>(X);

    int hasQuery = strcmp(cfg.getDataFile(), cfg.getQueryFile());

    MatrixXd Y;
    if (hasQuery != 0) {
        Y = dataUtils::readFile(cfg.getQueryFile(),
                                cfg.ignoreHeader(), M, cfg.getStartCol(), cfg.getEndCol());
    }

    // Read exact KDE
    bool sequential = (argc > 3 || N == M);
    double *exact = new double[M * 2];
    if (sequential) {
        dataUtils::readFile(cfg.getExactPath(), false, M, 0, 0, &exact[0]);
    } else {
        dataUtils::readFile(cfg.getExactPath(), false, M, 0, 1, &exact[0]);
    }

    AdaptiveRS rs(X_ptr, simpleKernel, tau, eps);

    std::cout << "------------------" << std::endl;
    rs.totalTime = 0;
    double rs_error = 0;
    double rs_samples = 0;

    for (int j = 0; j < M; j++) {
        int idx = j;
        VectorXd q = X.row(j);
        if (hasQuery != 0) {
            q = Y.row(j);
        } else {
            if (!sequential) {
                idx = j * 2;
                q = X.row(exact[idx + 1]);
            }
        }
        vector<double> rs_est = rs.query(q);
        rs_error += fabs(rs_est[0] - exact[idx]) / exact[idx];
        rs_samples += rs_est[1];
    }

    std::cout << "RS Sampling total time: " << rs.totalTime / 1e9 << std::endl;
    std::cout << "RS Average Samples: " << rs_samples / M << std::endl;
    rs_error /= M;
    printf("RS relative error: %f\n", rs_error);

    delete[] exact;
}