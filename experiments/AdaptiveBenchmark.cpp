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
#include "../alg/AdaptiveHBE.h"
#include "parseConfig.h"

void update(vector<double>& results, vector<double> est, double exact) {
    results[0] += fabs(est[0] - exact) / exact;
    results[1] += est[1];
}

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

    auto t1 = std::chrono::high_resolution_clock::now();
    AdaptiveHBE hbe(X_ptr, simpleKernel, tau, eps);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Adaptive Table Init: " <<
        std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count() << std::endl;

    std::cout << "------------------" << std::endl;
    rs.totalTime = 0;
    hbe.totalTime = 0;
    vector<double> rs_results(2,0);
    vector<double> hbe_results(2,0);

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
        vector<double> hbe_est = hbe.query(q);
        update(rs_results, rs_est, exact[idx]);
        update(hbe_results, hbe_est, exact[idx]);
    }

    std::cout << "RS Sampling total time: " << rs.totalTime / 1e9 << std::endl;
    std::cout << "RS Average Samples: " << rs_results[1] / M << std::endl;
    std::cout << "RS Relative Error: " << rs_results[0] / M << std::endl;

    std::cout << "HBE Sampling total time: " << hbe.totalTime / 1e9 << std::endl;
    std::cout << "HBE Average Samples: " << hbe_results[1] / M << std::endl;
    std::cout << "HBE Relative Error: " << hbe_results[0] / M << std::endl;

    delete[] exact;
}