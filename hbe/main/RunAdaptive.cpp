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
    double eps = atof(argv[3]);
    bool random = (argc > 4);

    parseConfig cfg(argv[1], scope);
    const double tau = cfg.getTau();
    // The dimensionality of each sample vector.
    int dim = cfg.getDim();
    // The number of sources which will be used for the gauss transform.
    int N = cfg.getN();
    int M = cfg.getM();

    double h = cfg.getH();
    if (!cfg.isConst()) {
        h *= pow(N, -1.0 / (dim + 4));
        if (strcmp(scope, "exp") != 0) {
            h *= sqrt(2);
        }
    }

    MatrixXd X = dataUtils::readFile(
            cfg.getDataFile(), cfg.ignoreHeader(), N, cfg.getStartCol(), cfg.getEndCol());
    auto band = make_unique<Bandwidth>(N, dim);
    std::cout << "bw: " << h << std::endl;
    band->useConstant(h);
    shared_ptr<Kernel> kernel;
    shared_ptr<Kernel> simpleKernel;
    if (strcmp(scope, "exp") == 0) {
        kernel = make_shared<Expkernel>(dim);
        simpleKernel = make_shared<Expkernel>(dim);
    } else {
        kernel = make_shared<Gaussiankernel>(dim);
        simpleKernel = make_shared<Gaussiankernel>(dim);
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
    bool sequential = (N == M);
    double *exact = new double[M * 2];
    dataUtils::readFile(cfg.getExactPath(), false, M, 0, 1, &exact[0]);

    shared_ptr<AdaptiveEstimator> est;
    std::cout << "eps = " << eps << std::endl;
    if (random) {
        std::cout << "RS" << std::endl;
        est = make_shared<AdaptiveRS>(X_ptr, simpleKernel, tau, eps);
        //est = make_shared<AdaptiveHBE>(X_ptr, simpleKernel, tau, eps, false);
    } else {
        std::cout << "HBE" << std::endl;
        auto t1 = std::chrono::high_resolution_clock::now();
        est = make_shared<AdaptiveHBE>(X_ptr, simpleKernel, tau, eps, true);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "Adaptive Table Init: " <<
                  std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() / 1000.0 << std::endl;
    }

    est->totalTime = 0;
    vector<double> results(2,0);

    for (int j = 0; j < M; j++) {
        int idx = j * 2;
        VectorXd q = X.row(j);
        if (hasQuery != 0) {
            q = Y.row(j);
        } else {
            if (!sequential) {
                q = X.row(exact[idx + 1]);
            }
        }
        vector<double> estimates = est->query(q);
        update(results, estimates, exact[idx]);
    }

    std::cout << "Sampling total time: " << est->totalTime / 1e9 << std::endl;
    std::cout << "Average Samples: " << results[1] / M << std::endl;
    std::cout << "Relative Error: " << results[0] / M << std::endl;

    delete[] exact;
}