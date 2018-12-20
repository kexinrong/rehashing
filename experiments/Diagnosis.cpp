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

    AdaptiveRS rs(X_ptr, simpleKernel, tau, eps);
    rs.setMedians(7);

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> distribution(0, M - 1);

    for (int iter = 0; iter < 3; iter ++) {
        double reals = 0;
        double level = 0;
        double rs_cost = 0;
        double hbe_cost = 0;
        for (int j = 0; j < 100; j++) {
            int idx = distribution(rng);
            VectorXd q = X.row(idx);
            if (hasQuery != 0) {
                q = Y.row(j);
            }
            vector<double> rs_est = rs.query(q);

            int target = rs.findTargetLevel(rs_est[0]);

            int actual = rs.findActualLevel(q, rs_est[0], eps);
            reals += actual;
            level += target;

            rs_cost += rs.findRSRatio(rs_est[0], eps);
            hbe_cost += rs.findHBERatio(q, actual, rs_est[0], eps);
        }
        //std::cout << "actual:" << reals/100 << ", target: " << level/100 << std::endl;
        std::cout << "rs:" << rs_cost/100 << ", hbe: " << hbe_cost/100 << std::endl;

    }

}