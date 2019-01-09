//
// Created by Kexin Rong on 12/31/18.
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
#include "../alg/AdaptiveRSDiag.h"
#include "../alg/AdaptiveRS.h"
#include <boost/math/distributions/normal.hpp>
#include "parseConfig.h"

double getAvg(vector<double>& results) {
    double sum = 0;
    for (auto& n : results) { sum += n; }
    return sum / results.size();
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
//    std::cout << "bw: " << h << std::endl;

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

    AdaptiveRSDiag rs(X_ptr, simpleKernel, tau, eps);
    rs.setMedians(7);

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> distribution(0, M - 1);


    auto t1 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 3; iter ++) {
        vector<double> rs_cost;
        vector<double> hbe_cost;
        int j = 0;
        while (j < 50) {
            int idx = distribution(rng);
            VectorXd q = X.row(idx);
            if (hasQuery != 0) {
                q = Y.row(j);
            }
            rs.clearSamples();
            vector<double> rs_est = rs.query(q);
            if (rs_est[0] < tau) { continue; }
            double r2 = max(rs_est[0], tau);
            r2 *= r2;

            int actual = rs.findActualLevel(q, rs_est[0], eps);
            rs.getConstants();
            rs.findRings(1, 0.5, q, actual);
//            std::cout << rs.lambda << "," << rs.l << std::endl;
            j ++;
            rs_cost.push_back(rs.RSDirect(rs_est[0]) / r2);
            hbe_cost.push_back(rs.HBEDirect() / r2);
        }
        std::cout << "rs:" << getAvg(rs_cost) << ", hbe: " <<  getAvg(hbe_cost) << std::endl;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Diagnosis took: " <<
              std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << "ms" << std::endl;

}