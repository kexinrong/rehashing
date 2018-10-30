//
// Created by Kexin Rong on 9/30/18.
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
#include "../alg/naiveKDE.h"
#include "../alg/BaseLSH.h"
#include "../alg/SketchLSH.h"
#include <boost/math/distributions/normal.hpp>
#include "parseConfig.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Need config file" << std::endl;
        exit(1);
    }

    char* scope = argv[2];
    parseConfig cfg(argv[1], scope);
    const double eps = cfg.getEps();
    const double tau = cfg.getTau();
    const double beta = cfg.getBeta();
    const double sample_ratio = cfg.getSampleRatio();
    int samples = cfg.getSamples();
    // The dimensionality of each sample vector.
    int dim = cfg.getDim();
    // The number of sources which will be used for the gauss transform.
    int N = cfg.getN();
    int M = cfg.getM();
    // The bandwidth.  NOTE: this is not the same as standard deviation since
    // the Gauss Transform sums terms exp( -||x_i - y_j||^2 / h^2 ) as opposed
    // to  exp( -||x_i - y_j||^2 / (2*sigma^2) ).  Thus, if sigma is known,
    // bandwidth can be set to h = sqrt(2)*sigma.
    double h = cfg.getH();
    if (!cfg.isConst()) {
        if (strcmp(scope, "exp") == 0) {
            h *= pow(N, -1.0/(dim+4));
        } else {
            h *= sqrt(2);
        }
    }

    MatrixXd X = dataUtils::readFile(
            cfg.getDataFile(), cfg.ignoreHeader(), N, cfg.getStartCol(), cfg.getEndCol());
    auto band = make_unique<Bandwidth>(N, dim);
    band->useConstant(h);
    shared_ptr<Kernel> kernel;
    double means = 0;
    shared_ptr<Kernel> simpleKernel;
    if (strcmp(scope, "gaussian") == 0) {
        kernel = make_shared<Gaussiankernel>(dim);
        simpleKernel = make_shared<Gaussiankernel>(dim);
        means = ceil(6 * mathUtils::gaussRelVar(tau) / eps / eps);
    } else {
        kernel = make_shared<Expkernel>(dim);
        simpleKernel = make_shared<Expkernel>(dim);
        means = ceil(6 * mathUtils::expRelVar(tau) / eps / eps);
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

    // Estimate parameters
    int tables = min((int)(means * 1.1), 1100);
    double diam = dataUtils::estimateDiameter(X, tau);
    int k = dataUtils::getPower(diam, beta);
    double w = dataUtils::getWidth(k, beta);

    // Algorithms init
    int subsample = int(sqrt(N));
    std::cout << "M=" << tables << ",w=" << w << ",k=" << k << ",samples=" << subsample << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    BaseLSH hbe(X_ptr, tables, w, k, 1, simpleKernel, subsample);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Uniform Sample Table Init: " << std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count() << std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    SketchLSH sketch(X_ptr, tables, w, k, 1, simpleKernel);
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Sketch Table Init: " << std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count() << std::endl;

    RS rs(X_ptr, simpleKernel);

    for (int i = 0; i < 10; i ++) {
        samples = 100 * (i + 1);
        std::cout << "------------------" << std::endl;
        std::cout << "HBE samples: " << samples << ", RS samples: " << int(samples * sample_ratio) << std::endl;
        hbe.totalTime = 0;
        rs.totalTime = 0;
        sketch.totalTime = 0;
        double hbe_error = 0;
        double sketch_error = 0;
        double rs_error = 0;
        for(int j = 0; j < M; j++) {
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

            double hbe_est = hbe.query(q, tau, samples);
            double sketch_est = sketch.query(q, tau, samples);
            double rs_est = rs.query(q, tau, int(samples * sample_ratio));
            hbe_error +=  fabs(hbe_est - exact[idx]) / exact[idx];
            rs_error += fabs(rs_est - exact[idx]) / exact[idx];
            sketch_error += fabs(sketch_est - exact[idx]) / exact[idx];

//            std::cout << exact[idx] << "," << hbe_est << "," << sketch_est << "," << rs_est << std::endl;
        }
        std::cout << "HBE Sampling total time: " << hbe.totalTime / 1e9 << std::endl;
        std::cout << "Sketch HBE total time: " << sketch.totalTime / 1e9 << std::endl;
        std::cout << "RS Sampling total time: " << rs.totalTime / 1e9 << std::endl;
        hbe_error /= M;
        rs_error /= M;
        sketch_error /= M;
        printf("HBE relative error: %f\n", hbe_error);
        printf("Sketch HBE relative error: %f\n", sketch_error);
        printf("RS relative error: %f\n", rs_error);
    }

    delete exact;
}