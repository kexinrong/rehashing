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
    const int samples = cfg.getSamples();
    // The dimensionality of each sample vector.
    int dim = cfg.getDim();
    // The number of sources which will be used for the gauss transform.
    int N = cfg.getN();
    int M = N;
    // The bandwidth.  NOTE: this is not the same as standard deviation since
    // the Gauss Transform sums terms exp( -||x_i - y_j||^2 / h^2 ) as opposed
    // to  exp( -||x_i - y_j||^2 / (2*sigma^2) ).  Thus, if sigma is known,
    // bandwidth can be set to h = sqrt(2)*sigma.
    double h = cfg.getH();
    if (strcmp(scope, "exp") == 0) {
        h *= pow(N, -1.0/(dim+4));
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
    dataUtils::checkBandwidthSamples(X, eps, kernel);
    // Normalized by bandwidth
    X = dataUtils::normalizeBandwidth(X, band->bw);
    shared_ptr<MatrixXd> X_ptr = make_shared<MatrixXd>(X);

    // Estimate parameters
    int tables = (int)(means * 1.1);
    double diam = dataUtils::estimateDiameter(X, tau);
    int k = dataUtils::getPower(diam, beta);
    double w = dataUtils::getWidth(k, beta);

    // Algorithms init
    std::cout << "M=" << tables << ",w=" << w << ",k=" << k << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    BaseLSH hbe(X_ptr, tables, w, k, 1, simpleKernel, 1);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "HBE Table Init: " << std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count() << std::endl;
    RS rs(X_ptr, simpleKernel);

    double * g = new double[M];
    double * g_r = new double[M];
    for(int j = 0; j < M; j++) {
        VectorXd q = X.row(j);
        g[j] = hbe.query(q, tau, samples);
        g_r[j] = rs.query(q, tau, int(samples * sample_ratio));
    }
    std::cout << "HBE Sampling total time: " << hbe.totalTime / 1e9 << std::endl;
    std::cout << "RS Sampling total time: " << rs.totalTime / 1e9 << std::endl;

    double exact[M];
    dataUtils::readFile(cfg.getExactPath(), false, M, 0, 0, &exact[0]);
    double hbe_error = 0;
    double rs_error = 0;
    for (int i = 0; i < M; i ++) {
        hbe_error +=  fabs(g[i] - exact[i]) / exact[i];
        rs_error += fabs(g_r[i] - exact[i]) / exact[i];
    }
    hbe_error /= M;
    rs_error /= M;
    printf("HBE relative error: %f\n", hbe_error);
    printf("RS relative error: %f\n", rs_error);
}