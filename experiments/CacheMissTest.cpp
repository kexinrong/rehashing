//
// Created by Kexin Rong on 10/9/18.
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
    const double sample_ratio = cfg.getSampleRatio();
    int samples = cfg.getSamples();
    // The dimensionality of each sample vector.
    int dim = cfg.getDim();
    // The number of sources which will be used for the gauss transform.
    int N = cfg.getN();
    int M = 100000;
    // The bandwidth.  NOTE: this is not the same as standard deviation since
    // the Gauss Transform sums terms exp( -||x_i - y_j||^2 / h^2 ) as opposed
    // to  exp( -||x_i - y_j||^2 / (2*sigma^2) ).  Thus, if sigma is known,
    // bandwidth can be set to h = sqrt(2)*sigma.
    double h = cfg.getH();
    if (strcmp(scope, "exp") == 0) {
        h *= pow(N, -1.0/(dim+4));
    } else {
        h *= sqrt(2);
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
    //dataUtils::checkBandwidthSamples(X, eps, kernel);
    // Normalized by bandwidth
    X = dataUtils::normalizeBandwidth(X, band->bw);
    shared_ptr<MatrixXd> X_ptr = make_shared<MatrixXd>(X);
    // Algorithms init
    RS rs(X_ptr, simpleKernel);

    for (int i = 0; i < 10; i ++) {
        samples = int(100 * (i + 1) * sample_ratio);
        std::cout << "------------------" << std::endl;
        std::cout << "RS samples: " << samples << std::endl;
        double * g = new double[M];
        double * g_r = new double[M];
        rs.totalTime = 0;
        for(int j = 0; j < M; j++) {
            VectorXd q = X.row(j);
            g_r[j] = rs.query(q, tau, samples);
        }

        double sequential_time = 0;
        for (int j = 0; j < M; j ++) {
            VectorXd q = X.row(j);
            auto t1 = std::chrono::high_resolution_clock::now();
            for (int idx = 0; idx < samples; idx ++) {
                g[j] += simpleKernel->density(q, X.row(idx));
            }
            g[j] /= samples;
            auto t2 = std::chrono::high_resolution_clock::now();
            sequential_time += std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
        }
        std::cout << "RS Sampling total time: " << rs.totalTime / 1e9 << std::endl;
        std::cout << "Sequential Total Sampling total time: " << sequential_time / 1e9 << std::endl;

    }

}