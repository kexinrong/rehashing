/*
 *  Initialize dataset and pararmeters from config files.
 */

#ifndef REHASHING_DATAINGEST_H
#define REHASHING_DATAINGEST_H

#include "parseConfig.h"
#include "kernel.h"
#include "expkernel.h"
#include "gaussiankernel.h"
#include "mathUtils.h"
#include "dataUtils.h"
#include "bandwidth.h"
#include <Eigen/Dense>

using Eigen::MatrixXd;


class DataIngest {
public:
    int N, M, dim, hasQuery, k;
    double tau, eps, beta, sample_ratio, h, diam, w;
    shared_ptr<MatrixXd> X_ptr, Y_ptr;
    shared_ptr<Kernel> kernel;
    bool sequential;
    double *exact;

    DataIngest(parseConfig cfg, bool read_exact) {
        try {
            eps = cfg.getEps();
        } catch (...) {
            eps = 0.5;
        }
        try {
            beta = cfg.getBeta();
        } catch (...) {
            beta = 0.5;
        }
        try {
            sample_ratio = cfg.getSampleRatio();
        } catch (...) {
            sample_ratio = 1;
        }
        // Minimum density
        tau = cfg.getTau();
        // The dimensionality of each sample vector.
        dim = cfg.getDim();
        // The number of source datapoints
        N = cfg.getN();
        // The number of querie datapoints
        M = cfg.getM();
        sequential = (N == M);


        // Read source dataset
        MatrixXd X = dataUtils::readFile(
                cfg.getDataFile(), cfg.ignoreHeader(), N, cfg.getStartCol(), cfg.getEndCol());

        // Read query dataset if it's a in a seperate file
        hasQuery = strcmp(cfg.getDataFile(), cfg.getQueryFile());
        if (hasQuery != 0) {
            Y_ptr = make_shared<MatrixXd>(dataUtils::readFile(cfg.getQueryFile(),
                                                          cfg.ignoreHeader(), M, cfg.getStartCol(), cfg.getEndCol()));
        }

        // Get bandwidth
        h = cfg.getH();
        const char* kernel_type = cfg.getKernel();
        // If input bandwidth is not constant
        if (!cfg.isConst()) {
            // Scott's rule
            h *= pow(N, -1.0 / (dim + 4));
            // Gaussian Kernel:  1/(2h^2)
            if (strcmp(kernel_type, "gaussian") == 0) {
                h *= sqrt(2);
            }
        }
        std::cout << "bandwidth: " << h << std::endl;
        auto band = make_unique<Bandwidth>(N, dim);
        band->useConstant(h);
        if (strcmp(kernel_type, "gaussian") == 0) {
            kernel = make_shared<Gaussiankernel>(dim);
        } else {
            kernel = make_shared<Expkernel>(dim);
        }

        // Normalized by bandwidth
        // dataUtils::checkBandwidthSamples(X, eps, kernel);
        X = dataUtils::normalizeBandwidth(X, band->bw);
        X_ptr = make_shared<MatrixXd>(X);

        // Whether or not to read ground truth KDE
        if (read_exact) {
            exact = new double[M * 2];
            dataUtils::readFile(cfg.getExactPath(), false, M, 0, 1, &exact[0]);
        }
    }

    void estimateHashParams() {
        diam = dataUtils::estimateDiameter(X_ptr, tau);
        k = dataUtils::getPower(diam, beta);
        w = dataUtils::getWidth(k, beta);
    }

    ~DataIngest() {
        delete[] exact;
    }

};


#endif //REHASHING_DATAINGEST_H
