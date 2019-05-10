/* This program finds the smallest epsilon for random sampling and HBE that achieves a true relative error < 0.1.
 * Epsilon is a parameter that controls error in the adaptive sampling procedure.
 */

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

void findEps(bool isRandom, shared_ptr<MatrixXd> X, MatrixXd &Y, double *exact, parseConfig cfg,
        bool sequential, bool hasQuery) {
    double eps = 0.6;
    bool stop = false;
    bool times = false;

    double tau = cfg.getTau();
    int M = cfg.getM();
    int dim = cfg.getDim();
    shared_ptr<Kernel> simpleKernel;
    double head = 0;
    double tail = 1;
    if (strcmp(cfg.scope, "exp") == 0) {
        simpleKernel = make_shared<Expkernel>(dim);
    } else {
        simpleKernel = make_shared<Gaussiankernel>(dim);
    }

    shared_ptr<AdaptiveEstimator> est;
    while (!stop) {
        std::cout << "------------------" << std::endl;
        std::cout << "eps = " << eps << std::endl;
        if (isRandom) {
            est = make_shared<AdaptiveRS>(X, simpleKernel, tau, eps);
        } else {
            auto t1 = std::chrono::high_resolution_clock::now();
            est = make_shared<AdaptiveHBE>(X, simpleKernel, tau, eps, true);
            auto t2 = std::chrono::high_resolution_clock::now();
            std::cout << "Adaptive Table Init: " <<
                      std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() / 1000.0  << std::endl;
        }
        est->totalTime = 0;
        vector<double> results(2,0);

        for (int j = 0; j < M; j++) {
            int idx = j * 2;
            VectorXd q = X->row(j);
            if (hasQuery != 0) {
                q = Y.row(j);
                idx = j;
            } else {
                if (!sequential) {
                    q = X->row(int(exact[idx + 1]));
                }
            }

            vector<double> vals = est->query(q);
            update(results, vals, exact[idx]);
        }

        std::cout << "Sampling total time: " << est->totalTime / 1e9 << std::endl;
        std::cout << "Average Samples: " << results[1] / M << std::endl;
        std::cout << "Relative Error: " << results[0] / M << std::endl;

        double err = results[0] / M;
        if (eps == 0.6) {
            if (err < 0.1) {
                times = true;
                eps *= 2;
            } else {
                eps /= 2;
            }
        } else {
            if (times) {
                if (err > 0.1) {
                    stop = true;
                    head = eps / 2;
                    tail = eps;
                    break;
                }
                eps *= 2;
            } else {
                if (err < 0.1) {
                    stop = true;
                    head = eps;
                    tail = eps * 2;
                    break;
                }
                eps /= 2;
            }
        }
    }

    std::cout << "Binary search: ["<< head << "," << tail << "]\n";
    while (true) {
        eps = (head + tail) / 2;

        std::cout << "------------------" << std::endl;
        std::cout << "eps = " << eps << std::endl;
        if (isRandom) {
            est = make_shared<AdaptiveRS>(X, simpleKernel, tau, eps);
        } else {
            auto t1 = std::chrono::high_resolution_clock::now();
            est = make_shared<AdaptiveHBE>(X, simpleKernel, tau, eps, true);
            auto t2 = std::chrono::high_resolution_clock::now();
            std::cout << "Adaptive Table Init: " <<
                      std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count() << std::endl;
        }
        est->totalTime = 0;
        vector<double> results(2,0);

        for (int j = 0; j < M; j++) {
            int idx = j * 2;
            VectorXd q = X->row(j);
            if (hasQuery != 0) {
                q = Y.row(j);
                idx = j;
            } else {
                if (!sequential) {
                    q = X->row(int(exact[idx + 1]));
                }
            }
            vector<double> vals = est->query(q);
            update(results, vals, exact[idx]);
        }

        std::cout << "Sampling total time: " << est->totalTime / 1e9 << std::endl;
        std::cout << "Average Samples: " << results[1] / M << std::endl;
        std::cout << "Relative Error: " << results[0] / M << std::endl;

        double err = results[0] / M;
        if (err < 0.11 && err > 0.09) {
            break;
        } else if (err > 0.1) {
            tail = eps;
        } else {
            head = eps;
        }
    }

}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Need config file" << std::endl;
        exit(1);
    }

    char *scope = argv[2];
    parseConfig cfg(argv[1], scope);
    // The dimensionality of each sample vector.
    int dim = cfg.getDim();
    // The number of sources which will be used for the gauss transform.
    int N = cfg.getN();
    int M = cfg.getM();

    double h = cfg.getH();
    const char* kernel_type = cfg.getKernel();

    MatrixXd X = dataUtils::readFile(
            cfg.getDataFile(), cfg.ignoreHeader(), N, cfg.getStartCol(), cfg.getEndCol());
    auto band = make_unique<Bandwidth>(N, dim);
    std::cout << "bw: " << h << std::endl;
    band->useConstant(h);
    shared_ptr<Kernel> kernel;
    if (strcmp(kernel_type, "exp") == 0) {
        kernel = make_shared<Expkernel>(dim);
    } else {
        kernel = make_shared<Gaussiankernel>(dim);
    }

    kernel->initialize(band->bw);
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
    if (hasQuery != 0) {
        dataUtils::readFile(cfg.getExactPath(), false, M, 0, 0, &exact[0]);
    } else {
        dataUtils::readFile(cfg.getExactPath(), false, M, 0, 1, &exact[0]);
    }

    std::cout << "RS\n";
    findEps(true, X_ptr, Y, exact, cfg, sequential, hasQuery);

    std::cout << "======================================\n";

    std::cout << "HBE\n";
    findEps(false, X_ptr, Y, exact, cfg, sequential, hasQuery);


    delete[] exact;
}


