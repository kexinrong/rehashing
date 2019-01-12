//
// Created by Kexin Rong on 1/10/19.
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
#include "../alg/MRSketch.h"
#include "../alg/Herding.h"
#include "../alg/KCenter.h"
#include <boost/math/distributions/normal.hpp>
#include "parseConfig.h"

int nsamples[] = {8, 13, 23, 39, 68, 116, 199, 341, 584, 1000};
//int nsamples[] = { 584, 1000, 5000, 10000};

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Need config file" << std::endl;
        exit(1);
    }

    char* scope = argv[2];
    //int m = atoi(argv[3]);

    parseConfig cfg(argv[1], scope);
    const char* name = cfg.getName();
    const double tau = cfg.getTau();
    const double beta = cfg.getBeta();
    // The dimensionality of each sample vector.
    int dim = cfg.getDim();
    // The number of sources which will be used for the gauss transform.
    int N = cfg.getN();
    int M = cfg.getM();

    double h = cfg.getH();
    if (!cfg.isConst()) {
        h *= pow(N, -1.0/(dim+4));
        if (strcmp(scope, "exp") != 0) {
            h *= sqrt(2);
        }
    }
    std::cout << "dataset: " << name << std::endl;
    std::cout << "bw: " << h << std::endl;
    std::cout << "----------------------------" << std::endl;

    MatrixXd X = dataUtils::readFile(
            cfg.getDataFile(), cfg.ignoreHeader(), N, cfg.getStartCol(), cfg.getEndCol());
    auto band = make_unique<Bandwidth>(N, dim);
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

    // Estimate parameters
    double diam = dataUtils::estimateDiameter(X, tau);
    int k = dataUtils::getPower(diam, beta);
    double w = dataUtils::getWidth(k, beta);

    // Read exact KDE
    double *exact = new double[M * 2];
    dataUtils::readFile(cfg.getExactPath(), false, M, 0, 1, &exact[0]);

    std::uniform_int_distribution<int> distribution(0, M - 1);
    //Will be used to obtain a seed for the random number engine
    std::random_device rd;
    std::mt19937_64 rng(rd());


    for (size_t idx = 0; idx < sizeof(nsamples)/sizeof(nsamples[0]); idx ++) {
        idx = 5;
        int m = nsamples[idx];
        std::cout << "k=" << m << std::endl;
        for (size_t iter = 0; iter < 10; iter ++) {
            RS rs(X_ptr, simpleKernel, m);
            MRSketch hbs = MRSketch(X_ptr, m, w, k, tau);
            MRSketch hbs_simple = MRSketch(X_ptr, m, w, k, 5);
            Herding herding = Herding(X_ptr, simpleKernel, m);
            KCenter kcenter = KCenter(X_ptr, simpleKernel, m, 1);

            // Evaluate errors on random samples
            auto& samples = hbs.final_samples;
            auto& hbs_samples = hbs_simple.final_samples;
            auto& h_samples = herding.samples;
            auto& hc_samples = kcenter.center_samples;
            auto& hc_rs_samples = kcenter.rs_samples;

            vector<double> err (5,0);
            int N_EVAL = 200;
            for (int i = 0; i < N_EVAL; i ++) {
                int idx = distribution(rng);
                double exact_val = exact[idx * 2];
                double query_idx = exact[idx * 2 + 1];
                VectorXd q = X.row(query_idx);

                // HBS
                double est = 0;
                for (size_t j = 0; j < samples.size(); j ++) {
                    est += samples[j].second * simpleKernel->density(q, X.row(samples[j].first));
                }
                err[0] += pow((est - exact_val) / max(tau, exact_val), 2);

                est = 0;
                for (size_t j = 0; j < hbs_samples.size(); j ++) {
                    est += hbs_samples[j].second * simpleKernel->density(q, X.row(hbs_samples[j].first));
                }
                err[3] += pow((est - exact_val) / max(tau, exact_val), 2);

                // Uniform
                double rs_est = rs.query(q, tau, m);
                err[1] += pow((rs_est - exact_val) / max(tau, exact_val), 2);

                // Herding
                est = 0;
                for (size_t j = 0; j < h_samples.size(); j ++) {
                    est += h_samples[j].second * simpleKernel->density(q, X.row(h_samples[j].first));
                }
                err[2] += pow((est - exact_val) / max(tau, exact_val), 2);

                // KCenter
                est = 0;
                for (size_t j = 0; j < hc_samples.size(); j ++) {
                    est += hc_samples[j].second * simpleKernel->density(q, X.row(hc_samples[j].first));
                }
                if (hc_rs_samples.size() > 0) {
                    double est1 = 0;
                    for (size_t j = 0; j < hc_rs_samples.size(); j ++) {
                        est1 += hc_rs_samples[j].second * simpleKernel->density(q, X.row(hc_rs_samples[j].first));
                    }
                    est = est / kcenter.kc + est1 * (1 - 1/kcenter.kc);
                }
                err[4] += pow((est - exact_val) / max(tau, exact_val), 2);

            }

            std::cout << "HBE: " << sqrt(err[0]/N_EVAL) << std::endl;
            std::cout << "HBE (single): " << sqrt(err[3]/N_EVAL) << std::endl;
            std::cout << "RS: " << sqrt(err[1]/N_EVAL) << std::endl;
            std::cout << "Herding: " << sqrt(err[2]/N_EVAL) << std::endl;
            std::cout << "KCenter: " << sqrt(err[4]/N_EVAL) << std::endl;

        }
    }




}