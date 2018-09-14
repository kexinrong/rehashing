//
// Created by Kexin Rong on 9/13/18.
//

#include "RealDataAll.h"
#include "expkernel.h"
#include "mathUtils.h"
#include "dataUtils.h"
#include "bandwidth.h"
#include "math.h"
#include "../alg/RS.h"
#include "../alg/naiveKDE.h"
#include "../alg/BaseLSH.h"
#include <chrono>
#include <boost/math/distributions/normal.hpp>

const double eps = 0.5;
const double tau = 0.001;
const double beta = 0.1;

const std::vector<int> hbe_samples = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};

int main() {
    MatrixXd X = dataUtils::readFile("resources/shuttle.csv", true, 43500, 9);
    int n = X.rows();
    int dim = X.cols();
    std::cout << "N=" << n << ", d=" << dim << std::endl;

    MatrixXd tmp = dataUtils::readFile("resources/true_densities/shuttle_all_errors.txt", true, 43500, 1);
    std::vector<double> densities(n);
    for (int i = 0; i < n; i ++) {
        densities[i] = tmp(i, 0);
    }
    int pos = int(n * 0.1);
    std::nth_element (densities.begin(), densities.begin()+pos, densities.end());
    // 10th percentile
    double thresh = densities[pos];

    // Bandwidth
    auto band = make_unique<Bandwidth>(n, dim);
    band->getBandwidth(X);
    shared_ptr<Kernel> kernel = make_shared<Expkernel>(dim);
    kernel->initialize(band->bw);
    dataUtils::checkBandwidthSamples(X, eps, kernel);
    // Normalized by bandwidth
    X = dataUtils::normalizeBandwidth(X, band->bw);
    shared_ptr<MatrixXd> X_ptr = make_shared<MatrixXd>(X);

    // Estimate parameters
    double means = ceil(6 * mathUtils::expRelVar(tau) / eps / eps);
    int M = (int)(means * 1.1);
    double diam = dataUtils::estimateDiameter(X, tau);
    int k = dataUtils::getPower(diam, beta);
    double w = dataUtils::getWidth(3, beta);

    // Algorithms init
    std::cout << "M=" << M << ",w=" << w << ",k=" << k << std::endl;
    shared_ptr<Kernel> simpleKernel = make_shared<Expkernel>(dim);
    naiveKDE naive(X_ptr, simpleKernel);
    RS rs(X_ptr, simpleKernel);
    auto t1 = std::chrono::high_resolution_clock::now();
    BaseLSH hbe(X_ptr, M, w, k, 1, simpleKernel, 1);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "HBE Table Init: " << std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count() << std::endl;

    double ratio = 3;
    for (size_t i = 0; i < hbe_samples.size(); i ++) {
        int m1 = hbe_samples[i];
        int m2 = int(hbe_samples[i] * ratio);
        std::cout << "#hbe samples: " << m1 << ", #rs samples: " << m2 << std::endl;
        for (int repeat = 0; repeat < 3; repeat++ ) {
            // Random
            vector<double> time = vector<double>(3, 0);
            vector<double> error = vector<double>(2, 0);
            int iterations = 0;
            for (int idx = 0; idx < n; idx++) {
                if (densities[idx] > thresh) { continue; }
                iterations += 1;
                VectorXd q = X.row(idx);
                // Naive
                t1 = std::chrono::high_resolution_clock::now();
                double kde = naive.query(q);
                t2 = std::chrono::high_resolution_clock::now();
                time[0] += std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();

                // RS
                t1 = std::chrono::high_resolution_clock::now();
                double rsKDE = rs.query(q, tau, m2);
                t2 = std::chrono::high_resolution_clock::now();
                time[1] += std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
                error[0] += fabs(kde - rsKDE) / kde;

                // HBE
                t1 = std::chrono::high_resolution_clock::now();
                double hbeKDE = hbe.query(q, tau, m1);
                t2 = std::chrono::high_resolution_clock::now();
                time[2] += std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
                error[1] += fabs(kde - hbeKDE) / kde;
            }
            std::cout << iterations << std::endl;
            std::cout << "Naive average time: " << time[0] / iterations / 1e6  << std::endl;
            std::cout << "RS average time: " << time[1] / iterations / 1e6 << std::endl;
            std::cout << "RS average error: " << error[0] / iterations << std::endl;
            std::cout << "HBE average time: " << time[2] / iterations / 1e6 << std::endl;
            std::cout << "HBE average error: " << error[1] / iterations << std::endl;
            std::cout << "-------------------------------------------------------" << std::endl;
        }
    }

}