//
// Created by Kexin Rong on 9/10/18.
//

#include "SyntheticDensity.h"
#include "expkernel.h"
#include "mathUtils.h"
#include "dataUtils.h"
#include "bandwidth.h"
#include "math.h"
#include "alg/RS.h"
#include "alg/naiveKDE.h"
#include "alg/BaseLSH.h"
#include <chrono>

const double eps = 0.5;
const double tau = 0.001;
const double beta = 0.1;

const int iterations = 1000;

int main() {
    srand (time(NULL));
    MatrixXd X = dataUtils::readFile("resources/shuttle.csv", true, 43500, 9);

    int n = X.rows();
    int dim = X.cols();
    std::cout << "N=" << n << ", d=" << dim << std::endl;

    // Bandwidth
    auto band = make_unique<Bandwidth>(n, dim);
    band->getBandwidth(X);
    shared_ptr<Kernel> kernel = make_shared<Expkernel>(dim);
    kernel->initialize(band->bw);
    dataUtils::checkBandwidthSamples(X, eps, kernel);
    // Normalized by bandwidth
    X = dataUtils::normalizeBandwidth(X, band->bw);

    // Estimate parameters
    double means = ceil(6 * mathUtils::expRelVar(tau) / eps / eps);
    int M = (int)(means * 1.1);
    double diam = dataUtils::estimateDiameter(X, tau);
    int k = dataUtils::getPower(diam, beta);
    double w = dataUtils::getWidth(k, beta);

    // Algorithms init
    std::cout << "M=" << M << ",w=" << w << ",k=" << k << std::endl;
    shared_ptr<Kernel> simpleKernel = make_shared<Expkernel>(dim);
    naiveKDE naive(X, simpleKernel);
    RS rs(X, simpleKernel);
    BaseLSH hbe(X, M, w, k, 1, simpleKernel, 1);

    vector<double> time = vector<double>(3, 0);
    vector<double> error = vector<double>(2, 0);
    int m1 = 100;
    for (int j = 0; j < iterations; j ++) {
        int idx = rand() % n;
        VectorXd q = X.row(idx);
        // Naive
        auto t1 = std::chrono::high_resolution_clock::now();
        double kde = naive.query(q);
        auto t2 = std::chrono::high_resolution_clock::now();
        time[0] += std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();

        // RS
        t1 = std::chrono::high_resolution_clock::now();
        double rsKDE = rs.query(q, tau, m1);
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
    std::cout << "Naive average time: " << time[0] / iterations / 1e6  << std::endl;
    std::cout << "RS average time: " << time[1] / iterations / 1e6 << std::endl;
    std::cout << "RS average error: " << error[0] / iterations << std::endl;
    std::cout << "HBE average time: " << time[2] / iterations / 1e6 << std::endl;
    std::cout << "HBE average error: " << error[1] / iterations << std::endl;
    std::cout << "-------------------------------------------------------" << std::endl;
}