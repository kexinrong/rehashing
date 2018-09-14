//
// Created by Kexin Rong on 9/10/18.
//

#include "expkernel.h"
#include "mathUtils.h"
#include "dataUtils.h"
#include "bandwidth.h"
#include "math.h"
#include "../alg/RS.h"
#include "../alg/naiveKDE.h"
#include "../alg/BaseLSH.h"
#include <chrono>

const double eps = 0.5;
const double tau = 0.0001;
const double beta = 0.1;

const int iterations = 1000;

const std::vector<int> hbe_samples = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};

int main() {
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
        // Random
        std::uniform_int_distribution<int> distribution(0, n - 1);
        std::random_device rd;
        std::mt19937_64 rng = std::mt19937_64(rd());
        for (int repeats = 0; repeats < 5; repeats ++) {
            vector<double> time = vector<double>(3, 0);
            vector<double> error = vector<double>(2, 0);
            for (int j = 0; j < iterations; j ++) {
                int idx = distribution(rng);
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
            std::cout << "# " << repeats << std::endl;
            std::cout << "Naive average time: " << time[0] / iterations / 1e6  << std::endl;
            std::cout << "RS average time: " << time[1] / iterations / 1e6 << std::endl;
            std::cout << "RS average error: " << error[0] / iterations << std::endl;
            std::cout << "HBE average time: " << time[2] / iterations / 1e6 << std::endl;
            std::cout << "HBE average error: " << error[1] / iterations << std::endl;
            std::cout << "-------------------------------------------------------" << std::endl;
        }

    }

}