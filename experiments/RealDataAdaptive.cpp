//
// Created by Kexin Rong on 9/10/18.
//

#include "RealDataAdaptive.h"
#include "expkernel.h"
#include "mathUtils.h"
#include "dataUtils.h"
#include "bandwidth.h"
#include "math.h"
#include "../alg/RS.h"
#include "../alg/naiveKDE.h"
#include "../alg/BaseLSH.h"
#include "../alg/multiHBE.h"
#include <chrono>

const double eps = 0.5;
const double tau = 0.001;

const int iterations = 1000;

const double sample_ratio = 4;

int main() {
    MatrixXd X = dataUtils::readFile("resources/shuttle.csv", true, 43500, 9);
    //MatrixXd X = dataUtils::readFile("resources/SUSY.csv", false, 500000, 1, 18);

    int n = X.rows();
    int dim = X.cols();
    std::cout << "N=" << n << ", d=" << dim << std::endl;

    // Bandwidth
    auto band = make_unique<Bandwidth>(n, dim);
    //band->multiplier = 5;
    band->getBandwidth(X);
    shared_ptr<Kernel> kernel = make_shared<Expkernel>(dim);
    kernel->initialize(band->bw);
    dataUtils::checkBandwidthSamples(X, eps, kernel);
    // Normalized by bandwidth
    X = dataUtils::normalizeBandwidth(X, band->bw);
    shared_ptr<MatrixXd> X_ptr = make_shared<MatrixXd>(X);

    double beta = 0;
    double stepSize = 1 / log(n);
    shared_ptr<Kernel> simpleKernel = make_shared<Expkernel>(dim);
    naiveKDE naive(X_ptr, simpleKernel);
    RS rs(X_ptr, simpleKernel);
    while (beta < 0.5) {
        beta += stepSize;
        beta = min(beta, 0.5);

        // Algorithms init
        auto t1 = std::chrono::high_resolution_clock::now();
        multiHBE mr(X_ptr, simpleKernel, 1, tau, eps, 0.52, 1);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "MR_HBE Table Init: " << std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count() << std::endl;

        // Random
        std::uniform_int_distribution<int> distribution(0, n - 1);
        std::random_device rd;
        std::mt19937_64 rng = std::mt19937_64(rd());
        vector<double> time = vector<double>(3, 0);
        vector<double> error = vector<double>(2, 0);
        double samples = 0;
        for (int j = 0; j < iterations; j ++) {
            int idx = distribution(rng);
            VectorXd q = X.row(idx);
            // Naive
            t1 = std::chrono::high_resolution_clock::now();
            double kde = naive.query(q);
            t2 = std::chrono::high_resolution_clock::now();
            time[0] += std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();

            // MR
            t1 = std::chrono::high_resolution_clock::now();
            vector<double> est = mr.query(q);
            t2 = std::chrono::high_resolution_clock::now();
            mr.clear();
            time[2] += std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
            error[1] += fabs(kde - est[0]) / kde;
            samples += est[1];

            // RS
            t1 = std::chrono::high_resolution_clock::now();
            double rsKDE = rs.query(q, tau, int(est[1] * sample_ratio));
            t2 = std::chrono::high_resolution_clock::now();
            time[1] += std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
            error[0] += fabs(kde - rsKDE) / kde;
        }
        std::cout << "Naive average time: " << time[0] / iterations / 1e6  << std::endl;
        std::cout << "RS average time: " << time[1] / iterations / 1e6 << std::endl;
        std::cout << "RS average error: " << error[0] / iterations << std::endl;
        std::cout << "MR average time: " << time[2] / iterations / 1e6 << std::endl;
        std::cout << "MR average error: " << error[1] / iterations << std::endl;
        std::cout << "MR average samples: " << samples / iterations << std::endl;
        std::cout << "-------------------------------------------------------" << std::endl;
        break;
    }



}