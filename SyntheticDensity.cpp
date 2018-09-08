//
// Created by Kexin Rong on 9/5/18.
//

#include "SyntheticDensity.h"
#include "expkernel.h"
#include "mathUtils.h"
#include "dataUtils.h"
#include "math.h"
#include "alg/RS.h"
#include "alg/naiveKDE.h"
#include "alg/E2LSH.h"
#include <chrono>

const double eps = 0.5;
const double beta = 0.5;

const int uN = 100000;
const int uC = 1000;
const int cN = 100000;
const int cC = 4;
const int scales = 4;
const int dim = 3;
const double spread = 0.01;

const int iterations = 1000;

int main() {
    for (long mu = 100; mu < 1000000; mu *= 10) {
        std::cout << "-------------------------------------------------------" << std::endl;
        double density = 1.0 / mu;
        GenericInstance data = SyntheticData::genMixed(uN, cN, uC, cC, dim, density, scales, spread);
        MatrixXd X = data.points;
        int n = X.rows();
        std::cout << "mu=" << mu << ", N=" << n << ", d=" << dim << std::endl;

        // minimum density that we wish to be able to approximate
        double tau = density * 0.5;

        // Estimate parameters
        double means = ceil(6 * mathUtils::expRelVar(tau) / eps / eps);
        int M = (int)(means * 1.1);
        double w = 2.5 * log(mu);
        int k = dataUtils::getPowerW(w, beta);

        shared_ptr<Expkernel> kernel = make_shared<Expkernel>(dim);

        // Algorithms init
        std::cout << "M=" << M << ",w=" << w << ",k=" << k << std::endl;
        naiveKDE naive(X, kernel);
        RS rs(X, kernel);
        E2LSH hbe(X, M, w, k, 50, kernel, 1);

        int m1 = min(n, (int)ceil(1 / eps / eps * mathUtils::randomRelVar(tau)));
        int m2 = min(n, (int)ceil(1 / eps / eps * mathUtils::expRelVar(tau)));
        vector<double> time = vector<double>(3, 0);
        vector<double> error = vector<double>(2, 0);
        std::cout << "RS samples:" << m1 << ", HBE samples:" << m2 << std::endl;
        for (int j = 0; j < iterations; j ++) {
            VectorXd q = data.query(1, false);
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
            double hbeKDE = hbe.query(q, tau, m2);
            t2 = std::chrono::high_resolution_clock::now();
            time[2] += std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
            error[1] += fabs(kde - hbeKDE) / kde;
        }
        std::cout << "Naive average time: " << time[0] / iterations / 1e9 << std::endl;
        std::cout << "RS average error: " << error[0] / iterations << std::endl;
        std::cout << "RS average time: " << time[1] / iterations / 1e9 << std::endl;
        std::cout << "HBE average error: " << error[1] / iterations << std::endl;
        std::cout << "HBE average time: " << time[2] / iterations / 1e9 << std::endl;
        std::cout << "-------------------------------------------------------" << std::endl;
    }

}