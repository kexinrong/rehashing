//
// Created by Kexin Rong on 9/11/18.
//

#include "SyntheticFixAcc.h"
#include "expkernel.h"
#include "mathUtils.h"
#include "dataUtils.h"
#include "math.h"
#include "alg/RS.h"
#include "alg/naiveKDE.h"
#include "alg/BaseLSH.h"
#include "data/SyntheticData.h"
#include <chrono>

const double eps = 0.5;
const double beta = 0.5;

const double acc = 0.1;

const int uN = 100;
const int uC = 1000;
const int cN = 125000;
const int cC = 4;
const int scales = 4;
const int dim = 3;
const double spread = 0.01;

const int iterations = 1000;


int main() {
    for (long mu = 100; mu <= 1000000; mu *= 10) {
        std::cout << "-------------------------------------------------------" << std::endl;
        double density = 1.0 / mu;
        GenericInstance data = SyntheticData::genMixed(uN, cN, uC, cC, dim, density, scales, spread);
        MatrixXd X = data.points;
        int n = X.rows();
        shared_ptr<MatrixXd> X_ptr = make_shared<MatrixXd>(X);
        std::cout << "mu=" << mu << ", N=" << n << ", d=" << dim << std::endl;

        // minimum density that we wish to be able to approximate
        double tau = density;

        // Estimate parameters
        double means = ceil(6 * mathUtils::expRelVar(tau) / eps / eps);
        int M = (int)(means * 1.1);
        double w = 2.5 * log(mu);
        int k = dataUtils::getPowerW(w, beta);

        shared_ptr<Expkernel> kernel = make_shared<Expkernel>(dim);

        // Algorithms init
        std::cout << "M=" << M << ",w=" << w << ",k=" << k << std::endl;
        naiveKDE naive(X_ptr, kernel);
        RS rs(X_ptr, kernel);
        BaseLSH hbe(X_ptr, M, w, k, 1, kernel, 1);

        int m1 = min(n, (int)ceil(1 / eps / eps * mathUtils::randomRelVar(tau)));
        int m2 = min(n, (int)ceil(1 / eps / eps * mathUtils::expRelVar(tau)));

        bool foundHBE = false;
        for (int s = m2 / 2; s < m1; s += 5) {
            vector<double> time = vector<double>(3, 0);
            vector<double> error = vector<double>(2, 0);
            for (int j = 0; j < iterations; j ++) {
                VectorXd q = data.query(1, false);
                // Naive
                auto t1 = std::chrono::high_resolution_clock::now();
                double kde = naive.query(q);
                auto t2 = std::chrono::high_resolution_clock::now();
                time[0] += std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();

                // HBE
                if (!foundHBE) {
                    t1 = std::chrono::high_resolution_clock::now();
                    double hbeKDE = hbe.query(q, tau, s);
                    t2 = std::chrono::high_resolution_clock::now();
                    time[2] += std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
                    error[1] += fabs(kde - hbeKDE) / kde;
                }

                // RS
                if (foundHBE) {
                    t1 = std::chrono::high_resolution_clock::now();
                    double rsKDE = rs.query(q, tau, s);
                    t2 = std::chrono::high_resolution_clock::now();
                    time[1] += std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
                    error[0] += fabs(kde - rsKDE) / kde;
                }
            }
            std::cout << "s=" << s << std::endl;
            if (foundHBE) {
                std::cout << "RS: " << time[1] / iterations / 1e6 << "," << error[0] / iterations << std::endl;
            } else {
                std::cout << "HBE: " << time[2] / iterations / 1e6 << "," << error[1] / iterations << std::endl;
                if (error[1] / iterations <= acc * 0.9) { foundHBE = true; }
            }
            if (foundHBE && error[0] / iterations <= acc * 0.9) { break; }
        }
        std::cout << "-------------------------------------------------------" << std::endl;
    }

}