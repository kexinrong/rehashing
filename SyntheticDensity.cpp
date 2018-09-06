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


const double eps = 0.5;
const double beta = 0.5;

const int uN = 50000;
const int uC = 1000;
const int cN = 50000;
const int cC = 4;
const int scales = 4;
const int dim = 3;
const double spread = 0.01;

const int iterations = 100;

int main() {
    for (long mu = 100; mu < 1000000; mu *= 10) {
        std::cout << "mu=" << mu << std::endl;
        double density = 1.0 / mu;
        GenericInstance data = SyntheticData::genMixed(uN, cN, uC, cC, dim, density, scales, spread);
        MatrixXd X = data.points;
        int n = X.rows();
        std::cout << "N=" << n << ", d=" << dim << std::endl;

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
        E2LSH hbe(X, M, w, k, 100, kernel, 1);

        int m1 = min(n, (int)ceil(1 / eps / eps * mathUtils::randomRelVar(tau)));
        int m2 = min(n, (int)ceil(1 / eps / eps * mathUtils::expRelVar(tau)));
        std::cout << "RS samples:" << m1 << ", HBE samples:" << m2 << std::endl;
        double error = 0;
        for (int j = 0; j < iterations; j ++) {
            VectorXd q = data.query(1, false);
            // Naive
            double kde = naive.query(q);
            // RS
            double rsKDE = rs.query(q, tau, m1);
            // HBE
            double hbeKDE = hbe.query(q, tau, m2);
            error += fabs(kde - rsKDE) / kde;
            std::cout << kde << "," << rsKDE << ',' << hbeKDE << std::endl;
        }
        std::cout << "RS average error: " << error / iterations << std::endl;
    }

}