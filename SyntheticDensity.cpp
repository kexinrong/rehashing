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

const double eps = 0.5;
const double delta = 0.001;
const double beta = 0.5;

const int uN = 100000;
const int uC = 1000;
const int cN = 125000;
const int cC = 4;
const int scales = 4;
const int dim = 100;
const double spread = 0.01;

const int iterations = 100;

int main() {
    for (long mu = 100; mu < 1000000; mu += 10) {
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
        double diam = dataUtils::estimateDiameter(X, tau);
        shared_ptr<Expkernel> kernel = make_shared<Expkernel>(dim);

        naiveKDE naive(X, kernel);
        RS rs(X, kernel);
        int m = min(n, (int)ceil(1 / eps / eps * mathUtils::randomRelVar(tau)));
        double error = 0;
        for (int j = 0; j < iterations; j ++) {
            VectorXd q = data.query(1, false);
            // Naive
            double kde = naive.query(q);
            // RS
            double rsKDE = rs.query(q, tau, m);
            error += fabs(kde - rsKDE) / kde;
        }
        std::cout << "RS average error: " << error / iterations << std::endl;
    }

}