//
// Created by Kexin Rong on 9/21/18.
//

#ifndef HBE_GAUSSIANKERNEL_H
#define HBE_GAUSSIANKERNEL_H

#include "kernel.h"
#include <math.h>
#include <iostream>

using Eigen::VectorXd;

class Gaussiankernel : public Kernel {
private:
    const double LOG_25 = log(0.25);
    const double E1 = exp(1.5);
    const double E2 = exp(1.854);

    double r;

public:
    Gaussiankernel(int len) : Kernel(len) {}

    Gaussiankernel(int len, double radius) : Kernel(len) {
        r = radius;
    }


    double getDimFactor() {
        return pow(2 * M_PI, -0.5 * dim);
    }

    double density(const VectorXd& d) {
        double dist = 0;
        for (int i = 0; i < dim; i ++) {
            dist += d(i) * d(i) * invBandwidth[i] * invBandwidth[i];
        }
        return dimFactor * bwFactor * exp(-dist);
    }

    double density(double dist) { return exp(-dist * dist); }
    double invDensity(double p) { return sqrt(-log(p)); }

    bool shouldReject(double weight, double prob, double prob_mu, double log_mu, double delta) {
        double alpha = log(weight) / log_mu;
        double beta = log(prob) / log_mu;
        double gamma = log(prob_mu) / log_mu;
        double c = log(4 * E1) / log_mu;
        return alpha - 1 + gamma - 2 * beta < -delta + c;
    }

    int findLevel(double mu, int T) {
        return min(T, int(ceil(log(mu * exp(2 / M_PI / r / r)) / LOG_25)));
    }

    double RelVar(double mu, double delta) { return 4 * E1 * pow(mu, -delta); }

    string getName() { return "gaussian"; }
    double RelVar(double mu) { return E2 / sqrt(mu); }
};


#endif //HBE_GAUSSIANKERNEL_H
