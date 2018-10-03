//
// Created by Kexin Rong on 9/3/18.
//

#ifndef HBE_EXPKERNEL_H
#define HBE_EXPKERNEL_H

#include "kernel.h"
#include <math.h>
#include <iostream>

using Eigen::VectorXd;

class Expkernel : public Kernel {
private:
    const double LOG_PI = log(M_PI);
    const double LOG_2PI = log(2 * M_PI);
    const double LOG_25 = log(0.25);

    double logVn(int d) { return 0.5 * d * LOG_PI - lgamma(0.5 * d + 1); }

    double logSn(int d) { return LOG_2PI + logVn(d - 1); }

public:
    Expkernel(int len) : Kernel(len) {}

    double getDimFactor() {
        return exp(-logSn(dim - 1)) / tgamma(dim);
    }

    double density(VectorXd d) {
        double dist = 0;
        for (int i = 0; i < dim; i ++) {
            dist += d(i) * d(i) * invBandwidth[i] * invBandwidth[i];
        }
        dist = sqrt(dist);
        return dimFactor * bwFactor * exp(-dist);
    }

    double density(double dist) { return exp(-dist); }
    double invDensity(double p) { return -log(p); }

    bool shouldReject(double weight, double prob, double prob_mu, double log_mu, double delta) {
        double alpha = log(weight) / log_mu;
        double beta = log(prob) / log_mu;
        double gamma = log(prob_mu) / log_mu;
        return alpha - 1 + gamma - 2 * beta < -delta;
    }

    int findLevel(double mu, int T) {
        return min(int(ceil(log(mu) / LOG_25)), T);
    }

    double RelVar(double mu, double delta) { return pow(mu, -delta); }
};


#endif //HBE_EXPKERNEL_H
