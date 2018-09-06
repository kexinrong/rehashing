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

    double logVn(int d) { return 0.5 * d * LOG_PI - lgamma(0.5 * d + 1); }

    double logSn(int d) { return LOG_2PI + logVn(d - 1); }

public:
    Expkernel(int len) : Kernel(len) {}

    double getDimFactor(int d) {
        return exp(-logSn(d - 1)) / tgamma(d);
    }

    double density(VectorXd d) {
        double dist = 0;
        for (int i = 0; i < dim; i ++) {
            dist += pow(d(i) * invBandwidth[i], 2);
        }
        dist = sqrt(dist);
        return dimFactor * bwFactor * exp(-dist);
    }

    double invDensity(double p) { return -log(p); }
};


#endif //HBE_EXPKERNEL_H
