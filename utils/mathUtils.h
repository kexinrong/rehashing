//
// Created by Kexin Rong on 9/3/18.
//

#ifndef HBE_UTILS_H
#define HBE_UTILS_H

#include <Eigen/Dense>
#include <math.h>
#include <iostream>
#include <random>

using Eigen::MatrixXd;
using Eigen::VectorXd;


const double E1 = exp(1.5);
const double E2 = exp(1.854);


class mathUtils {

public:
    static const double E1;
    static const double E2;

    static std::default_random_engine generator;

    static double expRelVar(double mu) { return E1 / sqrt(mu); }
    static double gaussRelVar(double mu) { return E2 / sqrt(mu); }
    static double randomRelVar(double mu) { return 1 / mu; }

    static double expKernel(double x) { return exp(-x); }
    static double inverseExp(double mu) { return -log(mu); }

    static double median(double *Z, int L) {
        if (L == 1) { return Z[0]; }
        std::sort(&Z[0], &Z[L]);
        double median = L % 2 ? Z[L / 2] : (Z[L / 2 - 1] + Z[L / 2]) / 2;
        return median;
    }

    static VectorXd randNormal(int n) {
        VectorXd r(n);
        static std::normal_distribution<double> normal(0.0, 1.0);
        for (int i = 0; i < n; i ++) {
            r(i) = normal(generator);
        }
        return r;
    }

    static MatrixXd randNormal(int n, int d) {
        MatrixXd r(n, d);
        static std::normal_distribution<double> normal(0.0, 1.0);
        for (int i = 0; i < n; i ++) {
            for (int j = 0; j < d; j ++) {
                r(i) = normal(generator);
            }
        }
        return r;
    }



};

#endif //HBE_UTILS_H
