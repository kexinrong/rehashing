//
// Created by Kexin Rong on 9/3/18.
//

#ifndef HBE_UTILS_H
#define HBE_UTILS_H


#include <Eigen/Dense>
#include <math.h>
#include <iostream>
#include <random>
#include <memory>

using Eigen::MatrixXd;
using Eigen::VectorXd;

const double E1 = exp(1.5);
const double E2 = exp(1.854);
const double SQRT_2PI = sqrt(2.0 / M_PI);

const double a1 =  0.254829592;
const double a2 = -0.284496736;
const double a3 =  1.421413741;
const double a4 = -1.453152027;
const double a5 =  1.061405429;
const double p  =  0.3275911;

class mathUtils {

public:
    static double expRelVar(double mu) { return E1 / sqrt(mu); }
    static double gaussRelVar(double mu) { return E2 / sqrt(mu); }
    static double randomRelVar(double mu) { return 1 / mu; }

    static double expKernel(double x) { return exp(-x); }
    static double inverseExp(double mu) { return -log(mu); }

    static double erf(double x) {
       // Save the sign of x
        int sign = 1;
        if (x < 0) { sign = -1; };
        x = fabs(x);

        // A&S formula 7.1.26
        double t = 1.0/(1.0 + p*x);
        double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);

        return sign * y;
    }

    static double collisionProb(double c, int k) {
        double base = erf(1 / c) - SQRT_2PI * c * (1 - exp(-1 / (2 * c * c)));
        // std::pow slow
        double prob = base;
        int e = 1;
        while (e * 2 < k) {
            prob = prob * prob;
            e = e * 2;
        }
        for (int i = e; i < k; i ++) {
            prob = prob * base;
        }
        return prob;
    }

    static double collisionProb(double dist, double w, double k) {
        // Normalized distance
        double c = dist / w;
        if (c < 1e-2) {
            return 1.0;
        } else {
            return collisionProb(c, k);
        }
    }

    static double median(std::vector<double>& Z) {
        int L = Z.size();
        if (L == 1) { return Z[0]; }
        std::sort(Z.begin(), Z.end());
        double median = L % 2 ? Z[L / 2] : (Z[L / 2 - 1] + Z[L / 2]) / 2;
        return median;
    }

    static VectorXd randNormal(int n, std::mt19937_64 &rng) {
        VectorXd r(n);
        std::normal_distribution<double> normal(0.0, 1.0);
        for (int i = 0; i < n; i ++) {
            r(i) = normal(rng);
        }
        return r;
    }

    static VectorXd randUniform(int n, std::mt19937_64 &rng) {
        VectorXd r(n);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        for (int i = 0; i < n; i ++) {
            r(i) = uniform(rng);
        }
        return r;
    }

    static MatrixXd randNormal(int n, int d, std::mt19937_64 &rng) { ;
        MatrixXd r(n, d);
        std::normal_distribution<double> normal(0.0, 1.0);
        for (int i = 0; i < n; i ++) {
            for (int j = 0; j < d; j ++) {
                r(i, j) = normal(rng);
            }
        }
        return r;
    }



};
#endif //HBE_UTILS_H
