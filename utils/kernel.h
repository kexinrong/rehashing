//
// Created by Kexin Rong on 9/3/18.
//

#ifndef HBE_KERNEL_H
#define HBE_KERNEL_H


#include <Eigen/Dense>
#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>
#include "mathUtils.h"

using Eigen::VectorXd;

using namespace std;

class Kernel {
public:
    bool denormalized = true;
    int dim;
    vector<double> invBandwidth;

    double dimFactor;
    double bwFactor;

    Kernel(int len) {
        dim = len;
        invBandwidth = vector<double>(dim, 1);
        dimFactor = 1.0;
        bwFactor = 1.0;
    }

    void setDenormalized(bool flag) {
        denormalized = flag;
    }

    virtual double getDimFactor(int dim) = 0;
    virtual double density(VectorXd d) = 0;
    virtual double density(double dist) = 0;
    virtual double invDensity(double p) = 0;

    double density(VectorXd p, VectorXd q) {
        return density(p - q);
    }

    void initialize(const vector<double>& dw) {
        invBandwidth = vector<double>(dim);
        for (int i = 0; i < dim; i++) {
            invBandwidth[i] = 1.0 / dw[i];
        }

        if (denormalized) {
            dimFactor = 1.0;
            bwFactor = 1.0;
        } else {
            dimFactor = getDimFactor(dim);
            bwFactor = 1;
            for (int i = 0; i < dim; i++) {
                bwFactor *= invBandwidth[i];
            }
        }
    }

    virtual bool shouldReject(double weight, double prob, double prob_mu, double mu, double delta) = 0;
    virtual int findLevel(double mu, int T) = 0;
    virtual double RelVar(double mu, double delta) = 0;

};

#endif //HBE_KERNEL_H
