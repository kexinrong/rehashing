//
// Created by Kexin Rong on 9/3/18.
//

#ifndef HBE_KERNEL_H
#define HBE_KERNEL_H


#include <cstddef>

using namespace std;

class kernel {
protected:
    double *delta;

public:
    bool denormalized = true;
    int dim;
    double *invBandwidth;

    double dimFactor;
    double bwFactor;

    void setDenormalized(bool flag) {
        denormalized = flag;
    }

    virtual double getDimFactor(int dim);
    virtual double density(double *d);
    virtual double invDensity(double p);

    double qdensity(double* p, double* q) {
        for (int i = 0; i < dim; i ++) {
            delta[i] = p[i] - q[i];
        }
        return density(delta);
    }


    void initialize(double* dw, size_t len) {
        dim = len;
        invBandwidth = new double[dim];
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

        delta = new double[dim];
    }
};


#endif //HBE_KERNEL_H
