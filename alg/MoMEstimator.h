//
// Created by Kexin Rong on 9/5/18.
//

#ifndef HBE_MOMESTIMATOR_H
#define HBE_MOMESTIMATOR_H

#include <Eigen/Dense>
#include "mathUtils.h"

using Eigen::VectorXd;

class MoMEstimator {
public:
    double query(VectorXd q, double lb, int m) {
        int L = 1;
        double* Z = MoM(q, L, m);
        double est = mathUtils::median(Z, L) / m;
        if (est < lb) {
            return 0;
        } else {
            return est;
        }
    }

protected:
    virtual double* MoM(VectorXd q, int L, int m) = 0;

};


#endif //HBE_MOMESTIMATOR_H
