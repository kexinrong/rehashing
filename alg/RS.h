//
// Created by Kexin Rong on 9/4/18.
//

#ifndef HBE_RS_H
#define HBE_RS_H
#include <Eigen/Dense>
#include "MoMEstimator.h"
#include "kernel.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class RS : public MoMEstimator {

public:
    MatrixXd X;
    int numPoints;
    Kernel *kernel;

    RS(MatrixXd data, Kernel *k);

protected:
    double* MoM(VectorXd q, int L, int m);
};


#endif //HBE_RS_H
