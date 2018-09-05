//
// Created by Kexin Rong on 9/4/18.
//

#ifndef HBE_NAIVEKDE_H
#define HBE_NAIVEKDE_H
#include <Eigen/Dense>
#include "kernel.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class naiveKDE {
public:
    MatrixXd X;
    int numPoints;
    Kernel *kernel;

    naiveKDE(MatrixXd data, Kernel *k);
    double query(VectorXd q);
};


#endif //HBE_NAIVEKDE_H
