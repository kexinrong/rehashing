//
// Created by Kexin Rong on 9/4/18.
//

#ifndef HBE_RS_H
#define HBE_RS_H
#include <Eigen/Dense>
#include "../utils/kernel.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class RS {

public:
    MatrixXd X;
    int numPoints;
    Kernel *kernel;

    RS(MatrixXd data, Kernel *k);
    double query(VectorXd q, double lb, int m);

private:
    double* MoM(VectorXd q, int L, int m);
};


#endif //HBE_RS_H
