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
    shared_ptr<Kernel> kernel;

    RS(MatrixXd data, shared_ptr<Kernel> k);

protected:
    std::vector<double> MoM(VectorXd q, int L, int m);
};


#endif //HBE_RS_H