#ifndef HBE_NAIVEKDE_H
#define HBE_NAIVEKDE_H


#include <Eigen/Dense>
#include "kernel.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

///
/// Class for exhaustively computing KDE
///
class naiveKDE {
public:
    naiveKDE(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k);
    double query(VectorXd q);
private:
    shared_ptr<MatrixXd> X;
    int numPoints;
    shared_ptr<Kernel> kernel;
};


#endif //HBE_NAIVEKDE_H
