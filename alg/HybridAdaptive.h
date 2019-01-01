//
// Created by Kexin Rong on 12/30/18.
//

#ifndef HBE_HYBRIDADAPTIVE_H
#define HBE_HYBRIDADAPTIVE_H


#include <Eigen/Dense>
#include "AdaptiveEstimator.h"
#include "kernel.h"
#include "BaseLSH.h"
#include "math.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class HybridAdaptive : public AdaptiveEstimator {

public:
    vector<BaseLSH> levels;
    shared_ptr<MatrixXd> X;
    int numPoints;
    shared_ptr<Kernel> kernel;

    vector<bool> random;

    HybridAdaptive(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k, double lb, double eps);


protected:
    double tau;
    std::mt19937_64 rng;
    int exp_k;
    double exp_w;
    std::vector<double> evaluateQuery(VectorXd q, int level);

private:
    const double LOG2 = log(2);
    const double SQRT_2PI = sqrt(2.0 / M_PI);

    void buildLevels(shared_ptr<MatrixXd> X, shared_ptr<Kernel> k, double tau, double eps);
};


#endif //HBE_HYBRIDADAPTIVE_H
