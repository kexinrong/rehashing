//
// Created by Kexin Rong on 2018-11-11.
//

#ifndef HBE_ADAPTIVEHBE_H
#define HBE_ADAPTIVEHBE_H

#include <Eigen/Dense>
#include "SketchLSH.h"
#include "AdaptiveEstimator.h"
#include "kernel.h"
#include "math.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class AdaptiveHBE : public AdaptiveEstimator {
public:
    vector<SketchLSH> levels;
    double tau;

    AdaptiveHBE(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k, double lb, double eps);

protected:
    std::vector<double> ti;
    std::vector<int> ki;
    std::vector<double> wi;

    std::vector<double> evaluateQuery(VectorXd q, int level, int maxSamples);

private:
    const double LOG2 = log(2);
    const double SQRT_2PI = sqrt(2.0 / M_PI);

    void buildLevels(shared_ptr<MatrixXd> X, shared_ptr<Kernel> k, double tau, double eps);

};


#endif //HBE_ADAPTIVEHBE_H
