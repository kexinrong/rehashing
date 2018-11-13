//
// Created by Kexin Rong on 2018-11-09.
//

#ifndef HBE_ADAPTIVERS_H
#define HBE_ADAPTIVERS_H


#include <Eigen/Dense>
#include "AdaptiveEstimator.h"
#include "kernel.h"
#include "math.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class AdaptiveRS : public AdaptiveEstimator {

public:
    shared_ptr<MatrixXd> X;
    int numPoints;
    shared_ptr<Kernel> kernel;

    vector<double> contrib;
    vector<int> samples;

    AdaptiveRS(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k, double lb, double eps);
    AdaptiveRS(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k, int samples, double lb, double eps);

    int findTargetLevel(double est);
    int findActualLevel(VectorXd &q, double est, double eps);

    double findRSRatio();
    double findHBERatio(VectorXd &q, int level);

protected:
    std::vector<double> evaluateQuery(VectorXd q, int level, int maxSamples);

private:
    const double LOG2 = log(2);
    const double SQRT_2PI = sqrt(2.0 / M_PI);

    void buildLevels(double tau, double eps);
};


#endif //HBE_ADAPTIVERS_H
