//
// Created by Kexin Rong on 1/2/19.
//

#ifndef HBE_ADAPTIVERSDIAG_H
#define HBE_ADAPTIVERSDIAG_H


#include <Eigen/Dense>
#include "AdaptiveEstimator.h"
#include "kernel.h"
#include "math.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class AdaptiveRSDiag : public AdaptiveEstimator {

public:
    shared_ptr<MatrixXd> X;
    int numPoints;
    shared_ptr<Kernel> kernel;

    vector<double> contrib;
    vector<int> samples;

    AdaptiveRSDiag(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k, double lb, double eps);
    AdaptiveRSDiag(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k, int samples, double lb, double eps);


    int findTargetLevel(double est);
    int findActualLevel(VectorXd &q, double est, double eps);

    void getConstants(double est, double eps);
    void clearSamples();


    double RSTriv();
    double HBETriv(VectorXd &q, int level);



protected:
    double lb;
    std::mt19937_64 rng;
    int exp_k;
    double exp_w;
    double thresh;

    // Diagnosis constants
    vector<double> u;
    vector<int> s;
    int sample_count;
    double u_global;
    double w1Max;
    double w1Min;
    double lambda;
    double l;

    std::vector<double> evaluateQuery(VectorXd q, int level);

private:
    const double LOG2 = log(2);
    const double SQRT_2PI = sqrt(2.0 / M_PI);

    void buildLevels(double tau, double eps);
};


#endif //HBE_ADAPTIVERSDIAG_H
