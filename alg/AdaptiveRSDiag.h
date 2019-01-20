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
    shared_ptr<Kernel> kernel;

    vector<double> contrib;
    vector<int> samples;

    double lambda;
    double l;

    AdaptiveRSDiag(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k, double lb, double eps);
    AdaptiveRSDiag(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k, int samples, double lb, double eps);

    int findActualLevel(VectorXd &q, double est, double eps);

    void findRings(int strategy, double eps, VectorXd &q, int level);

    void getConstants();
    void clearSamples();

    double RSDirect();
    double HBEDirect();

protected:
    double lb;
    std::mt19937_64 rng;
    int exp_k;
    double exp_w;
    double thresh;

    // Diagnosis constants
    vector<int> set_start;
    vector<double> u;
    int sample_count;
    double u_global;

    vector<double> w_mins;
    vector<double> w_maxs;
    vector<double> pmins;
    vector<double> pmaxs;
    vector<vector<double>> w_pps;
    vector<vector<double>> w_ps;
    vector<vector<int>> w_pp_idx;
    vector<vector<int>> w_p_idx;

    std::vector<double> evaluateQuery(VectorXd q, int level);

private:
    const double LOG2 = log(2);
    const double SQRT_2PI = sqrt(2.0 / M_PI);

    void buildLevels(double tau, double eps);
};


#endif //HBE_ADAPTIVERSDIAG_H
