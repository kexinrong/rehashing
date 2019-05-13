#ifndef HBE_ADAPTIVERSDIAG_H
#define HBE_ADAPTIVERSDIAG_H


#include <Eigen/Dense>
#include "AdaptiveEstimator.h"
#include "kernel.h"
#include "math.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

///
/// Diagnostic procedure implemented via adaptive random sampling.
///
class AdaptiveRSDiag : public AdaptiveEstimator {

public:
    ///
    /// S0 in Algorithm 1, line 4
    ///
    vector<int> samples;

    ///
    /// k(q, x) for each x in S0
    ///
    vector<double> contrib;

    ///
    /// Eq (9) in the main paper. Used for visualization.
    ///
    double lambda;

    ///
    /// Eq (10) in the main paper. Used for visualization.
    ///
    double l;

    AdaptiveRSDiag(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k, double lb, double eps);
    AdaptiveRSDiag(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k, int samples, double lb, double eps);

    ///
    /// \param q query
    /// \param est final estimate from adapative sampling
    /// \param eps relative error
    /// \return smallest level that we can get an esimate within (1+/-eps) of est
    int findActualLevel(VectorXd &q, double est, double eps);

    /// Set lambda and l according to different strategies
    /// \param strategy 0: S2, S3 = {}; 1: Eq(9), (10)
    /// \param eps
    /// \param q
    /// \param level
    void findRings(int strategy, double eps, VectorXd &q, int level);

    ///
    /// Precompute
    void getConstants();
    void clearSamples();

    ///
    /// \return variance upper bound for RS
    double vbRS();
    ///
    /// \return variance upper bound for HBE
    double vbHBE();

protected:
    std::vector<double> evaluateQuery(VectorXd q, int level);

private:
    shared_ptr<MatrixXd> X;
    shared_ptr<Kernel> kernel;

    double lb;
    std::mt19937_64 rng;
    int exp_k;
    double exp_w;
    double thresh;

    // Diagnosis constants
    vector<size_t> set_start;
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


    const double LOG2 = log(2);
    const double SQRT_2PI = sqrt(2.0 / M_PI);

    void buildLevels(double tau, double eps);
};


#endif //HBE_ADAPTIVERSDIAG_H
