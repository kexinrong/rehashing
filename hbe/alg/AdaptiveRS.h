#ifndef HBE_ADAPTIVERS_H
#define HBE_ADAPTIVERS_H


#include <Eigen/Dense>
#include "AdaptiveEstimator.h"
#include "kernel.h"
#include "math.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

///
/// Adaptive sampling via random sampling.
///
class AdaptiveRS : public AdaptiveEstimator {

public:
    ///
    /// \param data
    /// \param k
    /// \param lb minimum density
    /// \param eps relative error
    AdaptiveRS(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k, double lb, double eps);

    ///
    /// \param data dataset
    /// \param k kernel
    /// \param samples size of random sample reservoir
    /// \param lb minimum density
    /// \param eps relative error
    AdaptiveRS(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k, int samples, double lb, double eps);

protected:
    double lb;
    std::mt19937_64 rng;
    std::vector<double> evaluateQuery(VectorXd q, int level);

private:
    ///
    /// Dataset
    ///
    shared_ptr<MatrixXd> X;

    ///
    /// Kernel function
    ///
    shared_ptr<Kernel> kernel;
    const double LOG2 = log(2);
    const double SQRT_2PI = sqrt(2.0 / M_PI);

    ///
    /// Helper function to build each level of adaptive sampling.
    /// \param tau minimum density
    /// \param eps relative error
    void buildLevels(double tau, double eps);
};


#endif //HBE_ADAPTIVERS_H
