#ifndef HBE_ADAPTIVEHBE_H
#define HBE_ADAPTIVEHBE_H

#include <Eigen/Dense>
#include "SketchHBE.h"
#include "UniformHBE.h"
#include "AdaptiveEstimator.h"
#include "kernel.h"
#include "math.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

///
/// Adaptive sampling via HBE.
///
class AdaptiveHBE : public AdaptiveEstimator {
public:
    ///
    /// Collection of estimators at all levels.
    /// At each level, we have a collection of hashing tables created by evaluating
    /// i.i.d hash function with a partiuclar hashing schemes on random samples from the dataset.
    ///
    vector<UniformHBE> u_levels;

    ///
    /// Collection of estimators at all levels.
    /// At each level, we have a collection of hashing tables created by evaluating
    /// i.i.d hash function with a partiuclar hashing schemes on samples drawn from HBS.
    ///
    vector<SketchHBE> s_levels;

    ///
    /// \param data dataset
    /// \param k kernel
    /// \param lb tau (minimum density)
    /// \param eps relative error
    /// \param sketch if true, use HBS as a sketch; otherwise use uniform sampling as a sketch
    AdaptiveHBE(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k, double lb, double eps, bool sketch);

protected:
    std::vector<double> evaluateQuery(VectorXd q, int level);

private:
    ///
    /// Minimum density
    ///
    double tau;

    bool use_sketch;
    const double LOG2 = log(2);
    const double SQRT_2PI = sqrt(2.0 / M_PI);

    ///
    /// Helper function to build data structure for each level of adaptive sampling.
    /// \param X dataset
    /// \param k kernel funcation
    /// \param tau minimum density
    /// \param eps relative error
    /// \param sketch whether to use HBS or Uniform
    void buildLevels(shared_ptr<MatrixXd> X, shared_ptr<Kernel> k, double tau, double eps, bool sketch);

};


#endif //HBE_ADAPTIVEHBE_H
