#ifndef HBE_RS_H
#define HBE_RS_H


#include <Eigen/Dense>
#include "MoMEstimator.h"
#include "kernel.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

///
/// Random sampling estimator
///
class RS : public MoMEstimator {

public:
    /// Default constructor.
    /// \param data dataset
    /// \param k kernel
    RS(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k);

    /// Constructor with fixed reservoir size.
    /// \param data the whole dataset
    /// \param k kernel
    /// \param samples size of the reservoir
    RS(shared_ptr<MatrixXd> data, shared_ptr<Kernel> k, int samples);

protected:
    ///
    /// \param q: query
    /// \param L: median of L means
    /// \param m: means of m samples
    /// \return: a vector of L elements, where each element is a sum of m random samples
    ///
    std::vector<double> MoM(VectorXd q, int L, int m);

private:
    ///
    /// Reservoir of random samples. Could be the entire dataset of a subset of the dataset.
    ///
    shared_ptr<MatrixXd> X;

    ///
    /// Number of points in the reservoir.
    ///
    int numPoints;

    ///
    /// Kernel function.
    ///
    shared_ptr<Kernel> kernel;
};


#endif //HBE_RS_H
