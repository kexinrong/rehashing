#ifndef HBE_MOMESTIMATOR_H
#define HBE_MOMESTIMATOR_H


#include <Eigen/Dense>
#include "mathUtils.h"
#include <chrono>

using Eigen::VectorXd;

///
/// Base class for median-of-means estimators.
///
class MoMEstimator {
public:
    ///
    /// Timing variable
    ///
    double totalTime = 0;

    ///
    /// \param q: query
    /// \param lb: lower bound of query density
    /// \param m: means of m samples
    /// \return: estimate of query if it's > lb, otherwise 0
    ///
    double query(VectorXd q, double lb, int m) {
        int L = 1;
        auto t1 = std::chrono::high_resolution_clock::now();
        std::vector<double> Z = MoM(q, L, m);
        double est = mathUtils::median(Z) / m;
        auto t2 = std::chrono::high_resolution_clock::now();
        totalTime += std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
        if (est < lb) {
            return 0;
        } else {
            return est;
        }
    }

protected:
    ///
    /// \param q: query
    /// \param L: median of L means
    /// \param m: means of m samples
    /// \return: a vector of L elements, where each element is a sum of m samples
    ///
    virtual std::vector<double> MoM(VectorXd q, int L, int m) = 0;

};


#endif //HBE_MOMESTIMATOR_H
