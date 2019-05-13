#ifndef HBE_ADAPTIVEESTIMATOR_H
#define HBE_ADAPTIVEESTIMATOR_H


#include <Eigen/Dense>
#include "mathUtils.h"
#include <chrono>

using Eigen::VectorXd;

///
/// Base class for the adaptive sampling procedure.
/// Subclasses can be implemented via estimators like HBE and RS.
///
class AdaptiveEstimator {
public:
    double totalTime = 0;

    ///
    /// Estimate density of query q via adaptively sampling.
    ///
    std::vector<double> query(VectorXd q) {
        auto t1 = std::chrono::high_resolution_clock::now();
        std::vector<double> returns(2, 0);
        double est = 0;
        int i = 0;
        while (i < I) {
            std::vector<double> results = evaluateQuery(q, i);
            est = results[0];
            returns[1] += results[1];
//            std::cout << "Level: " << i << ", est: "<< est << ", target: "<< mui[i] << std::endl;
            if (est >= mui[i] || L * Mi[i] > numPoints) {
                break;
            } else {
                int k = (int) floor(log(est) / log(1 - gamma));
                i = std::max(k, i + 1);
            }
        }
        returns[0] = est;
        auto t2 = std::chrono::high_resolution_clock::now();
        totalTime += std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
        return returns;
    }

    void setMedians(int l) { L = l; }

protected:
    int numPoints;
    ///
    /// decay rate of target density between each level
    ///
    double gamma = 0.5;

    ///
    /// number of levels
    ///
    int I;

    ///
    /// median of L means
    ///
    int L = 3;

    ///
    /// target density of the level i
    ///
    std::vector<double> mui;

    ///
    /// # samples for level i
    ///
    std::vector<int> Mi;

    std::vector<double> ti;

    ///
    /// hashing scheme parameter for level i: # hash functions
    ///
    std::vector<int> ki;

    ///
    /// hashing scheme parameter for level i: binWidth
    ///
    std::vector<double> wi;

    ///
    /// Effective diameter sqrt(log(1/ tau))
    ///
    double r;

    std::string EXP_STR = "exp";
    ///
    /// For subclasses to implement: evaluate density of query q at the given level.
    ///
    virtual std::vector<double> evaluateQuery(VectorXd q, int level) = 0;
};


#endif //HBE_ADAPTIVEESTIMATOR_H
