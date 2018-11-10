//
// Created by Kexin Rong on 2018-11-09.
//

#ifndef HBE_ADAPTIVEESTIMATOR_H
#define HBE_ADAPTIVEESTIMATOR_H


#include <Eigen/Dense>
#include "mathUtils.h"
#include <chrono>

using Eigen::VectorXd;

class AdaptiveEstimator {
public:
    double totalTime = 0;
    double r;
    double gamma = 0.5;
    int I;
    std::vector<double> mui;
    std::vector<double> ti;
    std::vector<int> ki;
    std::vector<double> wi;
    std::vector<int> Mi;

    std::vector<double> query(VectorXd q) {
        auto t1 = std::chrono::high_resolution_clock::now();
        std::vector<double> returns(2, 0);
        double est = 0;
        int i = 0;
        while (i < I) {
            int j = 0;
            est = 0;
            while (j < Mi[i]) {
                std::vector<double> results = evaluateQuery(q, i, Mi[i] - j);
                est += results[0];
                j += (int) results[1];
            }
            returns[1] += Mi[i];

            est /= Mi[i];
            if (est >= mui[i]) {
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

protected:
    virtual std::vector<double> evaluateQuery(VectorXd q, int level, int maxSamples) = 0;

};


#endif //HBE_ADAPTIVEESTIMATOR_H
