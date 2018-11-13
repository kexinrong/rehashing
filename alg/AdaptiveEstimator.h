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
    int L = 3;
    std::vector<double> mui;
    std::vector<int> Mi;
    std::vector<double> ti;
    std::vector<int> ki;
    std::vector<double> wi;

    std::vector<double> query(VectorXd q) {
        auto t1 = std::chrono::high_resolution_clock::now();
        std::vector<double> returns(2, 0);
        double est = 0;
        int i = 0;
        while (i < I) {
            std::vector<double> results = evaluateQuery(q, i, Mi[i]);
            est = results[0];
            returns[1] += Mi[i];

//            std::cout << "Level: " << i << ", est: "<< est << std::endl;
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

    void setMedians(int l) { L = l; }

protected:
    virtual std::vector<double> evaluateQuery(VectorXd q, int level, int maxSamples) = 0;

};


#endif //HBE_ADAPTIVEESTIMATOR_H
