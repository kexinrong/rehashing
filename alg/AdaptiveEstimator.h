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
    int numPoints;
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

    std::string EXP_STR = "exp";

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
//        std::cout << "==========================\n";
        return returns;
    }

    void setMedians(int l) { L = l; }

protected:
    virtual std::vector<double> evaluateQuery(VectorXd q, int level) = 0;

};


#endif //HBE_ADAPTIVEESTIMATOR_H
