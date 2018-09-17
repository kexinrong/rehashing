//
// Created by Kexin Rong on 9/10/18.
//

#ifndef HBE_BANDWIDTH_H

#define HBE_BANDWIDTH_H

#include <vector>
#include <math.h>
#include <numeric>
#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;


class Bandwidth {
public:
    double multiplier = 1;
    double scaleFactor = 1;
    int dim;
    std::vector<double> bw;

    Bandwidth(int n, int d) {
        bw = std::vector<double>(d, 1);
        dim = d;
        scaleFactor = pow(n, -1.0/(d+4));
    }

    void useConstant(double h) {
        for (int i = 0; i < dim; i++) { bw[i] = h; }
    }

    void getBandwidth(MatrixXd &X) {
        int n = X.rows();
        for (int i = 0; i < dim; i ++) {
            VectorXd vec = X.col(i);
            std::vector<double> v(vec.data(), vec.data() + vec.size());
            double sum = std::accumulate(std::begin(v), std::end(v), 0.0);
            double m =  sum / n;
            double accum = 0.0;
            std::for_each (std::begin(v), std::end(v), [&](const double d) {
                accum += (d - m) * (d - m);
            });
            double stdev = sqrt(accum / (v.size()-1));
            bw[i] = stdev * scaleFactor * multiplier;
            if (bw[i] == 0) { bw[i] = 1; }
        }
    }
};

#endif //HBE_BANDWIDTH_H
