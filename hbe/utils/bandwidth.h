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

    // IMPORTANT: The code assume that the input dataset is preprocessed
    // such that the standard deviation for each column is 1.
    void useConstant(double h) {
        for (int i = 0; i < dim; i++) { bw[i] = h; }
    }
};

#endif //HBE_BANDWIDTH_H
