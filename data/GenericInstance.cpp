//
// Created by Kexin Rong on 9/4/18.
//

#include "GenericInstance.h"
#include "mathUtils.h"
#include <math.h>
#include <algorithm>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

typedef Eigen::Matrix<double, 1, Eigen::Dynamic> RowVector;

GenericInstance::GenericInstance(int p, int c, int s, int d,
                double density, double spread) {
    srand (time(NULL));

    numPoints = p;
    numClusters = c;
    numScales = std::max(s, 2);
    dim = d;

    directions.clear();
    for (int i = 0; i < c; i ++) {
        VectorXd x = mathUtils::randNormal(d);
        double n = x.norm();
        x /= n;
        directions.push_back(x);
    }

    double dist = mathUtils::inverseExp(density);
    // np.linspace(0, dist, num_scales)
    double scales[numScales];
    double interval = dist / (numScales - 1);
    scales[0] = 0;
    for (int i = 1; i < numScales; i ++) {
        scales[i] = scales[i - 1] + interval;
    }

    // Calculate number of points in each scale such that the contribution
    // of all distance scale is the same up to constant factors
    double sum = 0;
    for (int i = 0; i < numScales; i ++) { sum += 1 / mathUtils::expKernel(scales[i]); }
    int N = 0;
    double clusterSize = numPoints * 1.0 / numClusters;
    int sizes[numScales];
    for (int i = 0; i < numScales; i ++) {
        sizes[i] = (int) round(clusterSize / sum / mathUtils::expKernel(scales[i]));
        if (sizes[i] == 0 && numClusters < 10) { sizes[i] = 1; }
        N += sizes[i];
    }
    N *= numClusters;

    // Generate Points
    int cnt = 0;
    points = MatrixXd::Zero(N, d);
    for (int i = 0; i < numClusters; i ++) {
        for (int j = 0; j < numScales; j ++) {
            double scale = spread * scales[j] / sqrt(dim);
            MatrixXd rand = mathUtils::randNormal(sizes[j], dim) * scale;
            VectorXd vec = directions.at(i) * scales[j];
            RowVector v = vec.transpose();
            for (int k = cnt; k < cnt + sizes[j]; k ++) {
                points.row(k) += rand.row(k - cnt) + v;
            }
            cnt += sizes[j];
        }
    }
}

void GenericInstance::merge(MatrixXd data) {
    assert(data.cols() == dim);

    MatrixXd merged(points.rows() + data.rows(), dim);
    merged << points, data;
    points = merged;
}

VectorXd GenericInstance::query(double dist, bool correlated) {
    if (correlated) {
        int idx = rand() % numClusters;
        return directions.at(idx) * dist;;
    } else {
        return mathUtils::randNormal(dim) * dist / sqrt(dim);
    }
}