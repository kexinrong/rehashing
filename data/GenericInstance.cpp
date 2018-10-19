//
// Created by Kexin Rong on 9/4/18.
//

#include "GenericInstance.h"
#include "mathUtils.h"
#include <math.h>
#include <algorithm>
#include <iostream>
#include <fstream>      // std::ofstream

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::IOFormat;

typedef Eigen::Matrix<double, 1, Eigen::Dynamic> RowVector;

const static IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

GenericInstance::GenericInstance(int p, int c, int s, int d, double density, double spread) {
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    rng = std::mt19937_64(rd());

    numPoints = p;
    numClusters = c;
    numScales = std::max(s, 2);
    dim = d;

    directions.clear();
    for (int i = 0; i < c; i ++) {
        VectorXd x = mathUtils::randNormal(d, rng);
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
    int N = 0;
    int sizes[numScales];
    for (int i = 0; i < numScales; i ++) {
        sizes[i] = (int) floor(numPoints * density / mathUtils::expKernel(scales[i]));
        N += sizes[i];
    }
    N *= numClusters;

    // Generate Points
    int cnt = 0;
    points = MatrixXd::Zero(N, d);
    for (int i = 0; i < numClusters; i ++) {
        for (int j = 0; j < numScales; j ++) {
            double scale = spread * scales[j] / sqrt(dim);
            MatrixXd rand = mathUtils::randNormal(sizes[j], dim, rng) * scale;
            VectorXd vec = directions.at(i) * scales[j];
            points.block(cnt, 0, sizes[j], d) = rand + MatrixXd::Ones(sizes[j], 1) * vec.transpose();
            cnt += sizes[j];
        }
    }
}


GenericInstance::GenericInstance(int p, int c, int s, int d, double density, double spread, shared_ptr<Kernel> kernel) {
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    rng = std::mt19937_64(rd());

    numPoints = p;
    numClusters = c;
    numScales = std::max(s, 2);
    dim = d;

    directions.clear();
    for (int i = 0; i < c; i ++) {
        VectorXd x = mathUtils::randNormal(d, rng);
        double n = x.norm();
        x /= n;
        directions.push_back(x);
    }

    double dist = kernel->invDensity(density);
    // np.linspace(0, dist, num_scales)
    double scales[numScales];
    double interval = dist / (numScales - 1);
    scales[0] = 0;
    for (int i = 1; i < numScales; i ++) {
        scales[i] = scales[i - 1] + interval;
    }

    // Calculate number of points in each scale such that the contribution
    // of all distance scale is the same up to constant factors
    int N = 0;
    int sizes[numScales];
    for (int i = 0; i < numScales; i ++) {
        sizes[i] = (int) floor(numPoints * density / kernel->density(scales[i]));
        N += sizes[i];
    }
    N *= numClusters;

    // Generate Points
    int cnt = 0;
    points = MatrixXd::Zero(N, d);
    for (int i = 0; i < numClusters; i ++) {
        for (int j = 0; j < numScales; j ++) {
            double scale = spread * scales[j] / sqrt(dim);
            MatrixXd rand = mathUtils::randNormal(sizes[j], dim, rng) * scale;
            VectorXd vec = directions.at(i) * scales[j];
            points.block(cnt, 0, sizes[j], d) = rand + MatrixXd::Ones(sizes[j], 1) * vec.transpose();
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
    std::uniform_int_distribution<int> distribution(0, numClusters - 1);
    if (correlated) {
        int idx = distribution(rng);
        return directions.at(idx) * dist;;
    } else {
        return mathUtils::randNormal(dim, rng) * dist / sqrt(dim);
    }
}

void GenericInstance::output(std::string fname) {
    std::ofstream ofile(fname);
    ofile << points.format(CSVFormat);
    ofile << "\n";
    ofile.close();
}
