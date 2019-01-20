#ifndef HBE_GENERICINSTANCE_H
#define HBE_GENERICINSTANCE_H

#include <Eigen/Dense>
#include <random>
#include <memory>
#include "kernel.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class GenericInstance {
private:
    int numPoints;
    int numClusters;
    int numScales;
    int dim;

public:
    std::vector<VectorXd> directions;
    MatrixXd points;
    std::mt19937_64 rng;

    GenericInstance(int numPoints, int numClusters, int numScales, int dim,
        double density, double spread);

    GenericInstance(int numPoints, int numClusters, int numScales, int dim,
                    double density, double spread, shared_ptr<Kernel> kernel);

    void merge(MatrixXd data);

    VectorXd query(double dist, bool correlated);

    void output(std::string fname);
};


#endif //HBE_GENERICINSTANCE_H
