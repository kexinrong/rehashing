//
// Created by Kexin Rong on 9/4/18.
//

#ifndef HBE_GENERICINSTANCE_H
#define HBE_GENERICINSTANCE_H

#include <Eigen/Dense>
#include <vector>

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

    GenericInstance(int numPoints, int numClusters, int numScales, int dim,
        double density, double spread);

    void merge(MatrixXd data);

    VectorXd query(double dist, bool correlated);
};


#endif //HBE_GENERICINSTANCE_H