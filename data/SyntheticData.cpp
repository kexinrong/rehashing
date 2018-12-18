//
// Created by Kexin Rong on 9/5/18.
//

#include "SyntheticData.h"
#include <iostream>

GenericInstance SyntheticData::genMixed(int uN, int cN, int uC, int cC, int dim, double density,
        int numScales, double spread, shared_ptr<Kernel> kernel) {
    // Uncorrelated instance
    GenericInstance uncorrelated(uN, uC, numScales, dim, density, spread, kernel);
    // Correlated instance
    GenericInstance correlated(cN, cC, numScales, dim, density, spread, kernel);
    // Mixed datasets
    uncorrelated.merge(correlated.points);

    return uncorrelated;
}

GenericInstance SyntheticData::genSingle(int pts, int clusters, int dim, double density,
                                        int numScales, double spread, shared_ptr<Kernel> kernel) {
    // instance
    GenericInstance inst(pts, clusters, numScales, dim, density, spread, kernel);

    return inst;
}

GenericInstance SyntheticData::genUncorrelated(int n) {
    // Uncorrelated instance
    int numClusters = (int)ceil(sqrt(n));
    int numScales = 1;
    int dim = (int)ceil(pow(log(sqrt(n)), 3));
    double density = 1.0 / sqrt(n);
    double spread = 1.0 / log(1.0 / density);
    int numPoints = (int)ceil(sqrt(n));

    GenericInstance uncorrelated(numPoints, numClusters, numScales, dim, density, spread);
    return uncorrelated;
}