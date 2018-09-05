//
// Created by Kexin Rong on 9/5/18.
//

#include "SyntheticData.h"

GenericInstance SyntheticData::genMixed(int n, int dim, double density, int uncorrelatedClusters,
                         int correlatedClusters, int numScales, double spread) {
    // Uncorrelated instance
    GenericInstance uncorrelated(n, uncorrelatedClusters, numScales, dim, density, spread);
    // Correlated instance
    GenericInstance correlated(n, correlatedClusters, numScales, dim, density, spread);
    // Mixed datasets
    uncorrelated.merge(correlated.points);

    return uncorrelated;
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