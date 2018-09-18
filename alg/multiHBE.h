//
// Created by Kexin Rong on 9/17/18.
//

#ifndef HBE_MRHBE_H
#define HBE_MRHBE_H

#include "HashBucket.h"
#include "HashTable.h"
#include <Eigen/Dense>

using Eigen::VectorXd;

class multiHBE {
public:
    vector<vector<HashTable>> tables;
    vector<double> binWidth;
    vector<int> numHash;
    vector<double> targets;
    vector<double> numSamples;
    int numTables;
    int maxLevel;
    int numPoints;
    shared_ptr<Kernel> kernel;
    int batchSize = 1;
    int idx = 0;
    double delta;

    multiHBE(shared_ptr<MatrixXd> X, shared_ptr<Kernel> ker, int batch,
             double tau, double eps, double del, int threads);
    vector<double> query(VectorXd query);

protected:
    vector<double> evaluateQuery(VectorXd query, double mu, int maxSamples);
    double normalizeConst(double dist, double mu, double delta);


};


#endif //HBE_MRHBE_H
