//
// Created by Kexin Rong on 9/5/18.
//

#ifndef HBE_BASELSH_H
#define HBE_BASELSH_H

#include <Eigen/Dense>
#include "kernel.h"
#include "HashBucket.h"
#include "HashTable.h"
#include "MoMEstimator.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class BaseLSH : public MoMEstimator {
public:
    int numTables;
    double binWidth;
    int numHash;
    int numPoints;
    Kernel *kernel;
    int batchSize = 100;
    int idx = 0;

    BaseLSH(MatrixXd X, int M, double w, int k, int batch);

protected:
    virtual double* evaluateQuery(VectorXd query, int maxSamples) = 0;

    double* evaluate(std::vector<HashBucket> buckets, VectorXd query, int maxSamples);
    double* MoM(VectorXd query, int L, int m);

};


#endif //HBE_BASELSH_H
